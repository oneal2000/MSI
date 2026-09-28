"""Synthetic task generation (question-only, schema-validated)."""
from openai import OpenAI
import json
import os
import random
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from msi.generation.artifacts import TASK_PROGRESS_SCHEMA
from msi.models.config import EXTERNAL_DIR
from msi.models.llm_client import _chat
from msi.protocol import TEACHER_REQUEST_MAX_TOKENS
from msi.generation.prompts import DIFFICULTY_PROMPTS
from msi.generation.schemas import build_format_instruction, validate_instance
from msi.generation.teacher import teacher_extra_body


def _synthesize_retrieval_demo(action_type: str, env, corpus_dir) -> tuple[str, str] | None:
    """Synthesize one real retrieval-tool call when the skill text only has placeholder
    examples (e.g. a SciREX skill with just ``RetrieveScirex[keyword]``).

    This exposes the tool's actual return schema to the task generator and
    prevents questions that require fields the retrieval source cannot return.
    """
    import json as _json
    import random as _random
    import re as _re
    if action_type == "RetrieveScirex":
        path = corpus_dir / "scirex" / "Preprocessed_Scirex.jsonl"
        field = "content"
    elif action_type == "RetrieveAgenda":
        path = corpus_dir / "agenda" / "agenda_descriptions_merged.jsonl"
        field = "event"
    else:
        return None
    if not path.exists():
        return None
    lines = [ln for ln in path.read_text(errors="ignore").strip().split("\n") if ln.strip()]
    if not lines:
        return None
    sample = _random.sample(lines, min(8, len(lines)))
    _STOP = {"the", "this", "these", "those", "what", "when", "where", "which", "that", "then"}
    keyword = None
    for ln in sample:
        try:
            txt = _json.loads(ln).get(field, "") or ""
        except Exception:
            continue
        # capitalized phrase = likely a method/dataset/entity name
        cands = _re.findall(r"\b[A-Z][A-Za-z]+(?:[\s\-][A-Za-z0-9]+){0,3}\b", txt[:400])
        cands = [c.strip() for c in cands if len(c.strip()) >= 4 and c.split()[0].lower() not in _STOP]
        if cands:
            keyword = cands[0]
            break
    if not keyword:
        keyword = "neural network"  # generic — still returns prose passages
    return (action_type, keyword)


# Tools whose real call can be synthesized from the skill's DB/graph (grounded in real data).
_DB_TOOLS = {"FilterDB", "GetValue", "SQLInterpreter"}
_TOOL_RE = __import__("re").compile(
    r"(LoadDB|FilterDB|GetValue|Calculate|LoadGraph|NeighbourCheck|NodeCheck|"
    r"EdgeCheck|SQLInterpreter|RetrieveAgenda|RetrieveScirex|PythonInterpreter)")


def _db_column_and_value(corpus_dir, db: str) -> tuple[str | None, str | None]:
    """Load `db` via TableToolkit, return (a real column, a real value from row 0)."""
    try:
        from sragents.toolqa.tools.table import TableToolkit
        tk = TableToolkit(corpus_dir); tk.load_db(db); df = tk.data
        if df is None or len(df.columns) == 0:
            return None, None
        col = None
        for c in df.columns:  # prefer a data column, skip id-like
            if c.lower() not in ("id", "index", "_id") and len(c) > 1:
                col = c; break
        col = col or df.columns[0]
        val = str(df[col].iloc[0]) if len(df) else None
        return col, val
    except Exception:
        return None, None


def _synthesize_tool_demo(
    skill_content: str, mentioned: set, have: set, env, corpus_dir
) -> list[tuple[str, str]]:
    """For tools the skill USES (`mentioned`) but the demo LACKS (`have`), synthesize REAL
    calls grounded in the actual DB data (real columns/values, correct formats). Returns an
    ordered [(action, arg), ...] list; LoadDB is prepended for stateful dependents
    (FilterDB/GetValue/SQLInterpreter need the DB loaded first in the same env).

    Supported return formats:
      LoadDB[db] · FilterDB[col=val] · GetValue[col] · SQLInterpreter[SELECT col FROM {db}_data ...]
      · RetrieveScirex/RetrieveAgenda[keyword] · PythonInterpreter[ans = <code>]
    """
    import re as _re
    out: list[tuple[str, str]] = []
    dbs = _re.findall(r"LoadDB\[(\w+)\]", skill_content)
    db = dbs[0] if dbs else None
    # --- DB tools: ensure LoadDB runs first (stateful), then a real-col Filter/Get/SQL ---
    if db:
        load_added = "LoadDB" in have
        for t in ("FilterDB", "GetValue", "SQLInterpreter"):
            if t in mentioned and t not in have:
                if not load_added:
                    out.append(("LoadDB", db)); load_added = True
                col, val = _db_column_and_value(corpus_dir, db)
                if not col:
                    continue
                if t == "FilterDB" and val is not None:
                    out.append(("FilterDB", f"{col}={val}"))
                elif t == "GetValue":
                    out.append(("GetValue", col))
                elif t == "SQLInterpreter":
                    out.append(("SQLInterpreter", f"SELECT {col} FROM {db}_data LIMIT 3"))
    # --- PythonInterpreter: canonical "assign to ans" example (tool prints `ans`) ---
    if "PythonInterpreter" in mentioned and "PythonInterpreter" not in have:
        out.append(("PythonInterpreter", "ans = len([1, 2, 3])"))
    # --- retrieval (reuse the SciREX/agenda synthesizer) ---
    for t in ("RetrieveScirex", "RetrieveAgenda"):
        if t in mentioned and t not in have:
            s = _synthesize_retrieval_demo(t, env, corpus_dir)
            if s:
                out.append(s)
    return out


def _demonstrate_tool_workflow(skill_content: str, max_calls: int = 6) -> str:
    """Execute tool calls from the original skill text and show real input→output.

    Parses tool calls from the skill's example section, executes them
    via ToolEnvironment.execute(), and returns a formatted demo showing
    each step's input and output.  This lets the generator see exactly
    what each tool returns and understand their capabilities and
    limitations.

    Works uniformly for all tool types (DB, Graph, Text retrieval)
    because ToolEnvironment.execute() handles all dispatch internally.
    """
    import re as _re
    from sragents.toolqa.tools import ToolEnvironment

    corpus_dir = EXTERNAL_DIR / "toolqa"
    env = ToolEnvironment(corpus_dir)

    # Extract tool calls from examples in the original skill text.
    # Handles nested brackets for PythonInterpreter etc.
    tool_calls = _re.findall(
        r'(LoadDB|FilterDB|GetValue|Calculate|LoadGraph|NeighbourCheck|'
        r'NodeCheck|EdgeCheck|SQLInterpreter|RetrieveAgenda|RetrieveScirex)'
        r'\[([^\]]*(?:\[[^\]]*\])*[^\]]*)\]',
        skill_content,
    )

    # Deduplicate and skip template placeholders.
    # Known placeholder patterns:
    #   - YYYY-MM-DD           (date template)
    #   - ColumnName, DBName   (all-uppercase CamelCase identifiers)
    #   - <paper title>        (angle-bracket templates)
    #   - keyword, formula     (lowercase single-word in certain contexts)
    # But NOT real values like: coffee, flights, airbnb, dblp, author names, SQL
    seen = set()
    filtered = []
    _PLACEHOLDER_RE = _re.compile(
        r'^[A-Z][a-zA-Z_]*[A-Z_][A-Za-z_]*$'   # CamelCase: ColumnName, DBName, GraphName
        r'|^<.*>$'                                # <paper title>, <author name>
    )
    # Placeholder patterns that can appear INSIDE arguments (e.g. Date=YYYY-MM-DD)
    _PLACEHOLDER_CONTAINS = _re.compile(
        r'[A-Z]{4}-[A-Z]{2}-[A-Z]{2}'           # YYYY-MM-DD
        r'|<[A-Za-z\s]+>'                        # <anything>
    )
    # Single lowercase word is a placeholder ONLY for retrieval/calculate tools
    # where the argument is clearly generic (keyword, formula, condition)
    _RETRIEVAL_PLACEHOLDERS = {"keyword", "formula", "condition", "query"}
    for action_type, argument in tool_calls:
        key = (action_type, argument)
        if key in seen:
            continue
        seen.add(key)
        arg_stripped = argument.strip()
        if _PLACEHOLDER_RE.match(arg_stripped):
            continue
        if _PLACEHOLDER_CONTAINS.search(arg_stripped):
            continue
        if arg_stripped in _RETRIEVAL_PLACEHOLDERS:
            continue
        filtered.append((action_type, argument))

    if not filtered:
        # Every tool the skill USES should get a real-return example so the task generator
        # knows exactly what each tool can return (and thus what questions are answerable).
        # Fill missing demonstrations with calls grounded in the actual data.
        pass  # handled uniformly below regardless of whether filtered is empty
    mentioned = set(_TOOL_RE.findall(skill_content))
    have = {a for a, _ in filtered}
    filtered.extend(_synthesize_tool_demo(skill_content, mentioned, have, env, corpus_dir))
    # One example per tool (concrete example first; synthesized gap-fill otherwise) so every
    # Keep one concrete demonstration per tool.
    _seen: set[str] = set()
    filtered = [(a, v) for a, v in filtered if not (a in _seen or _seen.add(a))]
    if not filtered:
        return ""

    lines = [
        "\n### Tool Workflow Demonstration (real tool calls with actual outputs)",
        "Below are actual tool calls extracted from the skill examples, "
        "executed against the real database. Use these to understand each "
        "tool's capabilities, return format, and limitations.\n",
    ]

    for action_type, argument in filtered[:max_calls]:
        action_str = f"{action_type}[{argument}]"
        try:
            output = env.execute(action_str)
            if len(output) > 400:
                output = output[:400] + "... (truncated)"
            lines.append(f"{action_str}")
            lines.append(f"  → {output}")
            lines.append("")
        except Exception as e:
            lines.append(f"{action_str}")
            lines.append(f"  → ERROR: {e}")
            lines.append("")

    return "\n".join(lines)
def _sample_toolqa_entities(skill_content: str, n: int = 5) -> str:
    """Sample real entities from ToolQA databases to ground generated questions.

    Detects which data sources the skill references by parsing tool calls
    (LoadDB[x], LoadGraph[x], RetrieveAgenda[...], RetrieveScirex[...])
    directly from the original skill text, then samples representative rows.
    """
    import re as _re
    from sragents.toolqa.tools.table import TableToolkit
    from sragents.toolqa.tools.graph import GraphToolkit

    corpus_dir = EXTERNAL_DIR / "toolqa"
    samples = []

    db_names = set(_re.findall(r'LoadDB\[(\w+)\]', skill_content))
    graph_names = set(_re.findall(r'LoadGraph\[(\w+)\]', skill_content))
    uses_agenda = bool(_re.search(r'RetrieveAgenda\[', skill_content))
    uses_scirex = bool(_re.search(r'RetrieveScirex\[', skill_content))
    if not uses_agenda and 'agenda_descriptions' in skill_content:
        uses_agenda = True
    if not uses_scirex and 'Preprocessed_Scirex' in skill_content:
        uses_scirex = True

    db_map = {
        "flights": "flights", "coffee": "coffee",
        "airbnb": "airbnb", "yelp": "yelp",
    }
    resolved_dbs = {db_map.get(db.lower(), db.lower()) for db in db_names}

    # Prioritized columns per database so that frequently-queried fields
    # (stars, categories, price, Flight_Number, ArrDelay, etc.) appear in
    # the sampled row data even if they sit beyond the first 8 positions.
    _PRIORITY_COLS = {
        "flights": [
            "FlightDate", "Airline", "Origin", "Dest",
            "Flight_Number_Marketing_Airline", "Operating_Airline",
            "DepDelay", "ArrDelay", "AirTime", "Distance",
            "Cancelled", "CRSDepTime", "ArrTime", "OriginCityName", "DestCityName",
        ],
        "yelp": [
            "name", "address", "city", "state",
            "stars", "review_count", "categories", "is_open",
            "latitude", "longitude", "postal_code",
        ],
        "airbnb": [
            "NAME", "host name", "neighbourhood group", "neighbourhood",
            "room type", "price", "service fee", "Construction year",
            "availability 365", "number of reviews", "cancellation_policy",
        ],
    }
    _MAX_SHOW_COLS = 15  # show up to this many columns in sampled rows

    table = TableToolkit(corpus_dir)
    for db_name in sorted(resolved_dbs):
        try:
            table.load_db(db_name)
            df = table.data
            sample_rows = df.sample(n=min(n, len(df)), random_state=None)
            cols = df.columns.tolist()
            # Build show_cols: prioritized list first, then fill with remaining
            priority = _PRIORITY_COLS.get(db_name, [])
            show_cols = [c for c in priority if c in cols]
            remaining = [c for c in cols if c not in show_cols]
            show_cols.extend(remaining[: max(0, _MAX_SHOW_COLS - len(show_cols))])
            samples.append(f"### {db_name} — {len(df)} rows total, showing {len(show_cols)}/{len(cols)} columns")
            samples.append(f"Columns: {', '.join(cols)}")
            for _, row in sample_rows.iterrows():
                vals = ", ".join(f"{c}={row[c]}" for c in show_cols)
                samples.append(f"  {vals}")
        except Exception:
            pass

    if uses_agenda:
        try:
            import json as _json
            agenda_path = corpus_dir / "agenda" / "agenda_descriptions_merged.jsonl"
            if agenda_path.exists():
                lines = agenda_path.read_text().strip().split("\n")
                sampled = random.sample(lines, min(n, len(lines)))
                samples.append(f"### Agenda events — {len(lines)} events total")
                for line in sampled:
                    obj = _json.loads(line)
                    samples.append(f"  {obj.get('event', '')[:200]}")
        except Exception:
            pass

    if uses_scirex:
        try:
            import json as _json
            scirex_path = corpus_dir / "scirex" / "Preprocessed_Scirex.jsonl"
            if scirex_path.exists():
                lines = scirex_path.read_text().strip().split("\n")
                passages = []
                random.shuffle(lines)
                for line in lines:
                    obj = _json.loads(line)
                    content = str(obj.get("content", "")).strip()
                    if len(content) >= 300:
                        passages.append(content)
                    if len(passages) >= n:
                        break
                samples.append(f"### SciREX passages — {len(lines)} passages total")
                for passage in passages:
                    # A short prefix often contains only setup and cuts off the
                    # concrete result. Give the task model enough source text to
                    # formulate a question whose answer is actually retrievable.
                    samples.append(f"  {passage[:1200]}")
        except Exception:
            pass

    if "dblp" in graph_names:
        try:
            graph = GraphToolkit(corpus_dir)
            graph.load_graph("dblp")
            pn = graph.paper_net
            an = graph.author_net

            # --- Sample papers with full metadata ---
            # Build candidate lists once (the graphs are huge so we avoid
            # rebuilding on every call within a single session).
            if not hasattr(_sample_toolqa_entities, '_dblp_full_papers'):
                _sample_toolqa_entities._dblp_full_papers = [
                    (nid, graph.id2title_dict.get(nid, ""), pn.nodes[nid])
                    for nid in pn.nodes
                    if graph.id2title_dict.get(nid, "")
                    and pn.nodes[nid].get("authors")
                    and pn.nodes[nid].get("year")
                    and pn.nodes[nid].get("venue", {}).get("raw")
                ]
            full_papers = _sample_toolqa_entities._dblp_full_papers
            n_paper = min(n, len(full_papers))
            if n_paper:
                sampled_papers = random.sample(full_papers, n_paper)
                samples.append(
                    f"### DBLP PaperNet — {len(full_papers)} papers with full metadata, "
                    f"samples with title, authors, year, venue:"
                )
                for _, title, attrs in sampled_papers:
                    author_names = [a.get("name", "") for a in attrs["authors"][:3]]
                    venue = attrs["venue"]["raw"]
                    samples.append(
                        f'  Title: "{title}"\n'
                        f'    Authors: {", ".join(author_names)}'
                        f'{", ..." if len(attrs["authors"]) > 3 else ""}\n'
                        f'    Year: {attrs["year"]}, Venue: {venue}, '
                        f'Citations: {attrs.get("n_citation", 0)}'
                    )

            # --- Sample authors with org + collaborators ---
            # These are the entities the task model should use when posing
            # questions about affiliations, collaboration counts, mutual
            # collaborators, etc.
            if not hasattr(_sample_toolqa_entities, '_dblp_usable_authors'):
                _sample_toolqa_entities._dblp_usable_authors = [
                    (nid, graph.id2author_dict.get(nid, ""), an.nodes[nid])
                    for nid in an.nodes
                    if an.nodes[nid].get("org")
                    and an.degree(nid) >= 1
                ]
            usable_authors = _sample_toolqa_entities._dblp_usable_authors
            n_author = min(n, len(usable_authors))
            if n_author:
                sampled_authors = random.sample(usable_authors, n_author)
                n_authors_total = an.number_of_nodes()
                samples.append(
                    f"### DBLP AuthorNet — {n_authors_total} authors total, "
                    f"samples with affiliation and collaborators:"
                )
                for nid, name, attrs in sampled_authors:
                    nbr_ids = list(an.neighbors(nid))
                    nbr_names = [
                        graph.id2author_dict.get(nb, "??") for nb in nbr_ids[:4]
                    ]
                    # Grab edge details for first collaborator
                    edge_info = ""
                    if nbr_ids:
                        edge_data = an.edges[nid, nbr_ids[0]]
                        edge_papers = edge_data.get("papers", [])
                        if edge_papers:
                            # Resolve paper titles
                            paper_titles = [
                                graph.id2title_dict.get(p, p)
                                for p in edge_papers[:2]
                            ]
                            edge_info = (
                                f', shared papers: {paper_titles}'
                            )
                    samples.append(
                        f"  Author: {name}\n"
                        f"    Affiliation: {attrs['org']}\n"
                        f"    Collaborators ({len(nbr_ids)}): "
                        f'{", ".join(nbr_names)}'
                        f'{"..." if len(nbr_ids) > 4 else ""}'
                        f"{edge_info}"
                    )

            # --- IMPORTANT: only-use-listed-entities instruction ---
            if full_papers or usable_authors:
                samples.append(
                    "\nIMPORTANT: Only use author names, paper titles, and "
                    "other entities listed above (or clearly derivable from "
                    "them, e.g. other collaborators of a listed author) in "
                    "your questions. Do NOT use well-known researcher names "
                    "(e.g. Yann LeCun, Fei-Fei Li, Geoffrey Hinton) or "
                    "famous paper titles (e.g. 'Attention Is All You Need', "
                    "'BERT') from your training data — they are NOT in this "
                    "graph snapshot."
                )
        except Exception:
            pass

    if not samples:
        return ""
    header = (
        "\n## Real Data from Underlying Databases\n"
        "Below are sampled records from the databases this skill queries. "
        "Use these concrete entities (names, dates, IDs, values) to GROUND your question "
        "in real, queryable data.\n"
        "CRITICAL: The person answering the question does NOT have direct access to this data. "
        "They must call tools (LoadDB, FilterDB, GetValue, etc.) to retrieve any information "
        "from the databases. Therefore:\n"
        "- Do NOT embed data values from this sample as given facts in the question.\n"
        "- Design the question so that the answer must be DISCOVERED through tool calls.\n"
        "- You may reference entity names or IDs that exist in the data as search targets, "
        "but never reveal the actual answer values that come from the database.\n"
    )

    # --- Demonstrate tool workflow from skill examples ---
    demo = _demonstrate_tool_workflow(skill_content)
    if demo:
        samples.append(demo)

    return header + "\n".join(samples)
def _assign_difficulties(n: int) -> list[str]:
    """Distribute n tasks across difficulty levels.

    Uses a 30-40-30 split (easy-medium-hard). For very small n,
    distributes as evenly as possible starting with medium.
    """
    if n <= 0:
        return []
    if n == 1:
        return ["medium"]
    if n == 2:
        return ["easy", "hard"]

    n_easy = max(1, round(n * 0.30))
    n_hard = max(1, round(n * 0.30))
    n_medium = n - n_easy - n_hard
    if n_medium < 1:
        n_medium = 1
        if n_easy > 1:
            n_easy -= 1
        elif n_hard > 1:
            n_hard -= 1
    levels = (["easy"] * n_easy) + (["medium"] * n_medium) + (["hard"] * n_hard)
    # Shuffle so difficulty isn't correlated with generation order
    random.shuffle(levels)
    return levels
def _build_task_prompts(
    skill_content: str,
    dataset: str,
    skill: dict | None = None,
    prev_questions: list[str] | None = None,
    num_tasks: int = 1,
    is_batch: bool = False,
    difficulty: str | None = None,
) -> tuple[str, str]:
    """Build (system, user) prompts for task generation.

    All datasets now output question-only (no eval_data).
    System prompt holds quality rules, difficulty, and question format constraints.
    User prompt holds the original corpus skill and previous questions.
    """
    format_instruction = build_format_instruction(dataset, skill)
    prev_block = ""
    if prev_questions:
        prev_block = (
            "\n## Previously Generated Questions (avoid duplication):\n"
            + "\n".join(f"- {q[:200]}" for q in prev_questions[-5:])
        )

    # Inject difficulty-specific instruction if provided
    difficulty_block = ""
    if difficulty and difficulty in DIFFICULTY_PROMPTS:
        difficulty_block = "\n" + DIFFICULTY_PROMPTS[difficulty] + "\n"

    count_instruction = "a question" if num_tasks <= 1 else f"{num_tasks} questions"

    # For ToolQA, show the agent's complete tool environment so the generator
    # understands the full tool vocabulary + semantics (e.g. FilterDB takes
    # compound conditions) and avoids questions that no tool combination can solve.
    tool_env = ""
    answerability = ""
    if dataset == "toolqa":
        from sragents.toolqa.prompts import REACT_INSTRUCTION
        tool_env = f"\n## Available Tools (the agent's complete toolset)\n{REACT_INSTRUCTION}\n"
        answerability = (
            "- The question must have a concrete factual answer obtainable from "
            "actual tool output. Verify the answer against the supplied real data "
            "before writing the question.\n"
        )
        if any(
            tool in skill_content
            for tool in ("RetrieveScirex[", "RetrieveAgenda[")
        ):
            answerability += (
                "- For retrieval questions, ground the question in one of the supplied real "
                "passages and ask only for a fact explicitly stated in that "
                "passage. Include enough distinctive context for that passage to "
                "be recovered by retrieval.\n"
            )

    system = (
        "You are a training data generator for an AI skill system.\n"
        "Generate questions that require applying the given skill's methods.\n\n"
        "## Quality Requirements\n"
        "- The problem must genuinely require the skill — no shortcuts or bypasses.\n"
        "- The question must not exceed the skill's demonstrated scope — require\n"
        "  only methods the skill actually shows, and do not extrapolate\n"
        "  capabilities beyond its examples. Universal primitives any solver has\n"
        "  (arithmetic, comparison, formatting) may be composed in freely, but\n"
        "  cannot substitute for the skill.\n"
        "- Vary parameters, contexts, and sub-methods across tasks.\n"
        f"{answerability}"
        "- NO skill/tool names in questions. NO copied phrases from skill text.\n"
        "- Do NOT compute or include the answer. Only generate the question.\n"
        f"{difficulty_block}\n"
        f"{format_instruction}\n\n"
        "Output ONLY the question text, nothing else. No JSON, no labels, no explanation."
    )

    user = (
        f"Generate {count_instruction} based on this skill.\n\n"
        f"## Skill\n{skill_content}\n"
        f"{tool_env}"
        f"{prev_block}"
    )

    return system, user
def _generate_single_task(
    client: OpenAI, model: str, skill_content: str, dataset: str,
    skill: dict | None = None, prev_questions: list[str] | None = None,
    tools: list[dict] | None = None, delay: float = 0, temperature: float = 0.7,
    difficulty: str | None = None,
) -> dict | None:
    """Generate one task: question text only, no eval_data."""
    system, user = _build_task_prompts(
        skill_content, dataset, skill, prev_questions, num_tasks=1, is_batch=False,
        difficulty=difficulty,
    )

    # ToolQA: inject sampled database entities to ground the question
    if dataset == "toolqa":
        entity_block = _sample_toolqa_entities(skill_content)
        if entity_block:
            user += entity_block

    if delay > 0:
        time.sleep(delay)

    extra_body = teacher_extra_body(model)

    content = _chat(client, model, system, [{"role": "user", "content": user}],
                    max_tokens=TEACHER_REQUEST_MAX_TOKENS,
                    temperature=temperature, extra_body=extra_body)

    question = content.strip().strip('"').strip("'")
    if not question or len(question) < 10:
        print(f"    Question too short or empty: {question[:100]}", flush=True)
        return None

    task = {"question": question}
    ok, reason = validate_instance(dataset, task)
    if not ok:
        print(f"    Validation failed: {reason}", flush=True)
        return None

    return task
def _worker_generate_task(args: tuple) -> tuple[int, dict | None]:
    """Worker function for parallel task generation. Returns (index, task)."""
    idx, client, model, skill_content, dataset, skill, prev_questions, delay, difficulty = args
    task = _generate_single_task(client, model, skill_content, dataset, skill, prev_questions,
                                 delay=delay, difficulty=difficulty)
    return (idx, task)
def generate_tasks_parallel(
    client: OpenAI, model: str, skill_content: str, num_tasks: int,
    dataset: str, skill: dict | None = None,
    num_workers: int = 4, delay: float = 0,
    temperature: float = 0.7,
    difficulties: list[str] | None = None,
    progress_path: str | None = None,
    avoid_questions: list[str] | None = None,
    admission=None,
    requests: list[tuple[int, int]] | None = None,
) -> list[dict]:
    """Generate the requested task attempts and append every outcome.

    ``requests`` contains ``(request_id, attempt)`` pairs selected by the
    orchestrator.  Recovery policy therefore lives in one place: callers may
    omit prior failures or explicitly schedule their next attempt.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    requests = requests or [(index, 1) for index in range(num_tasks)]
    if not requests:
        return []
    print(
        f"  Using {num_workers} parallel workers for {len(requests)} task attempts",
        flush=True,
    )

    tasks: list[dict] = []
    lock = threading.Lock()
    completed_count = [0]
    failed_count = [0]
    _progress_file = open(progress_path, "a") if progress_path else None

    def update_progress(
        request_id: int, attempt: int, task: dict | None,
        reason: str = "generation failed",
    ):
        with lock:
            if task:
                tasks.append(task)
                completed_count[0] += 1
                row = {
                    "schema": TASK_PROGRESS_SCHEMA,
                    "request_id": request_id, "attempt": attempt,
                    "status": "success", "task": task,
                }
                if _progress_file:
                    _progress_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    _progress_file.flush()
                processed = completed_count[0] + failed_count[0]
                print(
                    f"  [{processed}/{len(requests)}] Request {request_id} "
                    f"attempt {attempt} OK: {task['question'][:50]}...",
                    flush=True,
                )
            else:
                failed_count[0] += 1
                row = {
                    "schema": TASK_PROGRESS_SCHEMA,
                    "request_id": request_id, "attempt": attempt,
                    "status": "fail", "reason": reason,
                }
                if _progress_file:
                    _progress_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    _progress_file.flush()
                processed = completed_count[0] + failed_count[0]
                print(
                    f"  [{processed}/{len(requests)}] Request {request_id} "
                    f"attempt {attempt} failed: {reason[:160]}",
                    flush=True,
                )

    def worker(args):
        request_id, attempt, client, model, skill_content, dataset, skill, delay, temperature, difficulty = args
        try:
            with lock:
                prev_qs = list(shared_prev) if shared_prev else None
            task = _generate_single_task(
                client, model, skill_content, dataset, skill, prev_qs,
                delay=delay, temperature=temperature, difficulty=difficulty,
            )
            if task and admission:
                accepted, reason = admission(task)
                if not accepted:
                    return (request_id, attempt, None, reason)
            if task:
                with lock:
                    shared_prev.append(task["question"])
            return (request_id, attempt, task, "generation failed")
        except Exception as error:
            return (
                request_id, attempt, None,
                f"WORKER_ERROR: {type(error).__name__}: {str(error)[:200]}",
            )

    if difficulties is None:
        difficulties = _assign_difficulties(len(requests))
    shared_prev = [
        str(question) for question in (avoid_questions or [])
        if str(question).strip()
    ]
    worker_args = [
        (
            request_id, attempt, client, model, skill_content, dataset, skill,
            delay, temperature, difficulties[position],
        )
        for position, (request_id, attempt) in enumerate(requests)
    ]

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(worker, args): args[0] for args in worker_args}
            for future in as_completed(futures):
                request_id, attempt, task, reason = future.result()
                update_progress(request_id, attempt, task, reason)
    finally:
        if _progress_file:
            _progress_file.close()

    print(
        f"  Attempt batch complete: {len(tasks)} valid, {failed_count[0]} failed",
        flush=True,
    )
    return tasks
def generate_tasks(
    client: OpenAI, model: str, skill_content: str, num_tasks: int,
    dataset: str, skill: dict | None = None,
    delay: float = 0, num_workers: int = 4, use_parallel: bool = True,
    temperature: float = 0.7,
    progress_path: str | None = None,
    avoid_questions: list[str] | None = None,
    admission=None,
    requests: list[tuple[int, int]] | None = None,
) -> list[dict]:
    """Generate diverse synthetic tasks (question-only)."""
    difficulties = _assign_difficulties(num_tasks)

    if use_parallel:
        return generate_tasks_parallel(
            client, model, skill_content, num_tasks, dataset, skill,
            num_workers=num_workers, delay=delay, temperature=temperature,
            difficulties=difficulties, progress_path=progress_path,
            avoid_questions=avoid_questions, admission=admission,
            requests=requests,
        )

    # Sequential single-task generation (fallback)
    tasks: list[dict] = []
    prev_questions = list(avoid_questions or [])
    for i in range(num_tasks):
        print(f"  [task {i+1}/{num_tasks}] generating (difficulty: {difficulties[i]})...", flush=True)
        task = _generate_single_task(
            client, model, skill_content, dataset, skill, prev_questions,
            delay=delay, temperature=temperature, difficulty=difficulties[i],
        )
        if task:
            tasks.append(task)
            prev_questions.append(task["question"])
        else:
            print(f"  [task {i+1}/{num_tasks}] skipped", flush=True)
    return tasks
