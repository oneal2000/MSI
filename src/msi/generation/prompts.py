"""Prompt constants for task/trajectory generation and evaluation."""
from sragents.toolqa.prompts import REACT_INSTRUCTION as _TOOLQA_REACT_INSTRUCTION


TRAJECTORY_QUALITY_SYSTEM = """You are solving the problem. The provided skill describes a method that may
help; use it where it applies. Produce a complete, fully-worked solution.

- When the skill's method can solve the problem, use it strictly — do not
  switch to a different method, and do not extrapolate to tools or procedures
  the skill does not teach. Universal primitives (direct reasoning over small
  sets, basic arithmetic, comparison) execute the method; they do not replace
  it. Reserve a different method only for problems the skill genuinely cannot
  solve. Apply the method as your own reasoning, without quoting or naming the
  skill. In your response, never mention or refer back to the skill, its text,
  provided reference material, or these instructions.
- Reason naturally and coherently through the problem, step by step, writing out
  each step of reasoning and its intermediate result. Do not skip or combine
  steps, in computation or in reasoning, and carry the method through to
  completion: every step that derives the answer must be present and actually
  performed — do not stop at an intermediate result or declare the answer
  absent before the method's steps are exhausted.
- Verify error-prone steps as you go: when a computation, an assignment, or a
  case choice could be wrong, briefly confirm it (substitute back, recount, or
  check the magnitude) before moving on.
- Do not use "obviously" or "clearly" in place of a step.
- Ensure the final result is unambiguous and directly extractable: for code
  tasks, output the complete function body; for other tasks, state the answer
  clearly on its own line.
"""
DIFFICULTY_PROMPTS = {
    "easy": (
        "## Difficulty Target: EASY\n"
        "Generate a task requiring straightforward, single-concept application of the skill:\n"
        "- Provide all necessary information directly and explicitly. "
        "Zero irrelevant, redundant, or misleading details.\n"
        "- The problem should require one clear reasoning step. "
        "No chaining, multi-stage synthesis, or sequential operations.\n"
        "- Use standard, typical scenarios and conventional parameter values. "
        "No edge cases, boundary values, or unusual inputs.\n"
        "- Keep the problem statement concise and focused — strip narrative fluff. "
        "The core task should be immediately recognizable.\n"
        "- The correct approach should be obvious to anyone familiar with the skill. "
        "No tricks, no hidden assumptions, no double meanings."
    ),
    "medium": (
        "## Difficulty Target: MEDIUM\n"
        "Generate a task requiring competent, multi-step application of the skill:\n"
        "- Include some extraneous context that is plausible but ultimately irrelevant "
        "to the solution — the solver must distinguish signal from noise.\n"
        "- Require 2–3 distinct reasoning steps that depend on each other "
        "(output of step 1 feeds step 2, etc.).\n"
        "- Vary at least one parameter or condition away from the most standard formulation. "
        "Use realistic but non-trivial values that require careful handling.\n"
        "- For applied domains (clinical, engineering), frame the problem in a concrete scenario. "
        "For abstract domains (math, logic), a direct but non-trivial formulation is fine.\n"
        "- The solution path should require genuine thought but should be discoverable "
        "by a competent practitioner without hints or external references."
    ),
    "hard": (
        "## Difficulty Target: HARD\n"
        "Generate a task that challenges even skilled practitioners.\n"
        "Difficulty may come from either or both of:\n"
        "  (a) Information complexity — verbose context with irrelevant/distracting/misleading "
        "details that the solver must filter through.\n"
        "  (b) Conceptual depth — a compact but deeply challenging problem requiring "
        "insight, non-obvious connections, or creative composition of ideas.\n"
        "Apply whichever mode fits the skill's domain:\n"
        "- Require 4+ reasoning steps with complex dependencies between intermediate results, "
        "OR a few steps that each require deep, non-routine thinking.\n"
        "- Use non-standard framing: unusual parameter combinations, edge cases that break "
        "naive assumptions, boundary values, or implicit constraints that must be inferred.\n"
        "- Mix multiple sub-concepts or sub-techniques in ways not explicitly illustrated "
        "in the skill documentation. The solver must adapt and compose, not follow a recipe.\n"
        "- Add subtle constraints that change the solution logic: exceptions to the typical "
        "rule, double negatives, or cases where the standard approach does not directly apply.\n"
        "- Multiple plausible but incorrect paths should exist — identifying the right one "
        "requires deep understanding of the skill's principles."
    ),
}
_DATASET_FORMAT_CONTEXT = {
    "champ": (
        "Dataset: champ (math competition).\n"
        "Expected answer format: ANSWER: <value> on its own line. "
        "Plain text, no LaTeX. Values: number, expression (2^n, C(n,k)), Yes/No, comma-separated."
    ),
    "theoremqa": (
        "Dataset: theoremqa (math/science).\n"
        "Expected answer format: 'Therefore, the answer is ...' in the final sentence. "
        "Answer must be one of: plain decimal number, integer, True/False, "
        "option like (a)-(f), or list like [1, 2, 3]. No symbolic expressions."
    ),
    "medcalcbench": (
        "Dataset: medcalcbench (medical calculations).\n"
        "Expected answer format: ANSWER: <numeric value> on its own line. "
        "Step-by-step calculation must be shown. Value can be integer, decimal, or date (MM/DD/YYYY)."
    ),
    "logicbench": (
        "Dataset: logicbench (logical reasoning).\n"
        "Expected answer format: BQA → yes or no (lowercase). "
        "MCQA → choice_1 through choice_5."
    ),
    "bigcodebench": (
        "Dataset: bigcodebench (code generation).\n"
        "Expected answer format: Python function body, 4-space indent, "
        "appended after the code_prompt. Must be executable code."
    ),
    "toolqa": (
        "Dataset: toolqa (tool-augmented QA).\n"
        "The trajectory uses the ReAct protocol. Each step is structured as:\n"
        "  Thought N: <single paragraph of reasoning>\n"
        "  Action N: <ActionType[args]>\n"
        "  Observation N: <tool output>\n"
        "The final step ends with: Action N: Finish[<answer>], followed by\n"
        "  Observation N: Answer: <answer>\n"
        "The answer is plain factual text."
    ),
}
_DATASET_TOOL_CONTEXT = {
    "medcalcbench": (
        "Tool usage: This skill provides a calculation tool (TOOL_CALL). "
        "Using the tool is preferred, but a trajectory that computes manually "
        "is also acceptable IF it satisfies ALL of the following:\n"
        "- Follows the exact formula/logic described in the skill content.\n"
        "- Each intermediate calculation step is shown and arithmetically correct "
        "(verify the math — reject on any arithmetic error).\n"
        "- The final answer matches what the tool would produce.\n"
        "Reject if the trajectory uses a fundamentally different method, has "
        "calculation errors, or skips intermediate verification steps."
    ),
    "toolqa": (
        "Tool usage: This skill's core method requires tool calls via the ReAct action protocol. "
        "The available tools are:\n\n"
        + _TOOLQA_REACT_INSTRUCTION
        + "\n\nReject if the trajectory uses tools not listed above or attempts to answer without tools."
    ),
}
_EVALUATOR_SYSTEM = """You are a trajectory quality evaluator for an AI skill training system.
Judge whether the reasoning trace is acceptable as training data.

## Evaluation Criteria (ALL must pass for acceptable=true)

1. Method Appropriateness (symmetric): The trajectory's method must match the
   skill-task relationship in BOTH directions.
   - If the skill's method applies, the trajectory uses it. Reject if it solves
     without the skill when the skill does apply (lost benefit: a fitting skill
     must actually be used).
   - If the skill's method does NOT apply, the trajectory solves by an
     appropriate method. Reject if it forces the skill onto a problem it does
     not fit (abandonment harm).
   No mismatch declaration is required or desired (implicit abandonment).
{tool_context}

2. Skill Independence: The trajectory reasons as self-contained problem-solving
   and does not lean on the skill text. Reject if it contains dangling references
   that would be incoherent without the skill (e.g. "the skill states",
   "according to the provided method", "as given"). Applies uniformly whether the
   skill is used or implicitly abandoned (no declaration, hence no exception).

3. Step Completeness: Every step of reasoning and computation is shown.
   Reject on a skipped or combined step, an unstated intermediate value, or an
   unjustified leap - EVEN IF the final answer is correct. A non-trivial
   computation must be evaluated one operation per line. Example to REJECT
   (answer correct, but collapsed): "0.75*9.2103 + 0.25*9.1050 = 9.184".
   Example to ACCEPT: "0.75 x 9.2103 = 6.9077; 0.25 x 9.1050 = 2.2762;
   sum = 6.9077 + 2.2762 = 9.184". Applies to arithmetic AND the setup/reasoning
   that precedes it. Computation delegated to a provided tool (TOOL_CALL) is exempt.

4. Soundness: No calculation errors, circular reasoning, or
   "obviously"/"clearly" gaps.

5. Answer: A final answer is clearly stated and extractable; magnitude plausible.

6. Format: The answer follows the dataset's expected format.
{format_context}

Output JSON only:
{{"acceptable": true/false, "reason": "brief explanation", "extracted_answer": "value"}}
"""
_EVALUATOR_USER = (
    "## Dataset\n{dataset}\n\n"
    "## Question\n{question}\n\n"
    "## Skill Content\n{skill_content}\n\n"
    "## Trajectory to Evaluate\n{trajectory_output}"
)
