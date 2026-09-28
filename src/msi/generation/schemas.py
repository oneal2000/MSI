"""Declarative per-dataset format schemas for training data generation.

Defines the exact format requirements for benchmark instances so the generator
produces output consistent with the benchmark's evaluation pipeline.

Each schema entry specifies:
- required_eval_keys: eval_data fields that MUST be present
- eval_field_spec: type + description for each field
- valid_values: allowed values for enum-like fields
- answer_format_rules: how answers must be formatted per answer/task type
- example_instance: a concrete synthetic example (NOT from benchmark)
- validate(eval_data): strict validator returning (ok, reason)

No benchmark instance data is referenced. All rules are derived from
evaluator code and dataset specifications.
"""

import re
from typing import Any


# ---------------------------------------------------------------------------
# MedCalc-Bench: calculator_id -> output_type mapping
# ---------------------------------------------------------------------------

# Rules extracted from sragents.evaluate.datasets.medcalcbench evaluator
_DATE_CALCULATOR_IDS = frozenset({13, 68})
_GESTATIONAL_AGE_ID = 69
_INTEGER_CALCULATOR_IDS = frozenset({
    4, 15, 16, 17, 18, 20, 21, 25, 27, 28, 29,
    32, 33, 36, 43, 45, 48, 51,
})
# Everything else is "decimal"

# Calculator catalog: id -> {output_type, names, tools}
# Derived from evaluator rules + corpus tool definitions (parameter semantics only)
CALCULATOR_CATALOG = {
    2: {"output_type": "decimal", "names": ["Creatinine Clearance (Cockcroft-Gault)"], "tools": ["compute_ibw", "compute_cg_crcl"]},
    3: {"output_type": "decimal", "names": ["CKD-EPI GFR"], "tools": ["compute_ckd_epi_gfr"]},
    4: {"output_type": "integer", "names": ["CHA2DS2-VASc Score"], "tools": ["compute_cha2ds2vasc"]},
    5: {"output_type": "decimal", "names": ["Mean Arterial Pressure"], "tools": ["compute_map"]},
    6: {"output_type": "decimal", "names": ["Body Mass Index"], "tools": ["compute_bmi"]},
    7: {"output_type": "decimal", "names": ["Calcium Correction for Hypoalbuminemia"], "tools": ["compute_corrected_calcium"]},
    8: {"output_type": "decimal", "names": ["Wells' Criteria for PE"], "tools": ["compute_wells_pe"]},
    9: {"output_type": "decimal", "names": ["MDRD GFR"], "tools": ["compute_mdrd_gfr"]},
    10: {"output_type": "decimal", "names": ["Ideal Body Weight"], "tools": ["compute_ibw"]},
    11: {"output_type": "decimal", "names": ["QTc Bazett"], "tools": ["compute_qtc_bazett"]},
    13: {"output_type": "date", "names": ["Estimated Due Date"], "tools": ["compute_edd"]},
    15: {"output_type": "integer", "names": ["Child-Pugh Score"], "tools": ["compute_child_pugh"]},
    16: {"output_type": "integer", "names": ["Wells' Criteria for DVT"], "tools": ["compute_wells_dvt"]},
    17: {"output_type": "integer", "names": ["Revised Cardiac Risk Index"], "tools": ["compute_rcri"]},
    18: {"output_type": "integer", "names": ["HEART Score"], "tools": ["compute_heart_score"]},
    19: {"output_type": "decimal", "names": ["FIB-4 Index"], "tools": ["compute_fib4"]},
    20: {"output_type": "integer", "names": ["Centor Score"], "tools": ["compute_centor"]},
    21: {"output_type": "integer", "names": ["Glasgow Coma Score"], "tools": ["compute_gcs"]},
    22: {"output_type": "decimal", "names": ["Maintenance Fluids"], "tools": ["compute_maintenance_fluids"]},
    23: {"output_type": "decimal", "names": ["MELD Na"], "tools": ["compute_meld_na"]},
    24: {"output_type": "decimal", "names": ["Steroid Conversion"], "tools": ["convert_steroid"]},
    25: {"output_type": "integer", "names": ["HAS-BLED Score"], "tools": ["compute_hasbled"]},
    26: {"output_type": "decimal", "names": ["Sodium Correction for Hyperglycemia"], "tools": ["compute_corrected_sodium"]},
    27: {"output_type": "integer", "names": ["Glasgow-Blatchford Score"], "tools": ["compute_gbs"]},
    28: {"output_type": "integer", "names": ["APACHE II Score"], "tools": ["compute_apache2"]},
    29: {"output_type": "integer", "names": ["PSI / Pneumonia Severity Index"], "tools": ["compute_psi"]},
    30: {"output_type": "decimal", "names": ["Serum Osmolality"], "tools": ["compute_serum_osmolality"]},
    31: {"output_type": "decimal", "names": ["HOMA-IR"], "tools": ["compute_homa_ir"]},
    32: {"output_type": "integer", "names": ["Charlson Comorbidity Index"], "tools": ["compute_cci"]},
    33: {"output_type": "integer", "names": ["FeverPAIN Score"], "tools": ["compute_feverpain"]},
    36: {"output_type": "integer", "names": ["Caprini Score"], "tools": ["compute_caprini"]},
    38: {"output_type": "decimal", "names": ["Free Water Deficit"], "tools": ["compute_free_water_deficit"]},
    39: {"output_type": "decimal", "names": ["Anion Gap"], "tools": ["compute_anion_gap"]},
    40: {"output_type": "decimal", "names": ["Fractional Excretion of Sodium"], "tools": ["compute_fena"]},
    43: {"output_type": "integer", "names": ["SOFA Score"], "tools": ["compute_sofa"]},
    44: {"output_type": "decimal", "names": ["LDL Calculated"], "tools": ["compute_ldl"]},
    45: {"output_type": "integer", "names": ["CURB-65 Score"], "tools": ["compute_curb65"]},
    46: {"output_type": "decimal", "names": ["Framingham Risk Score"], "tools": ["compute_framingham"]},
    48: {"output_type": "integer", "names": ["PERC Rule for PE"], "tools": ["compute_perc"]},
    49: {"output_type": "decimal", "names": ["Morphine Milligram Equivalents"], "tools": ["compute_mme"]},
    51: {"output_type": "integer", "names": ["SIRS Criteria"], "tools": ["compute_sirs"]},
    56: {"output_type": "decimal", "names": ["QTc Fridericia"], "tools": ["compute_qtc_fridericia"]},
    57: {"output_type": "decimal", "names": ["QTc Framingham"], "tools": ["compute_qtc_framingham"]},
    58: {"output_type": "decimal", "names": ["QTc Hodges"], "tools": ["compute_qtc_hodges"]},
    59: {"output_type": "decimal", "names": ["QTc Rautaharju"], "tools": ["compute_qtc_rautaharju"]},
    60: {"output_type": "decimal", "names": ["Body Surface Area"], "tools": ["compute_bsa"]},
    61: {"output_type": "decimal", "names": ["Target Weight"], "tools": ["compute_target_weight"]},
    62: {"output_type": "decimal", "names": ["Adjusted Body Weight"], "tools": ["compute_ibw", "compute_abw"]},
    63: {"output_type": "decimal", "names": ["Delta Gap"], "tools": ["compute_delta_gap"]},
    64: {"output_type": "decimal", "names": ["Delta Ratio"], "tools": ["compute_delta_ratio"]},
    65: {"output_type": "decimal", "names": ["Albumin Corrected Anion Gap"], "tools": ["compute_albumin_corrected_anion_gap"]},
    66: {"output_type": "decimal", "names": ["Albumin Corrected Delta Gap"], "tools": ["compute_albumin_corrected_delta_gap"]},
    67: {"output_type": "decimal", "names": ["Albumin Corrected Delta Ratio"], "tools": ["compute_albumin_corrected_delta_ratio"]},
    68: {"output_type": "date", "names": ["Estimated Conception Date"], "tools": ["compute_conception_date"]},
    69: {"output_type": "gestational_age", "names": ["Estimated Gestational Age"], "tools": ["compute_gestational_age"]},
}


def get_calculator_output_type(calculator_id: int) -> str:
    if calculator_id in _DATE_CALCULATOR_IDS:
        return "date"
    if calculator_id == _GESTATIONAL_AGE_ID:
        return "gestational_age"
    if calculator_id in _INTEGER_CALCULATOR_IDS:
        return "integer"
    return "decimal"


def get_calculators_for_skill(skill: dict) -> list[int]:
    """Return calculator_ids that use tools from this skill.

    Matches skill's tool names against the catalog.
    """
    skill_tools = {t["name"] for t in skill.get("tools", [])}
    matching = []
    for cal_id, info in CALCULATOR_CATALOG.items():
        if skill_tools & set(info["tools"]):
            matching.append(cal_id)
    return sorted(matching)


# ---------------------------------------------------------------------------
# Per-dataset schemas
# ---------------------------------------------------------------------------

DATASET_SCHEMAS = {
    "theoremqa": {
        "required_eval_keys": ["answer", "answer_type"],
        "eval_field_spec": {
            "answer": {
                "type": "str",
                "description": "The correct answer as a plain string",
                "format_rules": {
                    "float": "Plain decimal number (e.g., '0.6667'). NOT expressions like '2/3', 'sqrt(2)', or 'pi'.",
                    "integer": "Plain integer string (e.g., '42'). NOT expressions like '6*7' or 'C(5,2)'.",
                    "bool": "'True' or 'False' (capitalized).",
                    "option": "'(a)' through '(f)' with parentheses.",
                    "list of integer": "JSON-like list string (e.g., '[1, 2, 3]').",
                    "list of float": "JSON-like list string (e.g., '[1.0, 2.5]').",
                },
            },
            "answer_type": {
                "type": "str",
                "valid_values": ["float", "integer", "bool", "option", "list of integer", "list of float"],
            },
        },
        "valid_answer_types": ["float", "integer", "bool", "option", "list of integer", "list of float"],
        "question_style": "Direct math/science problem. May include LaTeX notation. 30-200 chars.",
        "example_instance": {
            "question": "What is the sum of the first 50 positive integers?",
            "eval_data": {"answer": "1275", "answer_type": "integer"},
        },
    },

    "logicbench": {
        "required_eval_keys": ["answer", "task_type"],
        "eval_field_spec": {
            "answer": {
                "type": "str",
                "format_rules": {
                    "BQA": "'yes' or 'no' (lowercase) — whether the conclusion logically follows.",
                    "MCQA": "'choice_1' through 'choice_5'. Choices MUST be embedded in the question text.",
                },
            },
            "task_type": {
                "type": "str",
                "valid_values": ["BQA", "MCQA"],
            },
        },
        "valid_task_types": ["BQA", "MCQA"],
        "question_style": "Long narrative with logical premises. 300-600+ chars. MCQA must include choice_N: text in question.",
        "example_instance": {
            "question": (
                "If all students study hard, then they pass the exam. We know that Alice is a student "
                "and she studies hard. Can we conclude that Alice passes the exam?"
            ),
            "eval_data": {"answer": "yes", "task_type": "BQA"},
        },
    },

    "medcalcbench": {
        "required_eval_keys": ["answer", "calculator_id", "output_type", "lower_limit", "upper_limit"],
        "eval_field_spec": {
            "answer": {
                "type": "str",
                "format_rules": {
                    "integer": "Integer string (e.g., '26'). lower_limit == upper_limit == answer.",
                    "decimal": "Decimal string (e.g., '25.24'). lower_limit ≈ answer*0.95, upper_limit ≈ answer*1.05.",
                    "date": "MM/DD/YYYY format (e.g., '05/02/2025'). lower_limit == upper_limit == answer.",
                    "gestational_age": "'(weeks, days)' tuple format (e.g., '(28, 4)'). lower_limit == upper_limit == answer.",
                },
            },
            "calculator_id": {
                "type": "int",
                "description": "Identifies which calculator this task uses. Must match a valid calculator from the catalog.",
                "valid_range": f"Must be one of: {sorted(CALCULATOR_CATALOG.keys())}",
            },
            "output_type": {
                "type": "str",
                "valid_values": ["integer", "decimal", "date", "gestational_age"],
                "determination_rule": "Determined by calculator_id. See CALCULATOR_CATALOG.",
            },
            "lower_limit": {
                "type": "str",
                "description": "Lower bound for evaluation. Correct if lower_limit <= extracted_answer <= upper_limit.",
                "computation": {
                    "integer": "Same as answer (exact match).",
                    "decimal": "answer * 0.95 (±5% tolerance).",
                    "date": "Same as answer (exact match).",
                    "gestational_age": "Same as answer (exact match).",
                },
            },
            "upper_limit": {
                "type": "str",
                "description": "Upper bound for evaluation.",
                "computation": {
                    "integer": "Same as answer (exact match).",
                    "decimal": "answer * 1.05 (±5% tolerance).",
                    "date": "Same as answer (exact match).",
                    "gestational_age": "Same as answer (exact match).",
                },
            },
        },
        "question_style": "Patient case note (500-4000 chars) followed by a specific calculation task.",
        "example_instance": {
            "question": "A 70-year-old female with hypertension and diabetes presents for cardiac risk assessment.\nTask: Calculate her CHA2DS2-VASc score.",
            "eval_data": {"answer": "4", "calculator_id": 4, "output_type": "integer", "lower_limit": "4", "upper_limit": "4"},
        },
    },

    "toolqa": {
        "required_eval_keys": ["answer"],
        "eval_field_spec": {
            "answer": {
                "type": "str",
                "description": "Factual answer as plain text. Compared via normalized exact match.",
            },
        },
        "question_style": "Factual question answerable by database/API tools. 40-150 chars.",
        "example_instance": {
            "question": "What is the total price for staying 3 nights at the Grand Hotel?",
            "eval_data": {"answer": "$450.00"},
        },
    },
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _check_type(value: Any, expected: str) -> bool:
    if expected == "str":
        return isinstance(value, str) and len(value) > 0
    if expected == "int":
        return isinstance(value, int)
    return isinstance(value, (str, int, float))


def validate_instance(dataset: str, task: dict) -> tuple[bool, str]:
    """Validate a generated task — question-only, no eval_data required.

    Returns (is_valid, reason) where reason is empty string if valid.
    """
    if not isinstance(task, dict):
        return False, "task must be a dict"

    if "question" not in task:
        return False, "missing 'question' key"
    if not isinstance(task["question"], str) or not task["question"].strip():
        return False, "'question' must be a non-empty string"

    if dataset not in DATASET_SCHEMAS:
        return False, f"unknown dataset: {dataset}"

    return True, ""


def _validate_theoremqa(eval_data: dict, _question: str = "") -> tuple[bool, str]:
    answer_type = eval_data.get("answer_type", "")
    valid_types = DATASET_SCHEMAS["theoremqa"]["valid_answer_types"]
    if answer_type not in valid_types:
        return False, f"invalid answer_type '{answer_type}', must be one of {valid_types}"

    answer = str(eval_data.get("answer", ""))
    if not answer or answer in ("X.XX", "PLACEHOLDER", "..."):
        return False, f"answer looks like a placeholder: '{answer}'"

    if answer_type == "float":
        try:
            float(answer)
        except ValueError:
            if not re.match(r"^\[", answer):
                return False, f"float answer_type but answer '{answer}' is not a number"
    elif answer_type == "integer":
        try:
            val = float(answer)
            if not val.is_integer():
                return False, f"integer answer_type but answer '{answer}' is not an integer"
        except ValueError:
            return False, f"integer answer_type but answer '{answer}' is not a number"
    elif answer_type == "bool":
        if answer not in ("True", "False"):
            return False, f"bool answer_type but answer '{answer}' is not 'True'/'False'"
    elif answer_type == "option":
        if not re.match(r"^\([a-f]\)$", answer):
            return False, f"option answer_type but answer '{answer}' is not like '(a)'-'(f)'"
    elif "list" in answer_type:
        if not (answer.startswith("[") and answer.endswith("]")):
            return False, f"list answer_type but answer '{answer}' doesn't look like a list"

    return True, ""


def _validate_logicbench(eval_data: dict, question: str = "") -> tuple[bool, str]:
    task_type = eval_data.get("task_type", "")
    if task_type not in DATASET_SCHEMAS["logicbench"]["valid_task_types"]:
        return False, f"invalid task_type '{task_type}'"

    answer = str(eval_data.get("answer", ""))
    if task_type == "BQA":
        if answer not in ("yes", "no"):
            return False, f"BQA task_type but answer '{answer}' is not 'yes'/'no'"
    elif task_type == "MCQA":
        if not re.match(r"^choice_[1-5]$", answer):
            return False, f"MCQA task_type but answer '{answer}' is not 'choice_1'-'choice_5'"
        # Verify choices are embedded in the question text
        if question:
            found_choices = re.findall(r"choice_\d", question)
            if len(set(found_choices)) < 2:
                return False, "MCQA task_type but question text does not contain embedded choice_N lines"
    return True, ""


def _validate_medcalcbench(eval_data: dict, _question: str = "") -> tuple[bool, str]:
    calculator_id = eval_data.get("calculator_id")
    if not isinstance(calculator_id, int):
        return False, f"calculator_id must be int, got {type(calculator_id).__name__}"
    if calculator_id not in CALCULATOR_CATALOG:
        return False, f"invalid calculator_id {calculator_id}"

    expected_type = get_calculator_output_type(calculator_id)
    output_type = eval_data.get("output_type", "")
    if output_type != expected_type:
        return False, f"calculator_id {calculator_id} requires output_type '{expected_type}', got '{output_type}'"

    answer = str(eval_data.get("answer", ""))
    if not answer or answer in ("X.XX", "PLACEHOLDER"):
        return False, f"answer looks like a placeholder: '{answer}'"

    lower = str(eval_data.get("lower_limit", ""))
    upper = str(eval_data.get("upper_limit", ""))
    if not lower or not upper:
        return False, "lower_limit and upper_limit must be non-empty strings"

    if expected_type in ("integer", "date", "gestational_age"):
        if lower != answer or upper != answer:
            return False, f"{expected_type} type requires lower_limit == upper_limit == answer"
    elif expected_type == "decimal":
        try:
            ans_f = float(answer)
            lower_f = float(lower)
            upper_f = float(upper)
            expected_lower = ans_f * 0.95
            expected_upper = ans_f * 1.05
            # Allow 1% relative tolerance on the limits themselves (rounding / precision)
            if abs(lower_f - expected_lower) > max(abs(expected_lower) * 0.01, 1e-6):
                return False, (
                    f"decimal lower_limit={lower} does not match answer*0.95={expected_lower:.6g}"
                )
            if abs(upper_f - expected_upper) > max(abs(expected_upper) * 0.01, 1e-6):
                return False, (
                    f"decimal upper_limit={upper} does not match answer*1.05={expected_upper:.6g}"
                )
        except (ValueError, TypeError):
            return False, f"decimal lower_limit/upper_limit/answer must be parseable as float"

    return True, ""


def _validate_toolqa(eval_data: dict, _question: str = "") -> tuple[bool, str]:
    answer = str(eval_data.get("answer", ""))
    if not answer:
        return False, "answer must be non-empty"
    return True, ""


_VALIDATORS = {
    "theoremqa": _validate_theoremqa,
    "logicbench": _validate_logicbench,
    "medcalcbench": _validate_medcalcbench,
    "toolqa": _validate_toolqa,
}


# ---------------------------------------------------------------------------
# Prompt helpers: build schema-aware instructions for the generator
# ---------------------------------------------------------------------------

_FEWSHOT_EXAMPLES: dict[str, str] = {
    "theoremqa": """\
## Reference Examples

Example 1:
A bag contains 6 red marbles and 4 blue marbles. Two marbles are drawn without replacement. What is the probability that both marbles are the same color?

Example 2:
Find the absolute maximum value of f(x) = x^3 - 3x^2 - 9x + 5 on the interval [-2, 4].""",

    "logicbench": """\
## Reference Examples

Example 1 (BQA):
If a city improves its public transit system, more residents will use it for daily commuting. However, if ticket prices are raised significantly, ridership may decrease even with better service. We know that either Metroville improved its transit system or ridership did not increase. Based on this information, can we conclude that Metroville raised ticket prices significantly?

Example 2 (MCQA):
A study finds that regular exercise reduces the risk of heart disease. However, excessive exercise without proper recovery can lead to injuries that discourage further physical activity. We know that at least one of the following holds: (1) John exercises regularly, or (2) John's risk of heart disease is not reduced. Given this, which of the following must be true?
choice_1: John exercises excessively
choice_2: John has a reduced risk of heart disease
choice_3: If John exercises regularly, his risk of heart disease is reduced
choice_4: John does not exercise at all
choice_5: John will definitely develop heart disease""",

    "medcalcbench": """\
## Reference Examples

Example 1:
A 72-year-old female with a history of hypertension and type 2 diabetes presents to the clinic for a routine cardiovascular risk assessment. She is a non-smoker. Her blood pressure is 138/85 mmHg, total cholesterol is 210 mg/dL, HDL cholesterol is 45 mg/dL, and she is currently on antihypertensive medication. Calculate her ASCVD 10-year risk score. All necessary clinical parameters are provided above.

Example 2:
A 65-year-old man with a history of atrial fibrillation presents to the emergency department with palpitations. He has a history of congestive heart failure and hypertension. His blood pressure is 145/90 mmHg. Calculate his CHA2DS2-VASc score. All necessary clinical parameters are provided above.""",

    "toolqa": """\
## Reference Examples

Example 1:
What was the average price of coffee products across all stores in the Downtown district during the month of March 2022?

Example 2:
Which airline had the highest number of delayed flights departing from Terminal B between January and March 2023?""",
}


def build_format_instruction(dataset: str, skill: dict | None = None) -> str:
    """Build question format instruction for the generator prompt.

    Only describes constraints on the question itself — what format the
    question and its answer must follow. No eval_data output requirements.
    """
    lines = [f"## Question Format for {dataset}\n"]

    if dataset == "theoremqa":
        lines.append(
            "The answer to your question must be exactly one of: "
            "a plain decimal number, an integer, True or False, "
            "an option like (a)-(f), or a list like [1, 2, 3]. "
        )

    elif dataset == "logicbench":
        lines.append(
            "Your question should present a logical reasoning problem, either as a binary question (BQA) or multiple choice (MCQA). "
            "For yes/no reasoning questions: end with a determinate yes/no question.\n"
            "For multiple choice: end with a question offer at most 5 distinct options embedded at the end (e.g., 'choice_1: ...', 'choice_2: ...')"
        )

    elif dataset == "medcalcbench":
        lines.append(
            "Describe a patient scenario specifying which score to calculate. "
            "The answer will be a number or date."
        )
        if skill:
            matching_cals = get_calculators_for_skill(skill)
            if matching_cals:
                cal_names = []
                for cal_id in matching_cals:
                    info = CALCULATOR_CATALOG[cal_id]
                    cal_names.append(f"{info['names'][0]}")
                lines.append(
                    f" This skill covers: {', '.join(cal_names)}."
                )

    elif dataset == "toolqa":
        lines.append(
            "The question must require the solver to retrieve information by calling tools "
            "(LoadDB, FilterDB, GetValue, LoadGraph, etc.). "
            "The solver starts with NO knowledge of database contents — "
            "every piece of data needed to answer must be obtained through tool calls. "
            "Do NOT embed database facts as given information in the question; "
            "only provide identifiers or criteria the solver can use as query parameters. "
            "The answer will be a plain factual result discovered via tools."
        )
        lines.append(
            "\nWrite DIRECT questions starting with What/Who/When/Where/How/Which/Can. "
            "Do NOT use narrative framing (no \"A researcher is...\", \"A traveler wants...\", "
            "\"You are planning...\", etc.). State the question concisely, under 200 characters."
        )

    lines.append(
        "\nThe question must be self-contained with all necessary parameters."
    )

    examples = _FEWSHOT_EXAMPLES.get(dataset, "")
    if examples:
        lines.append("\n" + examples)

    return "\n".join(lines)
