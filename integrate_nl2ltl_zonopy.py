import sys, os
from pathlib import Path

import json
import csv
import re
import anthropic
import requests
from collections import OrderedDict
import google.generativeai as genai
import time 

import numpy as np
from scipy.optimize import linprog

# --- make the fork visible ---
sys.path.append(os.path.expanduser("~/Desktop/zonopy-loizos/src"))   # zonopy-loizos fork
sys.path.append(os.path.expanduser("~/Desktop/nl2ltl"))              # nl2ltl

# Require zonopy directly (no fallback in this “direct” version)
from utils.sets.zonotopes import Zonotope


from nl2ltl.core import translate   # modern repo layout
from nl2ltl.engines.gpt.core import GPTEngine
from nl2ltl.filters.simple_filters import BasicFilter

SYSTEM_PROMPT_TEMPLATE = """
You are a translator from natural language instructions into Declare/LTL templates.

Use ONLY these region names: {allowed_names}.
Do NOT invent names. If the instruction mentions something not in the list, map it to the closest valid name.

Return format:
You MUST return valid JSON with a single key "formula".
Example:
{{"formula": "(Precedence slot1 exit)"}}

Allowed templates (with meaning):
- Response(A,B): If A occurs, then B must eventually occur after A. NEVER simplify this into (Eventually B).
- Precedence(A,B): B can only occur if A has already occurred before.
- RespondedExistence(A,B): If A occurs, then B must also occur (at some point).
- Eventually(A): A must occur at least once. (◇A)
- Always(A): A must hold in every state. (□A)
- Until(A,B): A must hold until B occurs, and B must occur eventually. (A U B)
- WeakUntil(A,B): A must hold until B occurs, OR forever if B never occurs. (A W B)
- Next(A): A must hold in the next state. (X A)
- Not(A): A must never occur. (¬A)
- Or(A,B): Either A or B (or both) must occur. (A ∨ B)
- And(A,B): Both A and B must occur. (A ∧ B)

Rules:
- Only use the operators listed above.
- Do not output synonyms like "sometime", "always avoid", "afterwards", etc. Map them to the closest allowed operator.
- Do not output code blocks or explanations.
- Only output valid JSON in the exact format above.

Examples:
Utterance: "Visit slot1 before exit"
Output: {{"formula": "(Precedence slot1 exit)"}}

Utterance: "Exit only after slot1"
Output: {{"formula": "(Response slot1 exit)"}}

Utterance: "If you visit slot1, then you must also reach exit"
Output: {{"formula": "(RespondedExistence slot1 exit)"}}

Utterance: "Eventually visit slot1"
Output: {{"formula": "(Eventually slot1)"}}

Utterance: "Always stay in road1"
Output: {{"formula": "(Always road1)"}}

Utterance: "Never visit car2"
Output: {{"formula": "(Not car2)"}}

Utterance: "Stay in road1 until reaching exit"
Output: {{"formula": "(Until road1 exit)"}}

Utterance: "Stay in road1 unless exit happens, otherwise remain in road1 forever"
Output: {{"formula": "(WeakUntil road1 exit)"}}

Utterance: "Next step must be slot1"
Output: {{"formula": "(Next slot1)"}}

Utterance: "Visit slot1 or slot2"
Output: {{"formula": "(Or slot1 slot2)"}}

Utterance: "Visit slot1 and exit"
Output: {{"formula": "(And slot1 exit)"}}

Now translate the following utterance into the same JSON format:

"""

# =========================
# CONFIG
# =========================
USE_GPT = True          # keep True to use a real model
USE_NL2LTL_CALL = False # set False to bypass nl2ltl.translate and call OpenAI directly

try:
    from openai import OpenAI  # openai>=1.0
except Exception:
    OpenAI = None

UTTERANCE = "Visit slot1 before exit"   # Precedence(slot1, exit)
VALID_TEMPLATES = {
    "Response", "Precedence", "RespondedExistence",
    "Eventually", "Always", "Until", "WeakUntil",
    "Next", "Not", "Or", "And"
}
# Two-argument operators
SEXPR_RE = re.compile(
    r"\(\s*(Response|Precedence|RespondedExistence|Until|WeakUntil|Or|And)\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s*\)"
)

# One-argument operators
SEXPR_RE_UNARY = re.compile(
    r"\(\s*(Eventually|Always|Next|Not)\s+([A-Za-z0-9_]+)\s*\)"
)


# =========================
# Region / labeling helpers
# =========================
def _build_box_zonotope(center, halfwidth):
    c = np.asarray(center, dtype=float).reshape(-1)
    h = np.asarray(halfwidth, dtype=float).reshape(-1)
    G = np.diag(h)
    Z = Zonotope(c, G)  # direct constructor for your fork
    return {"c": c, "G": G, "Z": Z}

def _zono_contains(region, x):
    c = region["c"]; G = region["G"]
    x = np.asarray(x, dtype=float).reshape(-1)
    d = x - c
    m = G.shape[1]
    res = linprog(c=np.zeros(m), A_eq=G, b_eq=d, bounds=[(-1.0, 1.0)] * m, method="highs")
    return bool(res.success)

class RegionLabeler:
    def __init__(self, regions: dict[str, dict]):
        self.regions = regions
    def labels_for_state(self, x_np: np.ndarray):
        labs = set()
        for name, R in self.regions.items():
            if _zono_contains(R, x_np):
                labs.add(name)
        return labs
    

def load_regions_from_json(json_path):
    import json
    with open(json_path, "r") as f:
        data = json.load(f)
    regions = {}
    for name, spec in data.items():
        c = spec["center"]
        h = spec["halfwidth"]

        # --- TEMP tuning for non-vacuous demo ---
        if name == "slot1":
            h = [1.3 * v for v in h]   # inflate slot1 a little
        if name == "exit":
            h = [0.5 * v for v in h]   # shrink exit so it’s not hit too early

        regions[name] = _build_box_zonotope(c, h)
    return regions


# =========================
# Declare monitors
# =========================
def monitor_RespondedExistence(A, B, trace):
    idx_A = [i for i, t in enumerate(trace) if A in t]
    if not idx_A:
        return True, {"reason": "A never occurs"}
    idx_B = [i for i, t in enumerate(trace) if B in t]
    return (len(idx_B) > 0), {"firstA": idx_A[0], "firstB": (idx_B[0] if idx_B else None)}

def monitor_Response(A, B, trace):
    for i, t in enumerate(trace):
        if A in t and not any(B in s for s in trace[i+1:]):
            return False, {"violating_A_index": i}
    return True, {}

def monitor_Precedence(A, B, trace):
    seen_A = False
    for i, t in enumerate(trace):
        if B in t and not seen_A:
            return False, {"violating_B_index": i}
        if A in t:
            seen_A = True
    return True, {}

def monitor_Eventually(A, trace):
    # True if A occurs at least once in the trace
    for i, t in enumerate(trace):
        if A in t:
            return True, {"first_occurrence": i}
    return False, {"reason": f"{A} never occurs"}

def monitor_Always(A, trace):
    # True if A is in every state
    for i, t in enumerate(trace):
        if A not in t:
            return False, {"violating_index": i}
    return True, {}

def monitor_Until(A, B, trace):
    # φ U ψ: B must occur eventually, and A must hold until then
    seen_B = False
    for i, t in enumerate(trace):
        if B in t:
            seen_B = True
            return True, {"until_index": i}
        if A not in t:
            return False, {"violating_index": i}
    return False, {"reason": f"{B} never occurs"}

def monitor_Next(A, trace):
    # True if whenever we check "next state", A holds there
    for i in range(len(trace) - 1):  # stop at second-to-last
        if A in trace[i] and A not in trace[i + 1]:
            return False, {"violating_index": i}
    return True, {}

def monitor_Not(A, trace):
    # True if A never occurs in the trace
    for i, t in enumerate(trace):
        if A in t:
            return False, {"violating_index": i}
    return True, {}

def monitor_Or(A, B, trace):
    # True if either A or B occurs at least once in the trace
    seen_A = any(A in t for t in trace)
    seen_B = any(B in t for t in trace)
    if seen_A or seen_B:
        return True, {}
    return False, {"reason": f"Neither {A} nor {B} ever occurs"}

def monitor_And(A, B, trace):
    # True if both A and B occur at least once in the trace
    seen_A = any(A in t for t in trace)
    seen_B = any(B in t for t in trace)
    if seen_A and seen_B:
        return True, {}
    return False, {"reason": f"Missing {A if not seen_A else B}"}

def monitor_WeakUntil(A, B, trace):
    # A W B: A must hold until B occurs, or forever if B never occurs
    seen_B = False
    for i, t in enumerate(trace):
        if B in t:
            seen_B = True
            return True, {"until_index": i}
        if A not in t:
            return False, {"violating_index": i}
    # If we reach here, B never occurred → require A in all states
    return True, {"reason": f"{B} never occurs, but {A} always holds"}


MONITORS = {
    "RespondedExistence": monitor_RespondedExistence,
    "Response":           monitor_Response,
    "Precedence":         monitor_Precedence,
    "Eventually":         monitor_Eventually,
    "Always":             monitor_Always,
    "Until":              monitor_Until,
    "Next":               monitor_Next,
    "Not":                monitor_Not,
    "Or":                 monitor_Or,
    "And":                monitor_And,
    "WeakUntil":          monitor_WeakUntil,

}

# =========================
# Helpers for formula parsing
# =========================
def parse_declare(formula_str: str):
    # First try binary operators
    m = SEXPR_RE.match(str(formula_str))
    if m:
        template, A, B = m.groups()
        return template, A, B

    # Then try unary operators
    m = SEXPR_RE_UNARY.match(str(formula_str))
    if m:
        template, A = m.groups()
        return template, A, None  # B = None for unary

    raise ValueError(f"Unrecognized formula format: {formula_str}")


def coerce_formulas(raw):
    """
    Accept nl2ltl outputs OR raw strings; extract first valid '(Template ...)'.
    Return OrderedDict({ '(Template ...)' : score }).
    """

    # Case 1: dict-like outputs from nl2ltl
    if isinstance(raw, dict) and raw:
        for k, score in raw.items():
            s = str(k)

            # Binary match
            m = SEXPR_RE.search(s)
            if m:
                return OrderedDict({m.group(0): float(score)})

            # Unary match
            m = SEXPR_RE_UNARY.search(s)
            if m:
                return OrderedDict({m.group(0): float(score)})

    # Case 2: single string (raw formula)
    s = str(raw)

    # Binary
    m = SEXPR_RE.search(s)
    if m:
        return OrderedDict({m.group(0): 1.0})

    # Unary
    m = SEXPR_RE_UNARY.search(s)
    if m:
        return OrderedDict({m.group(0): 1.0})

    raise ValueError(f"Could not find a valid formula in: {raw!r}")


def _extract_formula_json(text, region_names):
    import json, re

    # 1) Try JSON object
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            formula = obj.get("formula", "").strip()
            if formula:
                tokens = formula.replace("(", " ").replace(")", " ").replace(",", " ").split()
                op = tokens[0].capitalize()



                # Binary operators
                if len(tokens) >= 3 and op in {"Response", "Precedence", "RespondedExistence", "Until", "WeakUntil", "Or", "And"}:
                    A = closest_region(tokens[1], region_names)
                    B = closest_region(tokens[2], region_names)
                    formula = f"({op} {A} {B})"
                    return coerce_formulas(formula)

                # Unary operators
                if len(tokens) >= 2 and op in {"Eventually", "Always", "Next", "Not"}:
                    A = closest_region(tokens[1], region_names)
                    formula = f"({op} {A})"
                    return coerce_formulas(formula)

        except Exception:
            pass

    # 2) Fallback direct regex search
    m = SEXPR_RE.search(text)  # binary
    if m:
        return coerce_formulas(m.group(0))

    m = SEXPR_RE_UNARY.search(text)  # unary
    if m:
        return coerce_formulas(m.group(0))

    # 3) Fail
    raise ValueError(f"Could not parse formula from: {text}")


def closest_region(name: str, region_names: set[str]) -> str:
    # exact match
    if name in region_names:
        return name
    # lowercase fallback
    if name.lower() in region_names:
        return name.lower()
    # simple heuristic: substring match
    for r in region_names:
        if name.lower() in r.lower() or r.lower() in name.lower():
            return r
    # fallback: return unchanged (will fail region_ok check later)
    return name

def get_formulas(utterance: str, system_prompt: str, model_name="gpt", region_names=None):
    """
    Query an LLM (GPT, Claude, DeepSeek, Qwen via HF) to translate utterance -> Declare formula.
    """

    # === OpenAI GPT ===
    if model_name == "gpt":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": utterance},
            ],
        )
        text = resp.choices[0].message.content.strip()
        return _extract_formula_json(text, region_names)

 
    # === DeepSeek (OpenAI-compatible API) ===
    elif model_name == "deepseek":
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"   # ✅ DeepSeek uses OpenAI-compatible API
        )
        resp = client.chat.completions.create(
            model="deepseek-chat",   # ✅ DeepSeek’s main chat model
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": utterance},
            ],
        )
        text = resp.choices[0].message.content.strip()
        return _extract_formula_json(text, region_names)
    
    # === Gemma 3 (Google AI Studio) ===
    elif model_name == "gemma":
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel("models/gemma-3-12b-it")   
        response = model.generate_content(
            f"{system_prompt}\nUser: {utterance}"
        )
        text = response.text.strip()
        return _extract_formula_json(text, region_names)

    
    # === claude ===

    elif model_name == "claude":
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{system_prompt}\nUtterance: {utterance}"}
                    ],
                }
            ],
        )
        text = resp.content[0].text.strip()
        return _extract_formula_json(text, region_names)

    # === Mistral (official API, OpenAI-compatible) ===
    elif model_name == "mistral":
        import time, random, json
        client = OpenAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            base_url="https://api.mistral.ai/v1"
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model="mistral-medium",   # options: mistral-small, mistral-medium, mixtral-8x7b-instruct
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": utterance},
                    ],
                )
                text = resp.choices[0].message.content.strip()
                return _extract_formula_json(text, region_names)

            except Exception as e:
                err_msg = str(e)
                print(f"⚠️ Mistral API error on attempt {attempt+1}/{max_retries}: {err_msg}")

                # If it's a rate limit / capacity (429), wait longer
                if "429" in err_msg or "capacity" in err_msg.lower():
                    wait_time = 20 + random.uniform(0, 10)  # 20–30 sec for 429
                else:
                    wait_time = 5 + random.uniform(0, 5)    # 5–10 sec for other errors

                if attempt == max_retries - 1:
                    raise
                print(f"⏳ Waiting {wait_time:.1f} seconds before retry...")
                time.sleep(wait_time)


    

# =========================
# Main
# =========================
def main():
    # === Detect dataset from CLI (default = dynamic) ===
    dataset = "dynamic"
    for i, arg in enumerate(sys.argv):
        if arg == "--dataset" and i + 1 < len(sys.argv):
            dataset = sys.argv[i + 1]

    if dataset == "static":
        regions = load_regions_from_json("regions_static.json")
        csv_path = "robot_states_static.csv"
    else:
        regions = load_regions_from_json("regions_dynamic.json")
        csv_path = "robot_states_dynamic.csv"

    region_names = set(regions.keys())
    allowed_names = ", ".join(sorted(region_names))

    SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.format(allowed_names=allowed_names)

    # 🔹 Detect model from CLI (default = gpt)
    model_name = "gpt"
    for i, arg in enumerate(sys.argv):
        if arg == "--model" and i + 1 < len(sys.argv):
            model_name = sys.argv[i + 1]

    # 🔹 Call chosen LLM
    formulas = get_formulas(UTTERANCE, SYSTEM_PROMPT, model_name=model_name, region_names=region_names)

    labeler = RegionLabeler(regions)

    # 3) Trajectory from CSV
    if dataset == "static":
        regions = load_regions_from_json("regions_static.json")
        csv_path = "robot_states_static.csv"
    else:
        regions = load_regions_from_json("regions_dynamic.json")
        csv_path = "robot_states_dynamic.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}. Put it next to this script.")
    traj = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            traj.append(np.array([float(row["x"]), float(row["y"])]))
    if not traj:
        raise RuntimeError("Loaded 0 rows from robot_states.csv.")
    event_trace = [labeler.labels_for_state(x) for x in traj]

    slot1_idx = [i for i, t in enumerate(event_trace) if "slot1" in t]
    exit_idx  = [i for i, t in enumerate(event_trace) if "exit"  in t]
    start_idx = [i for i, t in enumerate(event_trace) if "start" in t]
    print("slot1 indices:", slot1_idx[:20])
    print("exit indices:",  exit_idx[:20])
    print("start indices:", start_idx[:20])
    if exit_idx and (not slot1_idx or slot1_idx[0] > exit_idx[0]):
        print("NOTE: Reached EXIT (index", exit_idx[0], ") before visiting SLOT1 — hence Precedence(slot1,exit) is False.")
    print("Event trace (first 10):", event_trace[:10], "..." if len(event_trace) > 10 else "")

    # 4) Pick top formula and monitor
    best = next(iter(formulas.keys()))
    template, A, B = parse_declare(str(best))
    if B:
        print(f"\nTop formula (Model: {model_name}): {template}({A},{B})")
    else:
        print(f"\nTop formula (Model: {model_name}): {template}({A})")

    monitor_fn = MONITORS.get(template)
    if monitor_fn is None:
        raise NotImplementedError(f"No monitor implemented for template '{template}'")

    # Call monitor depending on arity
    if B is None:  # unary
        ok, info = monitor_fn(A, event_trace)
    else:          # binary
        ok, info = monitor_fn(A, B, event_trace)

    # Print evaluation result
    if B:
        print(f"{template}({A},{B}):", ok, info)
    else:
        print(f"{template}({A}):", ok, info)



# =========================
# Evaluation Harness
# ========================
def evaluate_llm():
    import time

    def normalize_formula(s: str) -> str:
        if not s:
            return ""
        s = s.strip().replace(",", " ").replace("_", " ")
        s = re.sub(r"\s+", " ", s)
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1].strip()
        tokens = s.replace("(", " ").replace(")", " ").split()
        if len(tokens) == 2:  # unary
            template = tokens[0].capitalize()

            A = tokens[1].lower()
            return f"({template} {A})"
        elif len(tokens) >= 3:  # binary
            template = tokens[0].capitalize()

            A = tokens[1].lower()
            B = tokens[2].lower()
            return f"({template} {A} {B})"
        return s

    # === Detect model from CLI ===
    model_name = "gpt"
    for i, arg in enumerate(sys.argv):
        if arg == "--model" and i + 1 < len(sys.argv):
            model_name = sys.argv[i + 1]

    # === Detect dataset from CLI ===
    dataset_type = "static"
    for i, arg in enumerate(sys.argv):
        if arg == "--dataset" and i + 1 < len(sys.argv):
            dataset_type = sys.argv[i + 1].lower()

    if dataset_type == "static":
        regions_path = "regions_static.json"
        dataset_file = "data/static_dataset.json"
    elif dataset_type == "dynamic":
        regions_path = "regions_dynamic.json"
        dataset_file = "data/dynamic_dataset.json"
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # === Load regions ===
    regions = load_regions_from_json(regions_path)
    region_names = set(regions.keys())

    # === Load utterances for this dataset ===
    with open(dataset_file, "r") as f:
        dataset = json.load(f)
    TEST_SET = [(item["utterance"], item["expected"]) for item in dataset]

    # === Accuracy counters ===
    total = len(TEST_SET)
    syntax_ok = region_ok = full_ok = 0

    allowed_names = ", ".join(sorted(region_names))
    SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.format(allowed_names=allowed_names)

    # === Output CSV per dataset & model ===
    csv_filename = f"results_{model_name}_{dataset_type}.csv"

    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Utterance", "Expected", "Predicted", "Syntax_OK", "Region_OK", "Full_OK"])

        for utter, expected in TEST_SET:
            print("\n==============================")
            print("Utterance:", utter)
            print("Expected:", expected)

            pred = ""
            syntax_flag = region_flag = full_flag = 0
            error_flag = False

            # === Model call ===
            try:
                formulas = get_formulas(
                    utter, SYSTEM_PROMPT, model_name=model_name, region_names=region_names
                )
                pred = next(iter(formulas.keys()))
                print("Predicted:", pred)
            except Exception as e:
                print("Error:", e)   # 👈 shows reason in console
                pred = "ERROR"
                error_flag = True

            if model_name == "gemma":
                time.sleep(2)

            # === Check 1: Syntax ===
            if not error_flag:
                try:
                    template, A, B = parse_declare(pred)
                    syntax_ok += 1
                    syntax_flag = 1
                except Exception as e:
                    print("❌ Invalid syntax:", e)
                    pred = "ERROR"
                    error_flag = True

            # === Check 2: Region names valid ===
            if not error_flag:
                if B is None:  # unary operator
                    if A in region_names:
                        region_ok += 1
                        region_flag = 1
                    else:
                        print(f"❌ Invalid region: {A}")
                        pred = "ERROR"
                        error_flag = True
                else:  # binary operator
                    if A in region_names and B in region_names:
                        region_ok += 1
                        region_flag = 1
                    else:
                        print(f"❌ Invalid region(s): {A}, {B}")
                        pred = "ERROR"
                        error_flag = True
            
            # === Print evaluation result (only if valid so far) ===
            if not error_flag:
                if B:
                    print(f"Predicted formula (parsed): {template}({A},{B})")
                else:
                    print(f"Predicted formula (parsed): {template}({A})")


            # === Check 3: Full match with expected ===
            if not error_flag:
                norm_pred = normalize_formula(pred)
                norm_exp = normalize_formula(expected)

                if norm_pred == norm_exp:
                    full_ok += 1
                    full_flag = 1
                    print("✅ Correct")
                else:
                    print(f"⚠️ Wrong formula\n   Expected: {norm_exp}\n   Got:      {norm_pred}")

            # === Always write row ===
            writer.writerow([utter, expected, pred, syntax_flag, region_flag, full_flag])

    print("\n=== Accuracy Summary (Model:", model_name, "Dataset:", dataset_type, ") ===")
    print(f"Syntax accuracy: {syntax_ok}/{total} = {syntax_ok/total:.2f}")
    print(f"Region accuracy: {region_ok}/{total} = {region_ok/total:.2f}")
    print(f"Full accuracy:   {full_ok}/{total} = {full_ok/total:.2f}")
    print(f"Results saved to {csv_filename}")



if __name__ == "__main__":
    if "--eval" in sys.argv:
        evaluate_llm()
    else:
        main()