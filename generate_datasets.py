# generate_datasets.py
import argparse
import json
import os

def generate_utterances(regions, dataset_type):
    """Generate utterances for the given set of regions."""
    utterances = []
    region_names = list(regions.keys())

    # simple filters: avoid meaningless pairs
    def valid_pair(A, B):
        if A == B:
            return False
        if A == "exit" and B == "start":
            return False
        if B == "start":
            return False
        if A == "exit":
            return False
        return True
    # === Binary operators ===
    for A in region_names:
        for B in region_names:
            if not valid_pair(A, B):
                continue

            # Precedence
            variants = [
                f"Visit {A} before {B}",
                f"{B} can only occur after {A}",
                f"You must reach {A} first, then {B}"
            ]
            for u in variants:
                utterances.append({"utterance": u, "expected": f"(Precedence {A} {B})"})

            # Response
            variants = [
                f"{B} only after {A}",
                f"After visiting {A}, eventually reach {B}",
                f"If {A} happens, {B} must follow"
            ]
            for u in variants:
                utterances.append({"utterance": u, "expected": f"(Response {A} {B})"})

            # RespondedExistence
            variants = [
                f"If you visit {A}, then you must also reach {B}",
                f"Visiting {A} implies eventually visiting {B}",
                f"Whenever {A} happens, {B} must also occur"
            ]
            for u in variants:
                utterances.append({"utterance": u, "expected": f"(RespondedExistence {A} {B})"})

            # Until
            variants = [
                f"Stay in {A} until reaching {B}",
                f"{A} must hold true until {B} occurs",
                f"Remain in {A} and stop only when {B} happens"
            ]
            for u in variants:
                utterances.append({"utterance": u, "expected": f"(Until {A} {B})"})

            # Or
            variants = [
                f"Visit {A} or {B}",
                f"Either {A} or {B} must be reached",
                f"At least one of {A} or {B} must occur"
            ]
            for u in variants:
                utterances.append({"utterance": u, "expected": f"(Or {A} {B})"})

            # And
            variants = [
                f"Visit {A} and {B}",
                f"Both {A} and {B} must be reached",
                f"You must eventually visit both {A} and {B}"
            ]
            for u in variants:
                utterances.append({"utterance": u, "expected": f"(And {A} {B})"})

            # WeakUntil
            variants = [
                f"Stay in {A} unless {B} happens, otherwise remain in {A} forever",
                f"{A} must hold until {B}, or forever if {B} never occurs",
                f"Continue in {A} until {B} happens, or stay in {A} always if {B} never does"
            ]
            for u in variants:
                utterances.append({"utterance": u, "expected": f"(WeakUntil {A} {B})"})




    # === Unary operators ===
    for A in region_names:
        if A != "start":  # skip "start" for unary
            
            # Eventually
            variants = [
                f"Eventually visit {A}",
                f"Sometime in the future reach {A}",
                f"{A} must happen eventually"
            ]
            for u in variants:
                utterances.append({"utterance": u, "expected": f"(Eventually {A})"})

            # Always
            variants = [
                f"Always stay in {A}",
                f"{A} must always hold",
                f"In every state remain in {A}"
            ]
            for u in variants:
                utterances.append({"utterance": u, "expected": f"(Always {A})"})
            
            # Next
            variants = [
                f"Next step must be {A}",
                f"In the following state visit {A}",
                f"Immediately after, go to {A}"
            ]
            for u in variants:
                utterances.append({"utterance": u, "expected": f"(Next {A})"})

            # Not
            variants = [
                f"Never visit {A}",
                f"Avoid {A}",
                f"{A} must not occur at all"
            ]
            for u in variants:
                utterances.append({"utterance": u, "expected": f"(Not {A})"})


    print(f"✅ Generated {len(utterances)} utterances for {dataset_type}")
    return utterances




def save_dataset(utterances, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(utterances, f, indent=2)
    print(f"💾 Saved {len(utterances)} utterances -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["static", "dynamic", "all"], default="all",
                        help="Which dataset to generate (static, dynamic, or all).")
    args = parser.parse_args()

    if args.dataset in ["static", "all"]:
        with open("regions_static.json") as f:
            regions_static = json.load(f)
        utterances_static = generate_utterances(regions_static, "static")
        save_dataset(utterances_static, "data/static_dataset.json")

    if args.dataset in ["dynamic", "all"]:
        with open("regions_dynamic.json") as f:
            regions_dynamic = json.load(f)
        utterances_dynamic = generate_utterances(regions_dynamic, "dynamic")
        save_dataset(utterances_dynamic, "data/dynamic_dataset.json")


if __name__ == "__main__":
    main()
