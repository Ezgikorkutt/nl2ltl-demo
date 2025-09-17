from pathlib import Path
from nl2ltl import translate
from nl2ltl.engines.gpt.core import GPTEngine, Models
from nl2ltl.filters.simple_filters import BasicFilter
from nl2ltl.engines.utils import pretty

# Create the GPT engine:
# - model: choose one from the library's supported models
# - prompt: point to the local prompt.json we downloaded
engine = GPTEngine(model="gpt-3.5-turbo", prompt=Path("data/prompt.json"))


# Simple built-in filter to prune/rank the results
filt = BasicFilter()

# A sample natural language instruction
utterance = "Eventually send me a Slack after receiving a Gmail"

# Run the translation (NL -> LTL), returns {Formula: score}
formulas = translate(utterance, engine, filt)

# Print both raw and pretty versions
print("Raw results (score :: formula):")
for f, s in formulas.items():
    print(s, "::", f)

print("\nPretty:")
pretty(formulas)
