import pandas as pd
import glob

def summarize_results(filename):
    df = pd.read_csv(filename)
    total = len(df)
    syntax_acc = df["Syntax_OK"].sum() / total
    region_acc = df["Region_OK"].sum() / total
    full_acc = df["Full_OK"].sum() / total

    # Detect model and dataset from filename
    fname = filename.replace("results_", "").replace(".csv", "")
    parts = fname.split("_")
    if len(parts) == 2:
        model, dataset = parts
    else:
        model, dataset = fname, "unknown"

    return {
        "Model": model,
        "Dataset": dataset,
        "Syntax_Acc": round(syntax_acc, 2),
        "Region_Acc": round(region_acc, 2),
        "Full_Acc": round(full_acc, 2),
    }

# Load all result_*.csv files
results = []
for file in glob.glob("results_*.csv"):
    results.append(summarize_results(file))

df_summary = pd.DataFrame(results)

df_summary.to_csv("accuracy_summary.csv", index=False)
print("💾 Saved accuracy summary -> accuracy_summary.csv")

# Separate static and dynamic
df_static = df_summary[df_summary["Dataset"] == "static"]
df_dynamic = df_summary[df_summary["Dataset"] == "dynamic"]

print("\n=== Static Dataset Results ===")
if not df_static.empty:
    print(df_static.to_string(index=False))
else:
    print("No static results found.")

print("\n=== Dynamic Dataset Results ===")
if not df_dynamic.empty:
    print(df_dynamic.to_string(index=False))
else:
    print("No dynamic results found.")
