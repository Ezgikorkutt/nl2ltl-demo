Perfect 👍 Let me rewrite your README.md into a polished, professional version.
Here’s a clean structure you can copy into your repo:

# NL2LTL–Zonopy Evaluation Pipeline

This repository provides a **reproducible pipeline** for evaluating Large Language Models (LLMs) — **GPT, Claude, DeepSeek, Mistral, and Gemma** — on the task of translating natural language utterances into **Linear Temporal Logic (LTL)** and **Declare templates**.  
The pipeline integrates with **Zonopy** and supports verification in both static and dynamic environments.

---

## ✨ Features
- Integration of **5 LLMs** for NL2LTL translation
- Dataset generation for **static** and **dynamic** environments
- Evaluation across **three metrics**:
  - **Syntax Accuracy** (structural correctness)
  - **Region Accuracy** (valid environment references)
  - **Full Accuracy** (exact match with expected formula)
- Automated result collection and comparison

---

## 📂 Project Structure
```plaintext
├── integrate_nl2ltl_zonopy.py   # Core pipeline: LLM query, parsing, monitoring
├── generate_datasets.py         # Dataset generator (static & dynamic envs)
├── compare_results.py            # Aggregates results into summary tables
├── requirements.txt              # Python dependencies
├── .gitignore                    # Ignored files and directories
├── data/                         # Sample datasets & region definitions

⚙️ Installation

Clone the repository

git clone https://github.com/<your-username>/nl2ltl-demo.git
cd nl2ltl-demo


Create and activate a virtual environment

python3 -m venv .venv310
source .venv310/bin/activate    # Mac/Linux
.venv310\Scripts\activate       # Windows PowerShell


Install dependencies

pip install -r requirements.txt


Set API Keys (export them in your terminal)

export OPENAI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export GOOGLE_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export MISTRAL_API_KEY="..."

🚀 Usage

Generate datasets

python generate_datasets.py


Run evaluation

python integrate_nl2ltl_zonopy.py --model gpt --dataset static


Compare results

python compare_results.py

📊 Example Results
Model	Dataset	Syntax Acc.	Region Acc.	Full Acc.
GPT	Static	1.00	0.95	0.87
Claude	Static	0.95	0.90	0.79
Gemma	Dynamic	1.00	0.95	0.91
DeepSeek	Dynamic	1.00	0.95	0.86
📜 License

This project is licensed under the MIT License.

🙌 Acknowledgements

Zonopy
 for set-based reachability

NL2LTL
 for natural language to LTL translation

All evaluated LLM providers (OpenAI, Anthropic, Google, DeepSeek, Mistral)
