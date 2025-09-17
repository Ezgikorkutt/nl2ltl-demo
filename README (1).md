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
```

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/nl2ltl-demo.git
   cd nl2ltl-demo
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv310
   source .venv310/bin/activate    # Mac/Linux
   .venv310\Scripts\activate       # Windows PowerShell
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set API Keys** (export them in your terminal)
   ```bash
   export OPENAI_API_KEY="..."
   export DEEPSEEK_API_KEY="..."
   export GOOGLE_API_KEY="..."
   export ANTHROPIC_API_KEY="..."
   export MISTRAL_API_KEY="..."
   ```

---

## 🚀 Usage

1. **Generate datasets**
   ```bash
   python generate_datasets.py
   ```

2. **Run evaluation**
   ```bash
   python integrate_nl2ltl_zonopy.py --model gpt --dataset static
   ```

3. **Compare results**
   ```bash
   python compare_results.py
   ```

---

## 📊 Example Results


### Static Dataset
| Model    | Syntax Acc. | Region Acc. | Full Acc. |
|----------|-------------|-------------|-----------|
| Claude   | 0.95        | 0.90        | 0.79      |
| Gemma    | 1.00        | 0.94        | 0.91      |
| DeepSeek | 1.00        | 0.96        | 0.86      |
| GPT      | 1.00        | 0.95        | 0.87      |
| Mistral  | 1.00        | 0.92        | 0.85      |

### Dynamic Dataset
| Model    | Syntax Acc. | Region Acc. | Full Acc. |
|----------|-------------|-------------|-----------|
| Gemma    | 1.00        | 0.95        | 0.91      |
| Mistral  | 1.00        | 0.94        | 0.86      |
| Claude   | 0.99        | 0.94        | 0.86      |
| DeepSeek | 1.00        | 0.95        | 0.86      |
| GPT      | 1.00        | 0.93        | 0.87      |


---

## 📜 License
This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgements
- [Zonopy](https://github.com/zonopy/zonopy) for set-based reachability
- [NL2LTL](https://github.com/IBM/nl2ltl) for natural language to LTL translation
- All evaluated LLM providers (OpenAI, Anthropic, Google, DeepSeek, Mistral)
