# Safe Deployment of LLMs in Robotics: NL2LTL with Zonopy
Repository name: safe-llm-nl2ltl-zonopy

This repository contains the implementation of my research project **“Safe Deployment of LLMs in Robotics: NL2LTL with Zonopy”**, conducted during my research stay at TUM.  

## Overview
The project integrates **NL2LTL** with **Zonopy** to evaluate the safe deployment of large language models (LLMs) in robotic environments.  
Key contributions include:  
- Integration of NL2LTL with Zonopy for end-to-end evaluation  
- Automatic dataset generation from Zonopy trajectories (static & dynamic)  
- Coverage of **11 Declare/LTL operators** directly monitorable in Zonopy  
- Benchmarking of multiple LLMs (GPT, Claude, DeepSeek, Mistral, Gemma)  
- Evaluation using **syntax, region, and full accuracy** metrics  
- Comprehensive comparison of static vs. dynamic environments  

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
   git clone https://github.com/Ezgikorkutt/safe-llm-nl2ltl-zonopy.git
   cd safe-llm-nl2ltl-zonopy
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
- [Zonopy](https://github.com/loizoshad/zonopy) for set-based reachability
- [NL2LTL](https://github.com/IBM/nl2ltl) for natural language to LTL translation
- All evaluated LLM providers (OpenAI, Anthropic, Google, DeepSeek, Mistral)
