<h1 align="center">🛠️🤖 MatchTIR: Fine-Grained Supervision for
<br>
Tool-Integrated Reasoning via Bipartite Matching

</h1>

The implementation for ACL 2026 Submission: MatchTIR: Fine-Grained Supervision for Tool-Integrated Reasoning via Bipartite Matching.

## 🛠️ Setup

### Environment Setup
- Run the command to install the packages required.
  ```bash
  pip install -r requirements.txt
  ```
- Download LLMs from Huggingface.

## ⚙️ Training Configuration
You can adjust the hyperparameters in `Scripts/run.sh`:
- `--custom_reward_function.name`: Choose between `compute_process_KM` (Hard) or `compute_process_ot` (Soft).
- `--actor_rollout_ref.model.path`: Path to your local LLM.
  
## 🚀 Quick Start
- Run the shell script to perform policy optimization.
  ```bash
  Bash Scripts/run.sh
  ```

## ☕️ Acknowledgement
We employ the [VeRL 0.3.1.dev](https://arxiv.org/abs/2409.19256) framework for training.
