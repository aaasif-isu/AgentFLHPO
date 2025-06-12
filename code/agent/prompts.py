# code/agent/prompts.py


import json

def _build_dynamic_search_space_description(search_space: dict) -> str:
    """Helper function to dynamically create a string describing the search space."""
    lines = []
    for name, config in search_space.items():
        clean_name = name.replace('_', ' ').title()
        param_type = config.get('type', 'unknown')
        if param_type in ['float', 'int']:
            line = f"- **{clean_name}** (`{param_type}`): from `{config.get('min')}` to `{config.get('max')}`"
        elif param_type == 'choice':
            line = f"- **{clean_name}** (`{param_type}`): from options `{config.get('values')}`"
        else:
            line = f"- **{clean_name}**: {config}"
        lines.append(line)
    return "\n".join(lines)


def get_hp_suggestion_prompt(
    client_id: int,
    cluster_id: int,
    model_name: str,
    dataset_name: str,
    hpo_report: dict, # Changed from 'history' to 'hpo_report'
    search_space: dict
) -> str:
    """
    Generates a detailed prompt, now formatting the structured HPO report.
    """
    # --- NEW: Format the epoch-by-epoch history for the prompt ---
    history_lines = []
    if hpo_report:
        # for epoch, report in sorted(hpo_report.items(), key=lambda item: int(item[0].split(': ')[-1])):
        #     hps_str = ", ".join(f"{k}={v}" for k, v in report['hps_suggested'].items())
        #     acc = report['final_test_accuracy']
        #     history_lines.append(f"- In Epoch {epoch}: Used HPs ({hps_str}) and achieved Test Accuracy = {acc:.2f}%")
        for epoch, report in sorted(hpo_report.items(), key=lambda item: item[0]):
            hps_str = ", ".join(f"{k}={v}" for k, v in report['hps_suggested'].items())
            acc = report['final_test_accuracy']
            # We add 1 to the epoch for more human-readable display (e.g., "Epoch 1" instead of "Epoch 0")
            history_lines.append(f"- In Epoch {epoch + 1}: Used HPs ({hps_str}) and achieved Test Accuracy = {acc:.2f}%")
    else:
        history_lines.append("This is the first round for this client.")
    history_str = "\n".join(history_lines)
    
    search_space_str = _build_dynamic_search_space_description(search_space)
    example_hps = {name: config.get('initial') for name, config in search_space.items()}
    example_json_str = json.dumps(example_hps, indent=4)

    return f"""
You are an expert ML engineer specializing in Federated Learning.

**Your Goal:**
Suggest the single best set of hyperparameters for the specified client.

**Context for Your Decision:**
- **Client ID:** {client_id}
- **Client Cluster:** Cluster {cluster_id}
- **Model:** {model_name}
- **Dataset:** {dataset_name}
- **Client's Performance History:**
{history_str}

**Available Hyperparameter Search Space:**
{search_space_str}

**Reasoning Instructions:**
1.  Analyze the epoch-by-epoch performance history. Did accuracy improve or stagnate with the last set of HPs?
2.  If you see signs of overfitting, suggest changes to regularize the model (e.g., decrease `learning_rate`, increase `weight_decay`).
3.  If you see signs of underfitting, suggest changes to accelerate learning (e.g., increase `learning_rate`).
4.  Your suggested values MUST fall within the bounds of the provided search space.

**Output Format:**
Return your response as a single, valid JSON object.

**Example of a valid output format:**
{example_json_str}
"""


def get_analysis_prompt(
    client_id: int,
    cluster_id: int,
    model_name: str,
    dataset_name: str,
    results: dict,
    current_hps: dict,
    search_space: dict,
    global_epoch: int,
    local_epochs: int
) -> str:
    """
    Generates a more advanced, context-aware prompt for the Analyzer Agent
    with a clearer persona and more detailed reasoning instructions.
    """
    actual_input = {
      "client_id": client_id,
      "cluster_id": cluster_id,
      "global_epoch": global_epoch,
      "local_epochs_run": local_epochs,
      "model_name": model_name,
      "dataset_name": dataset_name,
      "results_from_last_round": results,
      "hyperparameters_used": current_hps,
      "current_search_space": search_space
    }

    return f"""
You are an expert Machine Learning engineer specializing in Federated Learning (FL).
Your task is to perform per-client hyperparameter optimization by analyzing a client's performance and proposing a refined search space for its next round of training.

You must consider all the provided context, including the client's cluster, the model architecture, the dataset, and its performance history to make an informed decision.

**CRITICAL OUTPUT INSTRUCTION:** Your entire output MUST be a single, valid JSON object containing two keys: "reasoning" and "new_search_space".

**Reasoning Instructions:**
The "reasoning" key must itself be a JSON object with the following four sub-keys:
1.  `performance_summary`: A one-sentence summary of the client's performance (e.g., "The client is overfitting, with a large gap between train and test accuracy.").
2.  `contextual_analysis`: A one-sentence analysis explaining HOW the broader context influences performance. (e.g., "The powerful {model_name} model is prone to overfitting on the simpler {dataset_name} dataset, and this client seems sensitive to the learning rate.")
3.  `strategy`: A one-sentence description of the strategy you will use to create the new search space. (e.g., "I will make the search space more conservative to combat overfitting and stabilize training.")
4.  `decision_summary`: A one-sentence summary of the specific changes you are making to the search space. (e.g., "I am lowering the max learning rate and increasing the minimum weight decay.")

The "new_search_space" key must contain the new search space dictionary, adjusted according to your strategy.

---
**INPUT FOR THIS TASK:**
{json.dumps(actual_input, indent=2)}

**YOUR OUTPUT (as a single JSON object):**
"""

def get_correction_prompt(original_prompt: str, bad_response: str, error_message: str) -> str:
    """
    Generates a prompt to ask the LLM to correct its previous invalid output.
    This function is already general and needs no changes.
    """
    # This prompt is already robust and does not need to be changed.
    return f"""
You are a helpful AI assistant. Your previous response was not correctly formatted, which caused an error. You must try again and strictly follow the output format instructions.

---
**Original Request:**
{original_prompt}
---
**Your Previous Invalid Response:**
{bad_response}
---
**The Error Caused by Your Response:**
{error_message}
---

**Instruction:**
Please try again. Generate a new response that strictly adheres to the requested JSON format and schema from the original request. Do not include any extra text, explanations, or markdown code blocks.
"""