# code/agent/prompts.py


import json

# --- NEW: Helper function to provide context about cluster capacity ---
def _get_cluster_capacity_string(cluster_id: int) -> str:
    """Provides a human-readable description of the cluster's capacity."""
    if cluster_id == 0:
        return f"Cluster {cluster_id} (Low-Resource): These are the weakest clients. They may require smaller batch sizes or lower learning rates to train stably."
    elif cluster_id == 1:
        return f"Cluster {cluster_id} (Medium-Resource): These clients have moderate capacity."
    else:
        return f"Cluster {cluster_id} (High-Resource): These are the most powerful clients. They can likely handle larger batch sizes and more aggressive learning rates."


def _build_dynamic_search_space_description(search_space: dict) -> str:
    """Helper function to dynamically create a string describing the search space."""
    lines = []
    for top_level_key, hp_group in search_space.items():
        lines.append(f"\n**{top_level_key.replace('_', ' ').title()} Search Space:**")
        if isinstance(hp_group, dict):
            for name, config in hp_group.items():
                if isinstance(config, dict):
                    clean_name = name.replace('_', ' ').title()
                    param_type = config.get('type', 'unknown')
                    if param_type in ['float', 'int']:
                        line = f"- **{clean_name}** (`{param_type}`): from `{config.get('min')}` to `{config.get('max')}`"
                    elif param_type == 'choice':
                        line = f"- **{clean_name}** (`{param_type}`): from options `{config.get('values')}`"
                    else:
                        continue # Skip non-dict items like 'mu' which is handled separately
                    lines.append(line)
    # Handle mu separately if it exists at the top level
    if 'mu' in search_space and isinstance(search_space['mu'], dict):
         lines.append(f"\n**Global Hyperparameters:**")
         mu_config = search_space['mu']
         lines.append(f"- **Mu** (`float`): from `{mu_config.get('min')}` to `{mu_config.get('max')}`")

    return "\n".join(lines)


def get_hp_suggestion_prompt(
    client_id: int, cluster_id: int, model_name: str, dataset_name: str,
    hpo_report: dict, search_space: dict, analysis_from_last_round: dict | None = None,
    peer_history: list | None = None, arc_cfg: int = 0, total_layers: int = 0
) -> str:

    # --- Re-introduced string formatting logic ---
    history_lines = []
    if hpo_report:
        for epoch, report in sorted(hpo_report.items()):
            hps_str = ", ".join(f"{k}={v}" for k, v in report.get('hps_suggested', {}).items())
            acc = report.get('final_test_accuracy', 0.0)
            history_lines.append(f"- In Epoch {epoch + 1}: Used HPs ({hps_str}) and achieved Test Accuracy = {acc:.2f}%")
    else:
        history_lines.append("This is the first round for this client.")
    history_str = "\n".join(history_lines)

    analysis_str = "No analysis from the previous round is available."
    if analysis_from_last_round:
        analysis_str = analysis_from_last_round.get('decision_summary', 'Analysis available but summary is missing.')

    peer_history_str = "No peers in this cluster have run yet in this epoch."
    if peer_history:
        peer_lines = []
        for peer_run in peer_history:
            hps_str = ", ".join(f"{k}={v}" for k, v in peer_run.get('hps_used', {}).items())
            peer_lines.append(f"- Client {peer_run['client_id']} used HPs ({hps_str}) --> {peer_run['result_and_decision']}")
        peer_history_str = "\n".join(peer_lines)

    # --- Create a detailed example for the new, complex HP structure ---
    example_hps_dict = {
        "reasoning": "A detailed, multi-part reason for all HP choices.",
        "hps": {
            "client": {
                "learning_rate": 0.001, "weight_decay": 5e-05, "momentum": 0.9,
                "optimizer": "AdamW", "scheduler": "CosineAnnealingLR",
                "local_epochs": 2, "batch_size": 32, "dropout_rate": 0.2
            },
            "server": {
                "learning_rate": 0.005, "momentum": 0.9,
                "optimizer": "SGD", "scheduler": "StepLR"
            },
            "mu": 0.01
        }
    }
    example_json_str = json.dumps(example_hps_dict, indent=4)
    search_space_str = _build_dynamic_search_space_description(search_space)

    return f"""
You are an expert ML engineer specializing in Split Federated Learning. Your goal is to suggest a complete set of hyperparameters for both the client and the server.

{'-'*40}
**OVERALL CONTEXT**
- **Client ID:** {client_id}
- **Model:** {model_name} on {dataset_name}
- **Federated Scheme:** SplitFed with FedProx regularization (controlled by `mu`).

{'-'*40}
**CLIENT-SIDE CONTEXT & INSTRUCTIONS**
- **Client Capacity:** {_get_cluster_capacity_string(cluster_id)}. Low-resource clients may need smaller `batch_size`, fewer `local_epochs`, or lower `learning_rate`.
- **Client's Own History:**
{history_str}
- **Peer History (This Epoch):**
{peer_history_str}
- **Last Round's Analysis:** {analysis_str}
- **Client HPs to select:** `learning_rate`, `weight_decay`, `momentum`, `optimizer`, `scheduler`, `local_epochs`, `batch_size`, `dropout_rate`.
- **Your Task:** Based on all context, choose the best HPs for the client's local training.

{'-'*40}
**SERVER-SIDE CONTEXT & INSTRUCTIONS**
- **Model Split:** The client runs the first {arc_cfg} of {total_layers} layers. The server runs the remaining {total_layers - arc_cfg} layers.
- **Server Task:** A deeper server model (lower arc_cfg) is more complex and may benefit from a different `learning_rate` or `optimizer`.
- **Server HPs to select:** `learning_rate`, `momentum`, `optimizer`, `scheduler`.

{'-'*40}
**AVAILABLE SEARCH SPACE**
{search_space_str}

{'-'*40}
**OUTPUT FORMAT & INSTRUCTIONS**
- Return a single, valid JSON object with "reasoning" and "hps" keys.
- **CRITICAL:** You must adhere strictly to the available `values` for choice parameters and the `min`/`max` for numerical parameters listed in the search space.
- The "hps" object must contain "client", "server", and "mu" keys as shown in the example.

**EXAMPLE OUTPUT:**
{example_json_str}
"""

def get_analysis_prompt(
    client_id: int, cluster_id: int, model_name: str, dataset_name: str,
    results: dict, current_hps: dict, search_space: dict,
    global_epoch: int, local_epochs: int
) -> str:
    
    results_str = f"Final Test Accuracy = {results.get('test_acc', [0.0])[-1]:.2f}%"

    return f"""
You are an expert ML HPO analysis agent.

**TASK:**
Analyze the client's performance and provide a list of actions to refine the hyperparameter search space.
Return a JSON object with two keys: "reasoning" (a brief summary) and "actions" (a list of modification commands).

**CONTEXT:**
- Client: {client_id} (Epoch: {global_epoch + 1}, Capacity: {_get_cluster_capacity_string(cluster_id)})
- Task: {model_name} on {dataset_name}
- HPs Used: {json.dumps(current_hps)}
- Result: {results_str}

**ACTIONS SYNTAX:**
The "actions" key must be a list of JSON objects. Each object has four keys:
- "param": The base name of the hyperparameter (e.g., "learning_rate").
- "key": The property to change.
    - For numerical parameters (float, int), use "min" or "max".
    - For choice parameters (like optimizer or batch_size), you can only use "values".
- "value": The new value for that property.
- "target": Must be either "client_hps" or "server_hps".

**EXAMPLE OUTPUT:**
{{
    "reasoning": "The client is overfitting, so I will lower the max learning rate for the client and increase the server's momentum.",
    "actions": [
        {{
            "param": "learning_rate",
            "key": "max",
            "value": 0.001,
            "target": "client_hps"
        }},
        {{
            "param": "momentum",
            "key": "min",
            "value": 0.9,
            "target": "server_hps"
        }}
    ]
}}

**YOUR JSON OUTPUT:**
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