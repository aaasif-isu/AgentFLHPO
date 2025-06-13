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


def get_hp_suggestion_prompt_1(
    client_id: int,
    cluster_id: int,
    model_name: str,
    dataset_name: str,
    hpo_report: dict,
    search_space: dict,
    analysis_from_last_round: dict | None = None,
    peer_history: list | None = None
) -> str:
    
    # --- This section is now highly simplified to ensure stability ---
    
    peer_history_str = "None"
    if peer_history:
        peer_lines = []
        for peer_run in peer_history:
            peer_lines.append(f"- Peer {peer_run['client_id']} Result: {peer_run['result_and_decision']}")
        peer_history_str = "\n".join(peer_lines)
    
    analysis_str = "None"
    if analysis_from_last_round:
        analysis_str = analysis_from_last_round.get('decision_summary', 'N/A')

    example_hps_dict = {"reasoning": "A one-sentence reason.", "hps": {"learning_rate": 0.001}}
    example_json_str = json.dumps(example_hps_dict, indent=2)

    return f"""
You are an expert ML HPO agent. Your goal is to suggest hyperparameters.

**CONTEXT:**
- Client: {client_id} (Capacity: {_get_cluster_capacity_string(cluster_id)})
- Model: {model_name}
- Dataset: {dataset_name}
- Last Round Analyzer Decision: {analysis_str}
- Peer Results This Epoch:
{peer_history_str}
- Search Space: {json.dumps(search_space)}

**INSTRUCTIONS:**
1.  Analyze all context.
2.  Suggest HPs inside the search space.
3.  Return a JSON object with "reasoning" and "hps" keys.

**EXAMPLE:**
{example_json_str}
"""

def get_hp_suggestion_prompt(
    client_id: int,
    cluster_id: int,
    model_name: str,
    dataset_name: str,
    hpo_report: dict, # Changed from 'history' to 'hpo_report'
    search_space: dict,
    analysis_from_last_round: dict | None = None,
    # --- NEW: Add peer_history to the function signature ---
    peer_history: list | None = None
) -> str:
    """
    Generates a detailed prompt, now formatting the structured HPO report.
    """
    # --- NEW: Format the epoch-by-epoch history for the prompt ---
    history_lines = []
    if hpo_report:
        for epoch, report in sorted(hpo_report.items(), key=lambda item: item[0]):
            hps_str = ", ".join(f"{k}={v}" for k, v in report['hps_suggested'].items())
            acc = report['final_test_accuracy']
            # We add 1 to the epoch for more human-readable display (e.g., "Epoch 1" instead of "Epoch 0")
            history_lines.append(f"- In Epoch {epoch + 1}: Used HPs ({hps_str}) and achieved Test Accuracy = {acc:.2f}%")
    else:
        history_lines.append("This is the first round for this client.")
    history_str = "\n".join(history_lines)

     # --- MODIFICATION 4: Create a new section for the analysis ---
    analysis_str = "No analysis from the previous round is available."
    if analysis_from_last_round:
        analysis_str = f"""
- **Performance Summary:** {analysis_from_last_round.get('performance_summary')}
- **Contextual Analysis:** {analysis_from_last_round.get('contextual_analysis')}
- **Strategy:** {analysis_from_last_round.get('strategy')}
- **Decision Summary:** {analysis_from_last_round.get('decision_summary')}
"""

    search_space_str = _build_dynamic_search_space_description(search_space)
    example_hps = {
        "reasoning": "A brief, one-sentence explanation for the chosen hyperparameters.",
        "hps": {name: config.get('initial') for name, config in search_space.items()}
    }
    #example_hps = {name: config.get('initial') for name, config in search_space.items()}
    example_json_str = json.dumps(example_hps, indent=4)

    # --- NEW: Format the peer history for the prompt ---
    peer_history_str = "No peers in this cluster have run yet in this epoch."
    if peer_history:
        peer_lines = []
        for peer_run in peer_history:
            hps_str = ", ".join(f"{k}={v}" for k, v in peer_run['hps_used'].items())
            #peer_lines.append(f"- Client {peer_run['client_id']} used HPs ({hps_str}) and its performance was summarized as: '{peer_run['result']}'")
            peer_lines.append(f"- Client {peer_run['client_id']} used HPs ({hps_str}) --> {peer_run['result_and_decision']}")

        peer_history_str = "\n".join(peer_lines)

    return f"""
You are an expert ML engineer specializing in Federated Learning.

**Your Goal:**
Suggest the single best set of hyperparameters and provide a brief justification.

**Context for Your Decision:**
- **Client ID:** {client_id}
- Client Cluster: Cluster {cluster_id}
- Model: {model_name}
- Dataset: {dataset_name}

- **Peer Performance in this Cluster (Current Epoch):**
{peer_history_str}

- **Client's Own Performance History:**
{history_str}

**Analysis from Last Round:**
{analysis_str}

**Available Hyperparameter Search Space:**
{search_space_str}

**REASONING INSTRUCTIONS:**
1.  Learn from your peers! If a peer tried something that resulted in overfitting, try to avoid making the same mistake.
2.  Consider all context, especially the Client Capacity and peer performance.
3.  Pay close attention to the "Analysis from Last Round". Your suggestion must align with its strategy.
4.  Based on all the context, formulate a one-sentence reason for your specific HP choices. For example: "Lowering the learning rate and increasing weight decay to combat the observed overfitting, as suggested by the analyzer."
5.  Your suggested values MUST fall within the bounds of the provided search space.

**OUTPUT FORMAT:**
Return your response as a single, valid JSON object with two keys: "reasoning" and "hps".

**Example of a valid output format:**
{example_json_str}
"""


    
#     search_space_str = _build_dynamic_search_space_description(search_space)
#     example_hps = {name: config.get('initial') for name, config in search_space.items()}
#     example_json_str = json.dumps(example_hps, indent=4)

#     return f"""
# You are an expert ML engineer specializing in Federated Learning.

# **Your Goal:**
# Suggest the single best set of hyperparameters for the specified client.

# **Context for Your Decision:**
# - **Client ID:** {client_id}
# - **Client Cluster:** Cluster {cluster_id}
# - **Model:** {model_name}
# - **Dataset:** {dataset_name}
# - **Client's Performance History:**
# {history_str}

# **Available Hyperparameter Search Space:**
# {search_space_str}

# **Reasoning Instructions:**
# 1.  Analyze the epoch-by-epoch performance history. Did accuracy improve or stagnate with the last set of HPs?
# 2.  If you see signs of overfitting, suggest changes to regularize the model (e.g., decrease `learning_rate`, increase `weight_decay`).
# 3.  If you see signs of underfitting, suggest changes to accelerate learning (e.g., increase `learning_rate`).
# 4.  Your suggested values MUST fall within the bounds of the provided search space.

# **Output Format:**
# Return your response as a single, valid JSON object.

# **Example of a valid output format:**
# {example_json_str}
# """


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
The "actions" key must be a list of JSON objects, where each object has three keys:
- "param": The name of the hyperparameter to change (e.g., "learning_rate").
- "key": The specific property to change ("min", "max", or "values").
- "value": The new value for that property.

**EXAMPLE OUTPUT:**
{{
    "reasoning": "The client is overfitting, so I will lower the max learning rate and increase the minimum weight decay to add regularization.",
    "actions": [
        {{
            "param": "learning_rate",
            "key": "max",
            "value": 0.001
        }},
        {{
            "param": "weight_decay",
            "key": "min",
            "value": 0.0001
        }}
    ]
}}

**CURRENT SEARCH SPACE (for reference only):**
{json.dumps(search_space, indent=2)}

**YOUR JSON OUTPUT:**
"""


# --- UPDATED and SIMPLIFIED AnalyzerAgent Prompt ---
# This is the simplified, more robust version of the Analyzer prompt
def get_analysis_prompt_2(
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
    
    results_str = f"Final Test Accuracy = {results.get('test_acc', [0.0])[-1]:.2f}%"

    return f"""
You are an expert Machine Learning engineer specializing in Federated Learning (FL). Your task is to analyze a client's performance and propose a refined search space.

**CRITICAL OUTPUT INSTRUCTION:** Your entire output MUST be a single, valid JSON object with two keys: "reasoning" and "new_search_space".

{'-'*20}
**CONTEXT FOR THIS TASK:**
- **Client Details:** Client {client_id}, running in global epoch {global_epoch + 1}.
- **Client Capacity:** {_get_cluster_capacity_string(cluster_id)}
- **Model & Data:** The client trained a {model_name} on the {dataset_name} dataset.
- **Hyperparameters Used:** {json.dumps(current_hps)}
- **Performance Result:** {results_str}
- **Current Search Space:** The agent should refine the following search space: {json.dumps(search_space)}
{'-'*20}

**REASONING INSTRUCTIONS:**
Based on all the context above, provide your analysis. The "reasoning" key must be a JSON object with four sub-keys:
1.  `performance_summary`: A one-sentence summary of the client's performance (e.g., "The client is overfitting.").
2.  `contextual_analysis`: A one-sentence analysis explaining HOW the context (like client capacity or model type) influences performance.
3.  `strategy`: A one-sentence description of the strategy for the new search space.
4.  `decision_summary`: A one-sentence summary of the specific changes you are making to the search space.

**YOUR OUTPUT (as a single JSON object):**
"""

def get_analysis_prompt_1(
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