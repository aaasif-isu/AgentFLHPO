import json
from .prompts import get_hp_suggestion_prompt
from .llm_api import call_llm

class HPAgent:
    """
    An agent that suggests hyperparameters by calling an LLM.
    """
    # --- THIS IS THE FIX ---
    # The method signature now correctly expects 'hpo_report' instead of 'history'.

    #def suggest(self, client_id, cluster_id, model_name, dataset_name, hpo_report, search_space):

    # --- MODIFICATION 5: Add the new parameter to the method signature ---
    def suggest(self, client_id, cluster_id, model_name, dataset_name, hpo_report, search_space, analysis_from_last_round: dict | None = None, peer_history: list | None = None):
 
        """
        Calls the LLM to suggest hyperparameters, using the structured HPO report.
        """
        # Pass the 'hpo_report' to the prompt function
        prompt = get_hp_suggestion_prompt(
            client_id=client_id,
            cluster_id=cluster_id,
            model_name=model_name,
            dataset_name=dataset_name,
            hpo_report=hpo_report,
            search_space=search_space,
            # --- MODIFICATION 6: Pass the analysis to the prompt function ---
            analysis_from_last_round=analysis_from_last_round,
            peer_history=peer_history

        )

        response_json_str = call_llm(prompt)
        
        try:
            response_data = json.loads(response_json_str)
            # llm_reasoning = response_data.get("response", {}).get("reasoning", "No reasoning provided.")
            # suggested_hps = response_data.get("response", {}).get("hps", {})
            llm_reasoning = response_data.get("reasoning", "No reasoning provided.")
            suggested_hps = response_data.get("hps", {})

            
            if not isinstance(suggested_hps, dict) or not suggested_hps:
                 raise json.JSONDecodeError("Response is not a valid, non-empty dictionary.", response_json_str, 0)
            
            # --- START: NEW VALIDATION AND CLAMPING FIX ---
            print(f"--- [HP Agent Verdict for Client {client_id}] ---")
            print(f"  - HP Agent Reasoning: {llm_reasoning}")
            print(f"LLM Suggested HPs (raw): {suggested_hps}")

            validated_hps = {}
            for hp, value in suggested_hps.items():
                if hp in search_space:
                    config = search_space[hp]
                    param_type = config.get('type')

                    if param_type in ['float', 'int']:
                        # Clamp the value within the min/max bounds
                        clamped_value = max(config['min'], min(config['max'], value))
                        if clamped_value != value:
                            print(f"  - WARNING: Clamped '{hp}' from {value} to {clamped_value}")
                        validated_hps[hp] = clamped_value
                    
                    elif param_type == 'choice':
                        # Ensure the value is one of the valid choices
                        if value not in config['values']:
                            valid_choice = random.choice(config['values'])
                            print(f"  - WARNING: Invalid choice for '{hp}'. Got '{value}', using random choice '{valid_choice}'")
                            validated_hps[hp] = valid_choice
                        else:
                            validated_hps[hp] = value
                    else:
                        validated_hps[hp] = value # For params without defined types
                else:
                    validated_hps[hp] = value # Keep param if not in search space (e.g., 'mu')

            print(f"Final suggested HPs (validated): {validated_hps}")
            print("---")
            return validated_hps
            # --- END: NEW VALIDATION AND CLAMPING FIX ---

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error: Could not parse HPs from LLM response: {e}")
            return {hp: config['initial'] for hp, config in search_space.items()}