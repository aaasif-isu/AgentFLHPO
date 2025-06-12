import json
from .prompts import get_hp_suggestion_prompt
from .llm_api import call_llm

class HPAgent:
    """
    An agent that suggests hyperparameters by calling an LLM.
    """
    # --- THIS IS THE FIX ---
    # The method signature now correctly expects 'hpo_report' instead of 'history'.
    def suggest(self, client_id, cluster_id, model_name, dataset_name, hpo_report, search_space):
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
            search_space=search_space
        )

        response_json_str = call_llm(prompt)
        
        try:
            response_data = json.loads(response_json_str)
            suggested_hps = response_data.get("response", {})
            
            if not isinstance(suggested_hps, dict) or not suggested_hps:
                 raise json.JSONDecodeError("Response is not a valid, non-empty dictionary.", response_json_str, 0)
            
            print(f"--- [HP Agent Verdict for Client {client_id}] ---")
            print(f"Final suggested HPs to be used for training: {suggested_hps}")
            print("---")
                 
            return suggested_hps
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error: Could not parse HPs from LLM response: {e}")
            return {hp: config['initial'] for hp, config in search_space.items()}