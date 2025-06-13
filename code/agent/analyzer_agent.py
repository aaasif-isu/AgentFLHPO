# In code/agent/analyzer_agent.py

import json
import copy
from .prompts import get_analysis_prompt
from .llm_api import call_llm

class AnalyzerAgent:
    """
    An agent that gets high-level analysis from an LLM, then uses
    that analysis to programmatically build the new search space.
    """

    def _apply_actions_to_search_space(self, search_space: dict, actions: list) -> dict:
        """
        Safely applies the LLM's suggested actions to the correct part of the search space,
        with strong validation to prevent corruption.
        """
        new_space = copy.deepcopy(search_space)
        if not isinstance(actions, list):
            print("  - WARNING: 'actions' field from LLM is not a list. No changes applied.")
            return new_space

        for action in actions:
            if not (isinstance(action, dict) and all(k in action for k in ['param', 'key', 'value', 'target'])):
                print(f"  - WARNING: Malformed action object skipped: {action}")
                continue

            target_space_key = action['target']
            param = action['param']
            key_to_change = action['key']
            value = action['value']
            
            # Check if the target and param are valid before proceeding
            if target_space_key not in new_space or param not in new_space[target_space_key]:
                print(f"  - WARNING: Invalid target/param '{target_space_key}/{param}'. Action skipped.")
                continue

            # --- THIS IS THE CRITICAL FIX ---
            # If the LLM wants to change the 'values' of a choice parameter,
            # we MUST ensure the new value is a list.
            if key_to_change == 'values':
                param_type = new_space[target_space_key][param].get('type')
                if param_type == 'choice' and not isinstance(value, list):
                    print(f"  - ERROR: Invalid value for '{param}' values. Expected a list but got {type(value)}. Action skipped.")
                    continue # Skip this invalid action to prevent corrupting the search space

            # Check if the key itself is valid for that parameter
            if key_to_change not in new_space[target_space_key][param]:
                print(f"  - WARNING: Invalid key '{key_to_change}' for param '{param}'. Action skipped.")
                continue
            
            # If all checks pass, apply the change
            print(f"  - Applying action: Setting {target_space_key}.{param}.{key_to_change} = {value}")
            new_space[target_space_key][param][key_to_change] = value
                
        return new_space

    def analyze(self, client_id, cluster_id, model_name, dataset_name, results, current_hps, search_space, global_epoch, local_epochs):
        """
        Calls the LLM to get reasoning and a list of actions, then builds the
        new search space and returns it along with the reasoning.
        """
        prompt = get_analysis_prompt(
            client_id=client_id, cluster_id=cluster_id, model_name=model_name,
            dataset_name=dataset_name, results=results, current_hps=current_hps,
            search_space=search_space, global_epoch=global_epoch, local_epochs=local_epochs
        )
        
        response_str = call_llm(prompt)
        
        try:
            response_data = json.loads(response_str)
            reasoning = response_data.get("reasoning", "No reasoning provided by LLM.")
            actions = response_data.get("actions", [])
            
            print(f"\n--- [Analyzer Reasoning for Client {client_id}] ---")
            print(f"  - LLM Reasoning: {reasoning}")
            
            new_search_space = self._apply_actions_to_search_space(search_space, actions)
            
            print("--- [Proposed New Search Space] ---")
            print(json.dumps(new_search_space, indent=2))
            print("-" * 45)

            final_reasoning_obj = {
                "performance_summary": reasoning,
                "decision_summary": f"Applied {len(actions)} action(s) to refine the search space."
            }
            return new_search_space, final_reasoning_obj

        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Warning: Analyzer for Client {client_id} could not generate a valid response. Error: {e}. Re-using old search space.")
            return search_space, None

class AnalyzerAgent3:
    """
    An agent that gets high-level analysis from an LLM, then uses
    that analysis to programmatically build the new search space.
    """

    def _apply_actions_to_search_space(self, search_space: dict, actions: list) -> dict:
        """
        Safely applies the LLM's suggested actions to the correct part of the search space.
        """
        new_space = copy.deepcopy(search_space)
        if not isinstance(actions, list):
            print("  - WARNING: 'actions' field from LLM is not a list. No changes applied.")
            return new_space

        for action in actions:
            if isinstance(action, dict) and all(k in action for k in ['param', 'key', 'value', 'target']):
                target_space_key = action['target'] # e.g., "client_hps"
                param = action['param']
                key_to_change = action['key']
                value = action['value']
                
                # Check if the target (e.g., 'client_hps') and param (e.g., 'learning_rate') are valid
                if target_space_key in new_space and param in new_space[target_space_key]:
                    if key_to_change in new_space[target_space_key][param]:
                        print(f"  - Applying action: Setting {target_space_key}.{param}.{key_to_change} = {value}")
                        new_space[target_space_key][param][key_to_change] = value
                    else:
                        print(f"  - WARNING: Invalid key '{key_to_change}' for param '{param}'. Action skipped.")
                else:
                    print(f"  - WARNING: Invalid target/param '{target_space_key}/{param}'. Action skipped.")
            else:
                print(f"  - WARNING: Malformed action object skipped: {action}")
                
        return new_space

    def _apply_actions_to_search_space_old(self, search_space: dict, actions: list) -> dict:
        """
        Safely applies the LLM's suggested actions to the current search space.
        """
        new_space = copy.deepcopy(search_space)
        if not isinstance(actions, list):
            print("  - WARNING: 'actions' field from LLM is not a list. No changes applied.")
            return new_space

        for action in actions:
            if isinstance(action, dict) and all(k in action for k in ['param', 'key', 'value']):
                param = action['param']
                key = action['key']
                value = action['value']
                
                if param in new_space:
                    if key in new_space[param]:
                        print(f"  - Applying action: Setting {param}.{key} = {value}")
                        new_space[param][key] = value
                    else:
                        print(f"  - WARNING: Invalid key '{key}' for param '{param}'. Action skipped.")
                else:
                    print(f"  - WARNING: Invalid param '{param}'. Action skipped.")
            else:
                print(f"  - WARNING: Malformed action object skipped: {action}")
                
        return new_space

    def analyze(self, client_id, cluster_id, model_name, dataset_name, results, current_hps, search_space, global_epoch, local_epochs):
        """
        Calls the LLM to get reasoning and a list of actions, then builds the
        new search space and returns it along with the reasoning.
        """
        prompt = get_analysis_prompt(
            client_id=client_id,
            cluster_id=cluster_id,
            model_name=model_name,
            dataset_name=dataset_name,
            results=results,
            current_hps=current_hps,
            search_space=search_space,
            global_epoch=global_epoch,
            local_epochs=local_epochs
        )
        
        response_str = call_llm(prompt)
        
        try:
            response_data = json.loads(response_str)
            reasoning = response_data.get("reasoning", "No reasoning provided by LLM.")
            actions = response_data.get("actions", [])
            
            print(f"\n--- [Analyzer Reasoning for Client {client_id}] ---")
            print(f"  - LLM Reasoning: {reasoning}")
            
            # Use Python to build the new search space from the actions
            new_search_space = self._apply_actions_to_search_space(search_space, actions)
            
            print("--- [Proposed New Search Space] ---")
            print(json.dumps(new_search_space, indent=2))
            print("-" * 45)

            # Create the final reasoning object to pass to the next agent
            final_reasoning_obj = {
                "performance_summary": reasoning,
                "decision_summary": f"Applied {len(actions)} action(s) to refine the search space."
            }

            return new_search_space, final_reasoning_obj

        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Warning: Analyzer for Client {client_id} could not generate a valid response. Error: {e}. Re-using old search space.")
            return search_space, None


# import json
# from .prompts import get_analysis_prompt, get_correction_prompt
# from .llm_api import call_llm


# class AnalyzerAgent:
#     """
#     An agent that analyzes the most recent training results and proposes
#     a refined search space. Includes self-correction and logging.
#     """

#     def _validate_and_parse(self, response_json_str: str) -> tuple[dict, dict] | None:
#         """
#         Tries to parse the LLM's JSON response and validates the structure
#         that includes a structured reasoning object and the new search space.
#         """
#         try:
#             response_data = json.loads(response_json_str)
#             response_content = response_data.get("response", {})
            
#             reasoning_obj = response_content.get("reasoning")
#             new_search_space = response_content.get("new_search_space")
            
#             # Check if the main keys exist and are the correct type
#             if isinstance(reasoning_obj, dict) and isinstance(new_search_space, dict):
#                 # Check if the sub-keys for reasoning exist
#                 required_reasoning_keys = ['performance_summary', 'contextual_analysis', 'strategy', 'decision_summary']
#                 if all(key in reasoning_obj for key in required_reasoning_keys):
#                     return reasoning_obj, new_search_space
#             return None
#         except (json.JSONDecodeError, AttributeError):
#             return None

#     def analyze(self, client_id, cluster_id, model_name, dataset_name, results, current_hps, search_space, global_epoch, local_epochs):
#         """
#         Calls the LLM with the full context of the LATEST round and prints the 
#         reasoning before returning the new search space.
#         """
#         # This call correctly passes all the necessary arguments to the prompt function.
#         original_prompt = get_analysis_prompt(
#             client_id=client_id,
#             cluster_id=cluster_id,
#             model_name=model_name,
#             dataset_name=dataset_name,
#             results=results, # The results from the most recent run
#             current_hps=current_hps,
#             search_space=search_space,
#             global_epoch=global_epoch,
#             local_epochs=local_epochs
#         )
        
#         # Call the LLM and attempt to parse the response
#         response_str = call_llm(original_prompt)
#         parsed_data = self._validate_and_parse(response_str)
        
#         if parsed_data:
#             reasoning, new_search_space = parsed_data
#             print(f"\n--- [Analyzer Reasoning for Client {client_id}] ---")
#             print(f"  - Performance: {reasoning.get('performance_summary', 'N/A')}")
#             print(f"  - Context:     {reasoning.get('contextual_analysis', 'N/A')}")
#             print(f"  - Strategy:    {reasoning.get('strategy', 'N/A')}")
#             print(f"  - Decision:    {reasoning.get('decision_summary', 'N/A')}")
#             print("--- [Proposed New Search Space] ---")
#             print(json.dumps(new_search_space, indent=2))
#             print("-" * 45)
#             return new_search_space, reasoning
            
#         # If the first attempt fails, we could add self-correction back here if needed.
#         # For simplicity, we will fall back to the old search space.
#         print(f"Warning: Analyzer for Client {client_id} could not generate a valid structured response. Re-using old search space.")
#         return search_space, None