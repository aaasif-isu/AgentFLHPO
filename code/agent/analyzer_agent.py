import json
from .prompts import get_analysis_prompt, get_correction_prompt
from .llm_api import call_llm


class AnalyzerAgent:
    """
    An agent that analyzes the most recent training results and proposes
    a refined search space. Includes self-correction and logging.
    """

    def _validate_and_parse(self, response_json_str: str) -> tuple[dict, dict] | None:
        """
        Tries to parse the LLM's JSON response and validates the structure
        that includes a structured reasoning object and the new search space.
        """
        try:
            response_data = json.loads(response_json_str)
            response_content = response_data.get("response", {})
            
            reasoning_obj = response_content.get("reasoning")
            new_search_space = response_content.get("new_search_space")
            
            # Check if the main keys exist and are the correct type
            if isinstance(reasoning_obj, dict) and isinstance(new_search_space, dict):
                # Check if the sub-keys for reasoning exist
                required_reasoning_keys = ['performance_summary', 'contextual_analysis', 'strategy', 'decision_summary']
                if all(key in reasoning_obj for key in required_reasoning_keys):
                    return reasoning_obj, new_search_space
            return None
        except (json.JSONDecodeError, AttributeError):
            return None

    def analyze(self, client_id, cluster_id, model_name, dataset_name, results, current_hps, search_space, global_epoch, local_epochs):
        """
        Calls the LLM with the full context of the LATEST round and prints the 
        reasoning before returning the new search space.
        """
        # This call correctly passes all the necessary arguments to the prompt function.
        original_prompt = get_analysis_prompt(
            client_id=client_id,
            cluster_id=cluster_id,
            model_name=model_name,
            dataset_name=dataset_name,
            results=results, # The results from the most recent run
            current_hps=current_hps,
            search_space=search_space,
            global_epoch=global_epoch,
            local_epochs=local_epochs
        )
        
        # Call the LLM and attempt to parse the response
        response_str = call_llm(original_prompt)
        parsed_data = self._validate_and_parse(response_str)
        
        if parsed_data:
            reasoning, new_search_space = parsed_data
            print(f"\n--- [Analyzer Reasoning for Client {client_id}] ---")
            print(f"  - Performance: {reasoning.get('performance_summary', 'N/A')}")
            print(f"  - Context:     {reasoning.get('contextual_analysis', 'N/A')}")
            print(f"  - Strategy:    {reasoning.get('strategy', 'N/A')}")
            print(f"  - Decision:    {reasoning.get('decision_summary', 'N/A')}")
            print("--- [Proposed New Search Space] ---")
            print(json.dumps(new_search_space, indent=2))
            print("-" * 45)
            return new_search_space
            
        # If the first attempt fails, we could add self-correction back here if needed.
        # For simplicity, we will fall back to the old search space.
        print(f"Warning: Analyzer for Client {client_id} could not generate a valid structured response. Re-using old search space.")
        return search_space