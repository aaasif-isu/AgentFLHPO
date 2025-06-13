# # # code/agent/workflow.py

# code/agent/workflow.py

import json
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Dict, Any
from agent.hp_agent import HPAgent
from agent.analyzer_agent import AnalyzerAgent
from ssfl.trainer_utils import train_single_client

# Define the state for the graph
class HPOState(TypedDict):
    client_id: int
    cluster_id: int
    model_name: str
    dataset_name: str
    hpo_report: dict
    search_space: dict
    training_args: dict
    # --- CHANGE 1: Add last_analysis to the state definition ---
    last_analysis: Dict[str, Any] | None
    peer_history: list
    
    # Fields that will be added during the graph execution
    hps: Dict[str, Any]
    results: Dict[str, Any]
    client_weights: Any
    server_weights: Any
    data_size: int

# Instantiate agents
hp_agent = HPAgent()
analyzer_agent = AnalyzerAgent()

# Define the nodes for the graph
def suggest_node(state: HPOState) -> HPOState:
    print(f"\n>>> Graph Node: SUGGEST for Client {state['client_id']}")
    hps = hp_agent.suggest(
        client_id=state['client_id'],
        cluster_id=state['cluster_id'],
        model_name=state['model_name'],
        dataset_name=state['dataset_name'],
        hpo_report=state['hpo_report'],
        search_space=state['search_space'],
        # --- CHANGE 2: Pass the analysis from the state to the agent ---
        analysis_from_last_round=state.get('last_analysis'), 
        peer_history=state.get('peer_history')
    )
    state['hps'] = hps
    return state

def train_node(state: HPOState) -> HPOState:
    print(f"\n>>> Graph Node: TRAIN for Client {state['client_id']}")
    w_c, w_s, sz, results = train_single_client(
        **state['training_args'], 
        hps=state['hps'], 
        cid=state['client_id']
    )
    state['client_weights'] = w_c
    state['server_weights'] = w_s
    state['data_size'] = sz
    state['results'] = results
    return state

def analyze_node(state: HPOState) -> HPOState:
    print(f"\n>>> Graph Node: ANALYZE for Client {state['client_id']}")
    # --- CHANGE 3: Capture both outputs from the analyzer ---
    new_search_space, reasoning = analyzer_agent.analyze(
        client_id=state['client_id'],
        cluster_id=state['cluster_id'],
        model_name=state['model_name'],
        dataset_name=state['dataset_name'],
        results=state['results'],
        current_hps=state['hps'],
        search_space=state['search_space'],
        global_epoch=state['training_args']['global_epoch'],
        local_epochs=state['training_args']['local_epochs']
    )
    # --- CHANGE 4: Update the state with both new values ---
    state['search_space'] = new_search_space
    state['last_analysis'] = reasoning # This is now passed to the final state
    return state

# Create and compile the graph
def create_graph():
    workflow = StateGraph(HPOState)
    workflow.add_node("suggest", suggest_node)
    workflow.add_node("train", train_node)
    workflow.add_node("analyze", analyze_node)

    workflow.set_entry_point("suggest")
    workflow.add_edge("suggest", "train")
    workflow.add_edge("train", "analyze")
    workflow.add_edge("analyze", END)
    
    return workflow.compile()

# import yaml
# from typing import TypedDict, List, Dict, Any
# from langgraph.graph import StateGraph, END
# from .hp_agent import HPAgent
# from .analyzer_agent import AnalyzerAgent
# from ssfl.trainer_utils import train_single_client

# # 1. Define the GraphState to include the new context fields
# class GraphState(TypedDict):
#     # Context passed from the trainer
#     client_id: int
#     cluster_id: int
#     model_name: str
#     dataset_name: str
#     training_args: Dict
#     # State managed across rounds
#     hps: Dict
#     search_space: Dict
#     hpo_report: Dict
#     #history: List[float]
#     # State for aggregation results
#     results: Dict
#     client_weights: Dict[str, Any]
#     server_weights: Dict[str, Any]
#     data_size: int

# # 2. Define the Graph Nodes with corrected agent calls
# def suggest_node(state: GraphState):
#     print(f"--- Running HP Agent for Client {state['client_id']} ---")
#     agent = HPAgent()
    
#     # THE FIX: Pass all the required context from the state to the agent's suggest method
#     suggested_hps = agent.suggest(
#         client_id=state['client_id'],
#         cluster_id=state['cluster_id'],
#         model_name=state['model_name'],
#         dataset_name=state['dataset_name'],
#         # history=state['history'],
#         hpo_report=state['hpo_report'],
#         search_space=state['search_space']
#     )
#     print(f"Suggested HPs: {suggested_hps}")
#     state['hps'] = suggested_hps
#     return state

# def train_node(state: GraphState):
#     print(f"--- Starting Training for Client {state['client_id']} ---")
#     w_c, w_s, sz, results = train_single_client(**state['training_args'], hps=state['hps'], cid=state['client_id'])
#     state['results'] = results
#     state['client_weights'] = w_c
#     state['server_weights'] = w_s
#     state['data_size'] = sz
#     return state

# def analyze_node(state: GraphState):
#     print(f"--- Running Analyzer Agent for Client {state['client_id']} ---")
#     agent = AnalyzerAgent()

#     # THE FIX: Pass all the required context from the state to the agent's analyze method
#     new_search_space = agent.analyze(
#         client_id=state['client_id'],
#         cluster_id=state['cluster_id'],
#         model_name=state['model_name'],
#         dataset_name=state['dataset_name'],
#         results=state['results'],
#         current_hps=state['hps'],
#         search_space=state['search_space'],
#         # hpo_report=state['hpo_report'],

#         # Get these from the context passed into the workflow
#         global_epoch=state['training_args'].get('global_epoch'), 
#         local_epochs=state['training_args'].get('local_epochs')
#     )
#     print("Analyzer has proposed a new search space.")
#     state['search_space'] = new_search_space
#     #state['history'].append(state['results']['test_acc'][-1])
#     return state

# # 3. Wire up the graph (This part remains unchanged)
# def create_graph():
#     workflow = StateGraph(GraphState)
#     workflow.add_node("suggest", suggest_node)
#     workflow.add_node("train", train_node)
#     workflow.add_node("analyze", analyze_node)
#     workflow.set_entry_point("suggest")
#     workflow.add_edge("suggest", "train")
#     workflow.add_edge("train", "analyze")
#     workflow.add_edge("analyze", END)
#     return workflow.compile()