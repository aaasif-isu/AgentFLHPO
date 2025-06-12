# # code/agent/workflow.py

import yaml
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from .hp_agent import HPAgent
from .analyzer_agent import AnalyzerAgent
from ssfl.trainer_utils import train_single_client

# 1. Define the GraphState to include the new context fields
class GraphState(TypedDict):
    # Context passed from the trainer
    client_id: int
    cluster_id: int
    model_name: str
    dataset_name: str
    training_args: Dict
    # State managed across rounds
    hps: Dict
    search_space: Dict
    hpo_report: Dict
    #history: List[float]
    # State for aggregation results
    results: Dict
    client_weights: Dict[str, Any]
    server_weights: Dict[str, Any]
    data_size: int

# 2. Define the Graph Nodes with corrected agent calls
def suggest_node(state: GraphState):
    print(f"--- Running HP Agent for Client {state['client_id']} ---")
    agent = HPAgent()
    
    # THE FIX: Pass all the required context from the state to the agent's suggest method
    suggested_hps = agent.suggest(
        client_id=state['client_id'],
        cluster_id=state['cluster_id'],
        model_name=state['model_name'],
        dataset_name=state['dataset_name'],
        # history=state['history'],
        hpo_report=state['hpo_report'],
        search_space=state['search_space']
    )
    print(f"Suggested HPs: {suggested_hps}")
    state['hps'] = suggested_hps
    return state

def train_node(state: GraphState):
    print(f"--- Starting Training for Client {state['client_id']} ---")
    w_c, w_s, sz, results = train_single_client(**state['training_args'], hps=state['hps'], cid=state['client_id'])
    state['results'] = results
    state['client_weights'] = w_c
    state['server_weights'] = w_s
    state['data_size'] = sz
    return state

def analyze_node(state: GraphState):
    print(f"--- Running Analyzer Agent for Client {state['client_id']} ---")
    agent = AnalyzerAgent()

    # THE FIX: Pass all the required context from the state to the agent's analyze method
    new_search_space = agent.analyze(
        client_id=state['client_id'],
        cluster_id=state['cluster_id'],
        model_name=state['model_name'],
        dataset_name=state['dataset_name'],
        results=state['results'],
        current_hps=state['hps'],
        search_space=state['search_space'],
        # hpo_report=state['hpo_report'],

        # Get these from the context passed into the workflow
        global_epoch=state['training_args'].get('global_epoch'), 
        local_epochs=state['training_args'].get('local_epochs')
    )
    print("Analyzer has proposed a new search space.")
    state['search_space'] = new_search_space
    #state['history'].append(state['results']['test_acc'][-1])
    return state

# 3. Wire up the graph (This part remains unchanged)
def create_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("suggest", suggest_node)
    workflow.add_node("train", train_node)
    workflow.add_node("analyze", analyze_node)
    workflow.set_entry_point("suggest")
    workflow.add_edge("suggest", "train")
    workflow.add_edge("train", "analyze")
    workflow.add_edge("analyze", END)
    return workflow.compile()