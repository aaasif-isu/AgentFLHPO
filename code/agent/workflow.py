# # # code/agent/workflow.py

# code/agent/workflow.py

import json
from langgraph.graph import StateGraph, END
from typing import TypedDict, Any, Dict, List
from torch.utils.data import DataLoader

# Assuming hp_agent and analyzer_agent are in the same directory
from .hp_agent import HPAgent
from .analyzer_agent import AnalyzerAgent
# Correctly import from the ssfl package structure
from ssfl.trainer_utils import train_single_client

# Define the complete and correct state for the graph
class HPOState(TypedDict):
    # Context passed from trainer.py
    client_id: int
    cluster_id: int
    model_name: str
    dataset_name: str
    training_args: dict
    peer_history: List[dict]
    arc_cfg: int
    total_layers: int
    train_subsets: List[Any]
    
    # Persistent state for the client
    hpo_report: dict
    search_space: dict
    last_analysis: Dict[str, Any] | None
    
    # Fields that will be added during the graph execution
    hps: Dict[str, Any]
    train_loader: DataLoader # Add this key for the dynamic loader
    results: Dict[str, Any]
    client_weights: Any
    server_weights: Any
    data_size: int

# Instantiate agents
hp_agent = HPAgent()
analyzer_agent = AnalyzerAgent()

# --- NODE DEFINITIONS ---

def suggest_node(state: HPOState) -> HPOState:
    print(f"\n>>> Graph Node: SUGGEST for Client {state['client_id']}")
    hps = hp_agent.suggest(
        client_id=state['client_id'],
        cluster_id=state['cluster_id'],
        model_name=state['model_name'],
        dataset_name=state['dataset_name'],
        hpo_report=state['hpo_report'],
        search_space=state['search_space'],
        analysis_from_last_round=state.get('last_analysis'),
        peer_history=state.get('peer_history'),
        arc_cfg=state.get('arc_cfg'),
        total_layers=state.get('total_layers')
    )
    state['hps'] = hps
    return state

def prepare_loader_node(state: HPOState) -> HPOState:
    print(f">>> Graph Node: PREPARE_LOADER for Client {state['client_id']}")
    cid = state['client_id']
    client_hps = state.get('hps', {}).get('client', {})
    batch_size = client_hps.get('batch_size', 32)
    dataset = state['train_subsets'][cid]

    final_batch_size = batch_size
    drop_last_flag = True

    # If the entire dataset is smaller than the batch size, DataLoader would still
    # produce one small batch. We must ensure this batch is not size 1.
    if len(dataset) < batch_size and len(dataset) <= 1:
        # This is an edge case that should not happen with your data check, but it's a critical safeguard.
        raise ValueError(f"FATAL: Client {cid} dataset has size {len(dataset)}, which is <= 1. Cannot train.")

    #print(f"  - Creating DataLoader for Client {cid} with batch_size={final_batch_size}, shuffle=True, drop_last={drop_last_flag}")


    
    dynamic_train_loader = DataLoader(state['train_subsets'][cid], batch_size=batch_size, shuffle=True, drop_last=True)
    #print(f"  - Created DataLoader for Client {cid} with batch_size={batch_size}")
    
    state['train_loader'] = dynamic_train_loader
    return state

def train_node(state: HPOState) -> HPOState:
    print(f">>> Graph Node: TRAIN for Client {state['client_id']}")
    
    # --- THIS IS THE FIX ---
    # Create a new dictionary containing ALL arguments for the training function
    # by merging the base arguments with the newly created dynamic train_loader.
    training_args_with_loader = {
        **state['training_args'],
        'train_loader': state['train_loader']
    }
    
    # Unpack the complete set of arguments for the function call
    w_c, w_s, sz, results = train_single_client(
        **training_args_with_loader, 
        hps=state['hps'], 
        cid=state['client_id']
    )
    
    state['client_weights'] = w_c
    state['server_weights'] = w_s
    state['data_size'] = sz
    state['results'] = results
    return state

def analyze_node(state: HPOState) -> HPOState:
    print(f">>> Graph Node: ANALYZE for Client {state['client_id']}")
    new_search_space, reasoning = analyzer_agent.analyze(
        client_id=state['client_id'],
        cluster_id=state['cluster_id'],
        model_name=state['model_name'],
        dataset_name=state['dataset_name'],
        results=state['results'],
        current_hps=state['hps'],
        search_space=state['search_space'],
        global_epoch=state['training_args']['global_epoch'],
        local_epochs=state['hps'].get('client',{}).get('local_epochs', 1)
    )
    state['search_space'] = new_search_space
    state['last_analysis'] = reasoning
    return state

# --- UPDATED GRAPH DEFINITION ---
def create_graph():
    workflow = StateGraph(HPOState)
    workflow.add_node("suggest", suggest_node)
    workflow.add_node("prepare_loader", prepare_loader_node) # Add the new node
    workflow.add_node("train", train_node)
    workflow.add_node("analyze", analyze_node)

    workflow.set_entry_point("suggest")
    
    # Define the new, correct flow of execution
    workflow.add_edge("suggest", "prepare_loader")
    workflow.add_edge("prepare_loader", "train")
    workflow.add_edge("train", "analyze")
    workflow.add_edge("analyze", END)
    
    return workflow.compile()
