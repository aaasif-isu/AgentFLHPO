# AgentFLHPO
# Agent-Based Per-Client Hyperparameter Optimization in Federated Learning

This project explores a novel framework for performing dynamic, per-client hyperparameter optimization (HPO) in a Federated Learning (FL) environment. It moves beyond the standard FL paradigm of using a single, fixed set of hyperparameters for all clients, which is suboptimal in heterogeneous settings where data and resources vary significantly across devices.

The core goal is to leverage the reasoning capabilities of Large Language Models (LLMs) to create an intelligent agent-based system that tunes hyperparameters for each client individually, adapting to their specific data distributions and performance over time.

## Our Approach: An LLM-Powered Agent System

The framework is built around a dual-agent system that operates within a "tuning-while-training" paradigm. This allows for efficient HPO without the need for costly "training-after-tuning" cycles. This approach is inspired by recent advancements in the field, such as those described in the FedPop and FedEx papers.

The system consists of:

* **The HP Agent (Suggester):** For each client in a given round, this agent receives the client's full context, including its cluster ID, performance history, and its personalized search space. It prompts an LLM to suggest the optimal set of hyperparameters for the upcoming training round.

* **The Analyzer Agent:** After a client completes its local training, this agent analyzes the results (e.g., training vs. test accuracy, signs of overfitting). It then prompts an LLM to provide structured, step-by-step reasoning and propose a *new, refined search space* for that specific client, which will be used in all future rounds. This creates a persistent, adaptive learning loop for each individual client.

* **`FedProx` for Stability:** To solve the critical challenge of model divergence that occurs when clients train with different hyperparameters, the framework uses the `FedProx` algorithm. By adding a proximal term to the client's loss function, it ensures the stability of the global model while still allowing the freedom of per-client optimization.

* **Detailed Reporting:** At the end of a training run, the system generates a detailed `client_hpo_states.yaml` file. This file contains a clear, epoch-by-epoch report for every client, logging the hyperparameters suggested by the agent, the resulting test accuracy, and the final refined search space, providing full transparency into the HPO process.

## Core Framework: The `ssfl` Directory

The core logic for the underlying Federated Learning system resides in the `code/ssfl/` directory. This module handles the fundamental mechanics of the FL process.

* **`model_splitter.py`**: Contains the logic for splitting a standard neural network (like ResNet-18) into client-side and server-side components.
* **`aggregation.py`**: Implements the `FedAvg` algorithm and the functions for combining the split models back into a whole for evaluation.
* **`trainer.py` and `trainer_utils.py`**: Orchestrate the main training loop, including client selection, cluster-based training rounds, aggregation, and evaluation. This is the engine that the HPO strategies plug into.

## Key Features

* **Per-Client HPO:** Moves beyond a single global hyperparameter set to tune each client individually.
* **LLM-Powered Agents:** Uses the advanced reasoning of LLMs to suggest hyperparameters and refine search spaces based on a rich set of contextual information.
* **Adaptive Search Space:** The hyperparameter search space for each client evolves over time based on its unique performance and characteristics.
* **Robust Training:** Implements the `FedProx` algorithm to ensure stable global model convergence in a heterogeneous environment.
* **Modular Experimentation:** Built with a "Strategy" pattern (`strategies.py`) that allows for easy comparison between the agent-based approach and other SOTA baselines like Random Search, SHA, or `FedAvg`/`FedProx` with fixed HPs.