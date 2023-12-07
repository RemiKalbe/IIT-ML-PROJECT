# GNN Explanation Framework for Recommendation Systems

This repository contains our efforts to develop an explainable Graph Neural Network (GNN) framework for recommendation systems, with a focus on ablation studies for elucidating decision-making processes.

## Project Overview

We introduce a specialized `Graph` data structure that standardizes various graph representations for consistent processing. Our `LocalGNNAnalyzer`, utilizing an ablation-based approach, systematically removes edges to measure their influence on GNN predictions. We seek to identify which substructures within the graph are most influential in driving the recommendations that GNNs make.

## Repository Structure

- **datasets/**: Contains the 'cora' dataset used for experiments.
- **doc/**: Includes various literature sources and our PDF write-ups on GNN and recommender systems.
- **GNNAnalyzerExperimentations.ipynb**: A Jupyter notebook at the root of the repository documenting the experimentation made with our algorithm.
- **experiments/**: Houses experimentation of different GNN models and algorithms that explain their predictions.
- **lib/**: This directory is the heart of the project where our algorithm resides. It comprises the analytic toolset required for GNN explanation:
  - **gnnanalyzer/**: Contains the `LocalGNNAnalyzer` module and associated calculators for impact analysis.
  - **graph/**: Implements the `Graph` class, a flexible structure representing graphs for the GNN model.

## Key Components

The critical module that forms the core of our explanation mechanism is located within the `lib` directory:

- **lib/gnnanalyzer/**: Here lies our `LocalGNNAnalyzer`, which conducts the ablation studies pivotal to understanding how GNNs reach their predictions.
- **lib/graph/**: This folder contains the `Graph` class that defines our versatile and unifying graph data structure, designed to facilitate easy interaction with GNNs.

The *GNNAnalyzerExperimentations.ipynb* notebook in the root directory provides a practical view of using the LocalGNNAnalyzer within a real-world application context. The comprehensive experiments therein offer insight into the performance and interpretability improvements our algorithm introduces.

## Experiments and Results

Our experimentation phase utilizes the Cora dataset, a popular benchmark in the GNN domain, alongside our own developed GNN models.

The experiments are designed to illuminate our algorithm's capabilities and identify future lines of inquiry to complete the development path.