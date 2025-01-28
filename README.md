# Exploring LSTM and GRU Models for Sequential Data Tasks

## Overview
This repository explores the performance of LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) models on sequential data tasks. The experiments were inspired by theoretical insights and practical guidelines from relevant literature. The primary goal was to evaluate the models' ability to learn long-term dependencies and their effectiveness under various hyperparameter configurations.

## Features
- Implementation of both LSTM and GRU architectures.
- Customizable parameters for:
  - Learning rate
  - Learning rate decay
  - Number of epochs
  - Batch size
  - Sequence length
  - Dropout rates
- Comparative analysis of model performance, including validation perplexity and training stability.

## Experiments and Key Findings
1. **Baseline Configurations**:
   - Batch size: 20
   - Sequence length: 20
   - Optimizer: SGD (with fallback experiments on Adam)
   - Learning rate and decay tuned iteratively based on validation perplexity.

2. **Hyperparameter Tuning**:
   - Learning rates and decay schedules significantly influenced validation perplexity.
   - Dropout was crucial in improving generalization, with models using dropout achieving lower perplexity (~108).
   - LSTMs outperformed GRUs in capturing long-term dependencies, as reflected in perplexity scores, though GRUs trained faster.

3. **Observations**:
   - Models with higher dropout values (e.g., 0.6) performed comparably to those with lower dropout values after adjusting other parameters.
   - GRUs, while computationally efficient, showed limitations in handling long-term dependencies compared to LSTMs.
   - Dynamic learning rate decay, triggered by validation perplexity thresholds, showed promise but required further standardization across models.
