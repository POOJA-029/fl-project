# Multi-Domain Privacy-Preserving, Fair, and Energy-Efficient Federated Learning System

**Master's in Artificial Intelligence - Final Year Project (600 Marks)**

This project addresses three critical challenges in modern AI systems: **Privacy, Fairness, and Computational Efficiency**, leveraging Federated Learning and Green AI techniques. To demonstrate versatility, this project evaluates the system across three diverse domains:

1.  **Healthcare**: Heart Disease Risk Prediction
2.  **Finance**: Credit Default / Fraud Prediction (Credit-g)
3.  **Standard Benchmark**: Adult Census Income (Standard Fairness Baseline)

---

## 🎯 Project Objectives

1.  **Privacy-Preserving Training**: Utilize Federated Learning (FL) to distribute artificial neural networks across multiple simulated edge clients (e.g., hospitals, local banks). The central server only aggregates model weights (`FedAvg`), preventing any raw data sharing.
2.  **Bias Detection & Mitigation**: Apply cutting-edge demographic parity and equal opportunity metrics to detect algorithmic bias against protected groups (e.g., Age > 55, Gender). Uses re-weighting and balanced dataloaders as mitigation techniques.
3.  **Green AI / Energy Optimization**: Compresses the trained deep learning model using structural **Pruning**. This reduces the carbon footprint, computational load, and memory overhead, preparing the model for efficient deployment on IoT edge devices.

---

## 🛠️ System Architecture

1.  **Dataset Preprocessing Module**: Dynamically fetches and cleans external OpenML datasets. Automatically identifies categorical vs. numerical features to OHE and Scale data for Neural Network ingestion.
2.  **Federated Module (`src/federated.py`)**: Simulates partitioned decentralized nodes.
3.  **Fairness Module (`src/fairness.py`)**: Tracks demographic parities and applies mitigation protocols.
4.  **Efficiency Module (`src/efficiency.py`)**: Prunes model layers by an objective sparsity ratio, measuring pre-and post-optimization inferences.

---

## 🚀 How to Run

1.  **Install Requirements**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Execute the Sandbox Interactive Presentation**
    ```bash
    streamlit run app.py
    ```

3.  **Navigate Domains**
    Use the sidebar in the Streamlit application to switch between Healthcare, Finance, and Benchmark test settings seamlessly. Evaluate the multi-tab layout traversing Data Overview -> Federated Training -> Bias Mitigation -> Model Compression.

---
*Created as part of the Master's AI Final Degree Implementation Requirements.*
