# 📖 PrivCL – Privacy-Enhanced Federated Contrastive Learning

## 🔎 Problem Statement
Federated Learning (FL) enables collaborative training without sharing raw data. However:
- **Privacy risks** remain due to gradient leakage and inference attacks.  
- **Non-IID data** across clients degrades representation quality.  
- Sensitive applications (e.g., healthcare, finance, IoT) require **formal privacy guarantees**.  
- Real-world datasets are **multimodal** (images, graphs, time-series), demanding advanced representation learning.  

**PrivCL** addresses these challenges by integrating:
- **Contrastive Learning (CL)** for robust representations.  
- **Graph-augmented and temporal-aware encoders** for multimodal signals.  
- **Adaptive Differential Privacy (DP)** to dynamically tune noise per client.  
- **Homomorphic Encryption (HE)** for secure aggregation at the server.  

Research Paper (Unpublished Draft)
This project was accompanied by a research study titled "PrivCL: Privacy-Enhanced Federated Contrastive Learning with Adaptive Differential Privacy and Homomorphic Encryption."

You can read the full draft here:
👉 PrivCL_Research_Draft.pdf

Note: This paper is an unpublished academic draft submitted as coursework but demonstrates the full methodology, experiments, and theoretical background behind the project.

This notebook implements a research prototype of PrivCL, starting with federated CL on MNIST, then extending to graph, time-series, DP, and HE.


This notebook implements a **research prototype** of PrivCL, starting with federated CL on MNIST, then extending to **graph, time-series, DP, and HE**.  

---

## ⚙️ Key Features
- **Federated Contrastive Learning**
  - SimCLR-style NT-Xent loss.  
  - Non-IID partitioning via Dirichlet distribution.  
- **Encoders**
  - CNN (`SmallConvEncoder`, `ImageNet`) for images.  
  - GNN (`GraphEncoder`, `GraphNet`) for graph data.  
  - RNN (`TemporalEncoder`, `TimeNet`) for time-series.  
  - `FusionProjection`, `MultimodalEncoder` for multimodal learning.  
- **Clients**
  - `PrivCLClient` – baseline FL+CL.  
  - `MultimodalPrivCLClient` – multimodal learning.  
  - `AdaptiveDPPrivCLClient` – DP-enabled FL.  
  - `HEClient` – homomorphic encryption enabled FL.  
- **Server**
  - Standard FedAvg via [Flower](https://flower.dev).  
  - `SecureHEFedAvg` – secure aggregation with HE.  
- **Synthetic Data Utilities**
  - `synth_graph` → random graphs.  
  - `synth_timeseries` → synthetic temporal signals.  
  - `MultimodalDataset` → combined graph+time-series.  

---

## 📦 Installation
```bash
# Core libraries
pip install torch torchvision flwr numpy tqdm

# Graph support
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Time-series
pip install tslearn

# Privacy & Encryption
pip install opacus tenseal

FED_CL.ipynb
│
├── Data Utilities
│   ├── TwoCropTransform, ContrastiveMNIST
│   ├── partition_dirichlet (non-IID partitioning)
│   ├── synth_graph, synth_timeseries, MultimodalDataset
│
├── Models
│   ├── SmallConvEncoder, ImageNet (images)
│   ├── GraphEncoder, GraphNet (graphs)
│   ├── TemporalEncoder, TimeNet (time-series)
│   ├── ProjectionHead, FusionProjection, MultimodalEncoder
│
├── Loss
│   ├── NTXentLoss, nt_xent_loss (contrastive loss)
│
├── Federated Clients
│   ├── PrivCLClient (baseline)
│   ├── MultimodalPrivCLClient
│   ├── AdaptiveDPPrivCLClient
│   ├── HEClient
│
├── Federated Server
│   ├── SecureHEFedAvg (HE aggregation)
│   ├── client_fn, main_simulation, start_simulation
```
```bash
from FED_CL import main_simulation
main_simulation(num_clients=5, rounds=10, local_epochs=1, batch_size=128, alpha=0.5)

from FED_CL import start_simulation
start_simulation(num_clients=4, rounds=5, local_steps=20)

client = AdaptiveDPPrivCLClient(...)

strategy = SecureHEFedAvg()
```
📊 Evaluation

Representation Quality: linear probing, clustering.

Privacy: (ε, δ)-DP budgets, resistance to inference attacks.

Utility: accuracy on downstream tasks.

Overhead: communication/computation cost.

🔮 Extensions

Replace MNIST with Cora (graphs) or UCR archive (time-series).

Scale clients (5 → 100+) to test federated robustness.

Add attack simulations (membership inference, backdoors).

📚 References

Flower
 – Federated learning framework.

Opacus
 – Differential Privacy in PyTorch.

TenSEAL
 – Homomorphic Encryption.

PyTorch Geometric
 – Graph learning.

tslearn
 – Time-series learning.

 
