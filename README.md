# InfoShield

**Misinformation detection and cascade suppression for social media.**

InfoShield is a research system that combines a graph neural network classifier, a probabilistic network model, and a linear-program optimizer to detect false information and suppress its spread in real propagation cascades — while preserving the spread of verified content.

---

## How It Works

The system has three layered components:

### 1. BiGCN Classifier (`gnn/`)
A Bi-directional Graph Convolutional Network classifies each cascade (propagation tree + source tweet text) as **true**, **false**, **unverified**, or **non-rumor**.

- Input: 772-dim per-node features — 768-dim RoBERTa text embeddings + 4 structural features
- Architecture: two parallel GCN branches (top-down + bottom-up), each 2 layers deep, with root-feature concatenation before layer 2 to push source context to deep nodes
- Output: 4-class label + confidence score
- Training: 5-fold cross-validation; only predictions with confidence ≥ 0.65 are passed downstream

### 2. Stochastic Block Model (`graph_engine/network_model.py`)
The SBM models user polarization structure. After BiGCN classifies cascades, Louvain community detection identifies *k* = 13 user classes from the WICO follower graph. Two *k × k* matrices are fitted:

- **b⁺** — edge probabilities in true-content cascades
- **b⁻** — edge probabilities in false-content cascades

Separate matrices are fitted per conspiracy category (segmented SBM) to exploit community-specific propagation differences.

### 3. LP Intervention Optimizer (`graph_engine/optimizer.py`)
At each BFS step of a live cascade, a Linear Program computes optimal per–class-pair dropout probabilities **d\***:

- **Primary LP**: minimise false-content spread subject to the true-content branching ratio staying ≥ α (default 1.5)
- **Softened LP**: fallback when the primary is infeasible — trades off false suppression against true preservation

The result is applied probabilistically to cascade edges, simulating a platform intervention that selectively dampens sharing between user communities.

---

## Datasets

| Dataset | Platform | Language | Classes | Cascades |
|---------|----------|----------|---------|---------|
| Twitter-15 | Twitter/X | English | 4 (true / false / unverified / non-rumor) | 1,490 |
| Twitter-16 | Twitter/X | English | 4 | ~1,200 |
| WICO | Twitter/X | English | 3 (5G conspiracy / other conspiracy / non-conspiracy) | ~4,000 |
| Weibo | Sina Weibo | Chinese | 2 (rumor / non-rumor) | 4,659 |

Raw data lives under `data/raw/` and is never modified. Processed PyTorch Geometric graph objects and fitted SBM matrices are written to `data/processed/`.

---

## Project Structure

```
infoshield/
├── config.py                   # Central config: paths, hyperparams, dataset settings
├── requirements.txt
│
├── data/
│   ├── raw/                    # Original datasets (read-only)
│   │   ├── wico-text/
│   │   ├── wico-graph/
│   │   ├── twitter15/
│   │   ├── twitter16/
│   │   └── weibo/
│   └── processed/              # PyG graphs, SBM matrices
│
├── gnn/                        # BiGCN model
│   ├── bigcn.py                # Model definition
│   ├── dataset.py              # Data loaders (Twitter15/16, WICO, Weibo)
│   ├── weibo_dataset.py        # Chinese-text handling
│   ├── train.py                # k-fold training harness
│   ├── predict.py              # Inference wrapper
│   └── evaluate.py             # Metrics
│
├── graph_engine/               # Network model + optimizer
│   ├── network_model.py        # SBM fitting
│   ├── optimizer.py            # LP solver (Algorithm 2)
│   ├── sir_simulation.py       # SIR baseline simulator
│   └── test_optimizer.py
│
├── pipeline/                   # End-to-end pipeline
│   ├── run_pipeline.py         # Main driver + Table II reproduction
│   ├── sbm_fitter.py           # Fit global SBM on WICO
│   ├── segmented_sbm_fitter.py # Fit per-category SBMs
│   └── counter_narrative/      # RAG + Claude rebuttal generation
│       ├── generator.py
│       ├── rag_retriever.py
│       └── formatter.py
│
├── api/                        # Flask REST API
│   ├── server.py
│   └── routes/
│       ├── classify.py         # POST /api/v1/classify
│       ├── simulate.py         # POST /api/v1/simulate
│       ├── rebuttal.py         # POST /api/v1/rebuttal
│       └── monitoring.py       # Telegram monitoring thread
│
├── evaluation/                 # Results, metrics, report prompts
├── checkpoints/                # Saved model weights
├── dashboard/                  # Frontend
└── tests/
    └── test_api.py
```

---

## Installation

```bash
# Python 3.10+ recommended
pip install -r requirements.txt
```

PyTorch Geometric requires matching CUDA/CPU wheels. See the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) if the default install fails.

---

## Running the Pipeline

### Step 1 — Train BiGCN

```bash
python -m gnn.train --dataset twitter15 --folds 5
```

Checkpoints are saved to `checkpoints/twitter15/`.

### Step 2 — Fit the SBM

```bash
# Global SBM across all WICO cascades
python -m pipeline.sbm_fitter

# Per-category SBMs (recommended)
python -m pipeline.segmented_sbm_fitter
```

Matrices are written to `data/processed/sbm_matrices/`.

### Step 3 — Run the Intervention Pipeline

```bash
python -m pipeline.run_pipeline --n-samples 500 --label-source wico_folders
```

This simulates 4,000 cascade interventions (500 samples × 4 (α, λ) configurations × true/false labels) and outputs a results CSV and Table II reproduction.

### API Server

```bash
python -m api.server
```

The server starts on port 5000. Components load from checkpoints and processed matrices; missing components fall back to mocks automatically.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/classify` | Classify a cascade: `{text, edges?}` → label + confidence |
| POST | `/api/v1/simulate` | Run LP intervention and return step-by-step d\* matrices |
| POST | `/api/v1/rebuttal` | Generate a counter-narrative for a false claim |

---

## Configuration

All paths, hyperparameters, and dataset settings are in [config.py](config.py). Key values:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BiGCNConfig.hidden_dim` | 256 | GCN hidden layer width |
| `BiGCNConfig.dropout` | 0.5 | GCN dropout rate |
| `SBMConfig.clustering_resolution` | 0.5 | Louvain resolution (higher = finer communities) |
| `SBMConfig.confidence_threshold` | 0.65 | Minimum BiGCN confidence to include in SBM fit |
| `LPOptimizerConfig.alpha` | 1.5 | Minimum true-content branching ratio |
| `LPOptimizerConfig.lambda_` | 1.0 | False-content suppression weight |
| Global `seed` | 42 | Reproducibility seed |

---

## Counter-Narrative Generation

The `pipeline/counter_narrative/` module generates grounded rebuttals using a RAG pipeline over official sources and the Claude API. Supported output languages: English, Russian, Azerbaijani.

Requires an `ANTHROPIC_API_KEY` environment variable.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Evaluation

Cross-community cascade statistics and model metrics are in `evaluation/`. To reproduce the cascade comparison (Twitter-15 vs Weibo):

```bash
python evaluation/cascade_metrics.py
```

AUC computation for Twitter-15:

```bash
python evaluation/compute_auc_twitter15.py
```

---

## Status

| Component | Status |
|-----------|--------|
| Cross-community cascade analysis | Complete |
| BiGCN architecture + training loop | Complete |
| BiGCN 5-fold CV (Twitter-15) | 2/5 folds trained |
| SBM fitting code | Complete |
| LP optimizer (Algorithm 2) | Complete |
| End-to-end pipeline evaluation | In progress |
| Flask API + mock fallbacks | Complete |
| Counter-narrative generation | Complete |
| Dashboard | In progress |

---

## License

This project is for academic research purposes.
