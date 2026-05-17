# InfoGuard: Network-Level Misinformation Suppression

> Suppress misinformation cascades without reading a single post.

**Authors:** Azerin Salahova · Dmitriy Kuramshin
**Institution:** UFAZ — French-Azerbaijani University
**Project type:** Student-led research project

---

InfoGuard is a deployable misinformation suppression system that operates entirely at the network level. Instead of classifying posts by content, it models how misinformation and verified content propagate through social community structures and applies a per-step linear-program dropout that throttles misinformation pathways while preserving verified content reach.

The system extends the network-level suppression framework of [Bayiz & Topcu (2022)](https://arxiv.org/abs/2211.04617) with four original contributions: a confidence-gated BiGCN classifier as input filter to the SBM fitter, an argued Louvain polarisation-discovery configuration, an emergency-deployment framing with a specified bias-audit module, and a prototype retrieval-augmented counter-narrative generator.

**Core result on WICO Graph:** at (α, λ) = (1.5, 1.0), mean misinformation cascade size falls from 59.8 to 38.5 (−36%) while verified content falls from 56.1 to 42.8 (−24%), reversing the cascade-size ordering — without reading any post content.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Running the Pipeline](#running-the-pipeline)
- [Training BiGCN](#training-bigcn)
- [API Server and Dashboard](#api-server-and-dashboard)
- [Counter-Narrative Generator](#counter-narrative-generator)
- [Configuration](#configuration)
- [Implementation Status](#implementation-status)
- [Limitations](#limitations)
- [Acknowledgements](#acknowledgements)
- [References](#references)
- [Citation](#citation)
- [License](#license)

---

## How It Works

InfoGuard runs in two modes.

**Offline — SBM fitting:**
Labelled cascade data → Louvain community detection (k = 4 classes, resolution 0.5, merging floor 5%) → b⁺ and b⁻ matrix estimation by frequentist transfer counting → saved to disk.

**Online — cascade suppression:**
Live cascade → BiGCN classification (confidence-gated at 0.65) → LP at every BFS step → d\* dropout matrix → Bernoulli mask on forward edges → R∞ recorded.

```
raw cascades ──────────────────────────► SBM fitter ──► b⁺, b⁻
                                              ▲
                                    (high-confidence labels only)
                                         BiGCN predictor

                         b⁺, b⁻, S_t, I_t
                              │
                              ▼
                         LP optimiser
                              │
                             d* per step
                              │
                              ▼
                    cascade-following simulator ──► R∞, Pr[R∞ < 5]

                    counter-narrative generator ──► rebuttal (az / ru / en)
```

BiGCN labels enter only the SBM fitter. The LP, the dropout matrix, and the simulator never see classifier output. A wrong BiGCN prediction can shift the fitted matrices over time but cannot block any single post.

The suppression signal concentrates on a single class pair (C3 ↔ C4) where b⁻ entries are 1.00 and 0.93 while the corresponding b⁺ entries sit at the Laplace floor (10⁻⁸). The LP responds by setting d\*[C3, C4] and d\*[C4, C3] low, leaving diagonal transfers that carry verified content largely intact.

---

## Key Results

Results on WICO Graph (500 cascades per cell, sampled with replacement, restricted to cascades with ≥ 20 downstream nodes):

| (α, λ) | E[R∞] verified | E[R∞] misinfo. | Pr[R∞ < 5] verified | Pr[R∞ < 5] misinfo. |
|--------|---------------|----------------|---------------------|---------------------|
| Control | 56.1 | 59.8 | 0.00 | 0.00 |
| (1.5, 1.0) | **42.8** | **38.5** | 0.05 | **0.14** |
| (2.0, 1.5) | 45.0 | 41.8 | 0.04 | 0.10 |
| (3.0, 2.0) | 48.8 | 46.2 | 0.02 | 0.08 |

The cascade-size ordering reverses at every measured α setting. At (1.5, 1.0): misinformation −36%, verified −24%. Early-collapse ratio 2.8× against misinformation.

BiGCN on Twitter15: binary accuracy 0.84, macro AUC 0.8015, 5-fold cross-validation.

---

## Project Structure

```
infoshield/
├── config.py                        # Central config — all paths and hyperparameters
│
├── data/
│   ├── raw/
│   │   ├── twitter15/               # Propagation trees + label.txt
│   │   ├── twitter16/               # Propagation trees + label.txt
│   │   ├── wico-graph/              # 3 class folders with edges.txt + nodes.csv
│   │   │   ├── 5G_Conspiracy_Graphs/
│   │   │   ├── Other_Graphs/
│   │   │   └── Non_Conspiracy_Graphs/
│   │   └── weibo/                   # weibotree.txt + weibo_id_label.txt
│   └── processed/
│       └── sbm_matrices/            # b_plus.npy, b_minus.npy, class_sizes.npy
│
├── graph_engine/
│   ├── network_model.py             # SBM class and SBMFitter
│   ├── sir_simulation.py            # SIR cascade simulator
│   └── optimizer.py                 # DropoutOptimizer — Algorithm 2 (LP)
│
├── gnn/
│   ├── dataset.py                   # TwitterRumourDataset, PyG graph builder
│   ├── bigcn.py                     # BiGCN model (in_dim = 772)
│   ├── train.py                     # 5-fold stratified CV training loop
│   ├── evaluate.py                  # 4-class and binary metrics, ROC/AUC
│   └── predict.py                   # Predictor inference class
│
├── pipeline/
│   ├── sbm_fitter.py                # Loads WICO, fits SBM matrices
│   ├── run_pipeline.py              # End-to-end cascade-following simulation
│   ├── segmented_sbm_fitter.py      # Per-community-segment SBM fitting
│   ├── ingestors/
│   │   ├── base_ingestor.py         # Abstract base + ValidationReport
│   │   └── instagram_ingestor.py    # Instagram cascade builder (mock + real API)
│   └── counter_narrative/
│       ├── generator.py             # RAG-grounded LLM rebuttal generator
│       ├── rag_retriever.py         # TF-IDF retrieval over source documents
│       ├── formatter.py             # Platform-specific output formatting
│       └── sources/                 # Verified plain-text source documents
│
├── api/
│   ├── server.py                    # Flask REST API server
│   └── routes/
│       ├── classify.py              # POST /api/v1/classify
│       ├── simulate.py              # POST /api/v1/simulate
│       ├── rebuttal.py              # POST /api/v1/rebuttal
│       └── monitoring.py            # GET  /api/v1/live_cascades
│
├── dashboard/
│   ├── index.html                   # Single-page operator dashboard (D3.js)
│   └── demo_data.json               # Pre-computed demo data for offline use
│
└── evaluation/
    ├── explore_data.ipynb           # Phase 1: dataset inspection and loading
    ├── visualize_trees.ipynb        # Phase 1: cascade structure visualisation
    ├── weibo_exploration.ipynb      # Cross-community structural analysis
    ├── results.ipynb                # Table II reproduction and figures
    └── tree_figures/                # Generated PNG figures
```

---

## Installation

Python 3.11 required.

```bash
git clone https://github.com/Krmsh1n5/infoshield.git
cd infoshield
pip install -r requirements.txt
```

PyTorch Geometric requires matching CUDA/CPU wheels. See the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) if the default install fails.

Core dependencies:

```
torch>=2.0
torch-geometric
networkx
scipy
scikit-learn
numpy
pandas
matplotlib
flask
flask-cors
anthropic
python-louvain
transformers
```

---

## Datasets

| Dataset | Platform | Language | Cascades | Used for |
|---------|----------|----------|----------|----------|
| Twitter15 | Twitter/X | English | 1,490 | BiGCN training |
| Twitter16 | Twitter/X | English | 818 | BiGCN training |
| WICO Graph | Twitter/X | English | 2,502 | SBM fitting + evaluation |
| Weibo | Sina Weibo | Chinese | 4,659 | Cross-community analysis |

### Twitter15 and Twitter16

Download from the [BiGCN repository](https://github.com/TianBian95/BiGCN) and place at:

```
data/raw/twitter15/
    label.txt
    source_tweets.txt
    tree/              ← one .txt file per cascade
data/raw/twitter16/
    (same structure)
```

### WICO Graph

Download from [Simula Research Laboratory](https://datasets.simula.no/wico-graph/) and place at:

```
data/raw/wico-graph/
    5G_Conspiracy_Graphs/     ← false content (conspiracy)
        1/
            edges.txt
            nodes.csv
        2/ ...
    Other_Graphs/             ← false content (other conspiracy)
    Non_Conspiracy_Graphs/    ← true content
```

**Edge direction note:** WICO `edges.txt` encodes "retweeter → original poster". Content flows against the stored edge direction. The pipeline corrects this automatically at load time with `G.reverse()`.

### Weibo

Download from the [BiGCN repository](https://github.com/TianBian95/BiGCN) and place at:

```
data/raw/weibo/
    weibotree.txt
    weibo_id_label.txt
```

---

## Running the Pipeline

### Step 1 — Fit the SBM

```bash
# Using WICO folder labels (recommended — no BiGCN checkpoint required)
python -m pipeline.sbm_fitter --label-source wico_folders

# Using BiGCN predictions (requires trained checkpoint at fold 0)
python -m pipeline.sbm_fitter --label-source bigcn --fold 0 --force-refit
```

Verify the fitted matrices:

```bash
python3 -c "
import numpy as np
b_plus  = np.load('data/processed/sbm_matrices/b_plus.npy')
b_minus = np.load('data/processed/sbm_matrices/b_minus.npy')
sizes   = np.load('data/processed/sbm_matrices/class_sizes.npy')
k = b_plus.shape[0]
mask = np.eye(k, dtype=bool)
print(f'k={k}, sizes={sizes.tolist()}')
print(f'b+ range: [{b_plus.min():.2e}, {b_plus.max():.2e}]')
print(f'b- off/diag ratio: {b_minus[~mask].mean()/b_minus[mask].mean():.0f}x')
"
```

Expected: `k=4`, class sizes `[38113, 54956, 8484, 12682]`, b⁻ off/diagonal ratio in the hundreds.

### Step 2 — Run the evaluation

```bash
# Full run — 500 cascades per (α, λ) setting per label
python -m pipeline.run_pipeline --n-samples 500 --skip-sbm-fit

# Quick test — 50 cascades
python -m pipeline.run_pipeline --n-samples 50 --skip-sbm-fit
```

Results are saved to `evaluation/pipeline_results_raw.csv` and `evaluation/pipeline_table2.csv`.

### Step 3 — Reproduce Table II

Open `evaluation/results.ipynb` and run all cells.

---

## Training BiGCN

```bash
# Train on Twitter15, all 5 folds
python -m gnn.train --split twitter15 --folds 5

# Train fold 0 only (quick iteration)
python -m gnn.train --split twitter15 --folds 5 --only-fold 0

# Evaluate fold 0
python -m gnn.evaluate --split twitter15 --fold 0
```

Checkpoints are saved to `data/processed/checkpoints/twitter15/fold{N}_best.pt`.

**BiGCN architecture:**
- Input: 772-dim per-node features — 768-dim RoBERTa-base CLS embedding broadcast to all nodes + 4 structural features (BFS depth, in-degree, out-degree, root indicator)
- Two parallel GCNConv branches (top-down + bottom-up), 2 layers each, hidden dim 256, output dim 128, root-feature enhancement before layer 2
- Graph embedding: 256-dim (mean-pool per branch, concatenate)
- Classifier head: 256 → 128 → 4
- Training: Adam (lr=5×10⁻⁴, weight decay=5×10⁻⁴), label smoothing=0.05, gradient clip=5.0, dropout=0.5, early stopping on val macro-F1 (patience 20)

**Performance on Twitter15 (5-fold CV):**

| Fold | 4-class Acc | 4-class F1 | Binary Acc | Binary F1 |
|------|-------------|------------|------------|-----------|
| 0 | 0.66 | 0.66 | 0.84 | 0.76 |
| 1 | 0.60 | 0.67 | 0.78 | 0.57 |
| 2 | 0.65 | 0.65 | 0.79 | 0.67 |
| 3 | 0.56 | 0.55 | 0.73 | 0.49 |
| 4 | 0.68 | 0.68 | 0.85 | 0.76 |
| **Mean** | **0.63** | **0.64** | **0.80** | **0.65** |

Macro-AUC: 0.8015 (true: 0.837, false: 0.788, unverified: 0.774, non-rumour: 0.808).

---

## API Server and Dashboard

### Start the server

```bash
# Mock mode — no credentials required, uses replayed WICO data
python -m api.server --mock

# Live mode — requires ANTHROPIC_API_KEY
python -m api.server

# Custom port
python -m api.server --mock --port 8080
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Component status and mock mode flag |
| POST | `/api/v1/classify` | BiGCN inference on cascade graph or text |
| POST | `/api/v1/simulate` | Cascade-following simulation at specified (α, λ) |
| POST | `/api/v1/rebuttal` | Generate counter-narrative rebuttal |
| GET | `/api/v1/live_cascades` | Active monitored cascades |
| GET | `/api/v1/sbm/matrices` | Fitted b⁺ and b⁻ matrices |

### Dashboard

Navigate to `http://localhost:5000` with the server running. Four views:

- **Live Feed** — active Instagram cascades with BiGCN confidence and cascade size
- **Cascade Graph** — side-by-side false/true tree visualisations with depth profiles
- **Heatmap** — 4×4 d\* dropout matrix, control vs Algorithm 2
- **Rebuttal** — detected claim, generated rebuttal, platform-formatted outputs

For offline use, open `dashboard/index.html` directly — it loads from `dashboard/demo_data.json`.

---

## Counter-Narrative Generator

Requires `ANTHROPIC_API_KEY` set as an environment variable.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

```python
from pipeline.counter_narrative.generator import CounterNarrativeGenerator

gen = CounterNarrativeGenerator()
result = gen.generate_rebuttal(
    false_claim="5G towers in Baku are causing health problems",
    topic="health",
    language="az",          # "az" | "ru" | "en"
    confidence=0.91,
    cascade_pattern="wide_burst",
    top_k_sources=3,
)
print(result.text)
print(result.sources_used)
print(result.generation_time_ms)
```

The generator retrieves the top-3 most relevant passages from the local source index via TF-IDF cosine similarity and passes them as grounding context to `claude-sonnet-4-6`. The model is instructed to cite only retrieved passages and to produce a maximum three-sentence rebuttal. If retrieved context is insufficient, it returns a language-appropriate under-investigation message rather than generating unsupported claims.

---

## Configuration

All paths and hyperparameters are in `config.py`. Key values:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bigcn.text_embed_dim` | 772 | Node feature dimension (768 + 4 structural) |
| `bigcn.gcn_hidden_dim` | 256 | GCN hidden layer width |
| `bigcn.gcn_output_dim` | 128 | GCN output dimension per branch |
| `bigcn.dropout` | 0.5 | Dropout rate in branches and classifier head |
| `bigcn.learning_rate` | 5×10⁻⁴ | Adam learning rate |
| `bigcn.weight_decay` | 5×10⁻⁴ | Adam weight decay |
| `bigcn.label_smoothing` | 0.05 | Cross-entropy label smoothing |
| `bigcn.num_epochs` | 200 | Maximum training epochs per fold |
| `bigcn.patience` | 20 | Early stopping patience (val macro-F1) |
| `sbm.clustering_resolution` | 0.5 | Louvain resolution |
| `sbm.min_class_fraction` | 0.05 | Merging floor |
| `sbm.label_confidence_threshold` | 0.65 | Minimum BiGCN confidence for SBM fitter |
| `lp.alpha` | 1.5 | Minimum verified content branching ratio |
| `lp.lambda_weight` | 1.0 | Soft-LP penalty weight |

---

## Implementation Status

| Component | Status |
|-----------|--------|
| Cross-community cascade analysis (Twitter15, Weibo, WICO) | ✅ Complete |
| BiGCN architecture | ✅ Complete |
| BiGCN 5-fold CV on Twitter15 | ✅ Complete |
| SBM fitting with Louvain polarisation discovery | ✅ Complete |
| LP optimiser — Algorithm 2 (primary + soft) | ✅ Complete |
| Cascade-following simulation + Table II reproduction | ✅ Complete |
| Flask REST API with mock fallback | ✅ Complete |
| Counter-narrative RAG prototype | ✅ Complete |
| Instagram cascade ingestor (mock + real API) | ✅ Complete |
| Operator dashboard (4 views) | ✅ Complete |
| **Bias-audit observer** | ❌ Specified, not implemented |
| Telegram live monitoring | 🔄 Designed, not connected |
| WhatsApp forward-to-bot | 🔄 Designed, not connected |
| Azerbaijani-language cascade dataset | ❌ Does not exist |

> **Deployment note:** The system is not ready for production deployment until the bias-audit observer is implemented. Without it, the 15% per-category dropout divergence threshold — which prevents the suppression mechanism from targeting political opposition — exists on paper only.

---

## Limitations

**Bias audit not implemented.** The component that decides whether the system is acting against misinformation or against political opposition does not exist in code. Until it is implemented and validated, the system cannot be deployed.

**Single resolved cross-class pathway.** The discrimination signal on WICO concentrates in one class pair (C3 ↔ C4). A network whose misinformation spreads through multiple overlapping pathways would offer the LP much weaker discrimination. Whether the WICO pattern generalises to other emergency-context networks is an open empirical question.

**Classifier drift.** BiGCN binary accuracy is 0.84. The remaining 16% of high-confidence predictions are wrong and enter the SBM fitter with the wrong label. Over many refit cycles, drift can pull b⁺ and b⁻ away from the true asymmetry the LP relies on. The current pipeline has no drift detector.

**Single-language evaluation.** All training and evaluation use English-language data. The Azerbaijani deployment scenario is illustrative. The text encoder, the community partition, and the fitted matrices all need to be rebuilt on Azerbaijani cascade data before any operational claim can be made.

**Coarse polarisation classes.** The pipeline reduces 114,235 users to k = 4 classes. Real polarisation is continuous and shifts over time. Periodic refitting would help; refitting without restarting the running LP is an open implementation question.

---

## Acknowledgements

The mathematical framework for network-level misinformation suppression is taken from:

> Bayiz, Y. E., & Topcu, U. (2022). *Countering misinformation on social networks using graph alterations* (arXiv:2211.04617). arXiv.

The BiGCN architecture is based on:

> Bian, T., Xiao, X., Xu, T., Zhao, P., Huang, W., Rong, Y., & Huang, J. (2020). Rumor detection on social media with bi-directional graph convolutional networks. *Proceedings of the AAAI Conference on Artificial Intelligence*, 34(1), 549–556.

Community detection uses Louvain modularity clustering:

> Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. *Journal of Statistical Mechanics*, 2008(10), P10008.

The WICO Graph dataset is provided by:

> Pogorelov, K., Schroeder, D. T., Filkuková, P., Brenner, S., & Langguth, J. (2021). WICO Text: A labeled dataset of conspiracy theory and 5G-corona misinformation tweets. *Proceedings of the 2021 Workshop on Open Challenges in Online Social Networks*, 21–25.

Text embeddings use RoBERTa-base:

> Liu, Y., Ott, M., Goyal, N., et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach (arXiv:1907.11692). arXiv.

The retrieval-augmented generation approach follows:

> Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, 33, 9459–9474.

The LP is solved using SciPy with the HiGHS backend:

> Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, 17(3), 261–272.

> Huangfu, Q., & Hall, J. A. J. (2018). Parallelizing the dual revised simplex method. *Mathematical Programming Computation*, 10(1), 119–142.

Graph manipulation uses NetworkX:

> Hagberg, A. A., Schult, D. A., & Swart, P. J. (2008). Exploring network structure, dynamics, and function using NetworkX. *Proceedings of the 7th Python in Science Conference*, 11–15.

PyTorch Geometric is used for GNN implementation:

> Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. *ICLR Workshop on Representation Learning on Graphs and Manifolds*.

Counter-narrative generation uses the Anthropic Messages API (claude-sonnet-4-6). GCN layers follow Kipf & Welling (2017). Optimisation uses Adam (Kingma & Ba, 2015). The SIR propagation model follows Newman (2002). Empirical motivation draws on Vosoughi et al. (2018) and Del Vicario et al. (2016).

---

## References

Full reference list as cited in the accompanying research report:

1. Bayiz, Y. E., & Topcu, U. (2022). Countering misinformation on social networks using graph alterations (arXiv:2211.04617). arXiv.
2. Bian, T., Xiao, X., Xu, T., Zhao, P., Huang, W., Rong, Y., & Huang, J. (2020). Rumor detection on social media with bi-directional graph convolutional networks. *AAAI*, 34(1), 549–556.
3. Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. *Journal of Statistical Mechanics*, P10008.
4. Del Vicario, M., et al. (2016). The spreading of misinformation online. *PNAS*, 113(3), 554–559.
5. Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. *ICLR Workshop*.
6. Hagberg, A. A., Schult, D. A., & Swart, P. J. (2008). Exploring network structure, dynamics, and function using NetworkX. *SciPy Conference*, 11–15.
7. Holland, P. W., Laskey, K. B., & Leinhardt, S. (1983). Stochastic blockmodels: First steps. *Social Networks*, 5(2), 109–137.
8. Huangfu, Q., & Hall, J. A. J. (2018). Parallelizing the dual revised simplex method. *Mathematical Programming Computation*, 10(1), 119–142.
9. Kempe, D., Kleinberg, J., & Tardos, É. (2003). Maximizing the spread of influence through a social network. *KDD*, 137–146.
10. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
11. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR*.
12. Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS*, 33, 9459–9474.
13. Liu, Y., et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach (arXiv:1907.11692). arXiv.
14. Ma, J., et al. (2016). Detecting rumors from microblogs with recurrent neural networks. *IJCAI*, 3818–3824.
15. Ma, J., Gao, W., & Wong, K.-F. (2017). Detect rumors in microblog posts using propagation structure via kernel learning. *ACL*, 708–717.
16. Newman, M. E. J. (2002). Spread of epidemic disease on networks. *Physical Review E*, 66(1), 016128.
17. Pennycook, G., & Rand, D. G. (2021). The psychology of fake news. *Trends in Cognitive Sciences*, 25(5), 388–402.
18. Pogorelov, K., et al. (2021). WICO Text: A labeled dataset of conspiracy theory and 5G-corona misinformation tweets. *OCOSNA Workshop*, 21–25.
19. Virtanen, P., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, 17(3), 261–272.
20. Vosoughi, S., Roy, D., & Aral, S. (2018). The spread of true and false news online. *Science*, 359(6380), 1146–1151.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{salahova2024infoguard,
  author    = {Salahova, Azerin and Kuramshin, Dmitriy},
  title     = {InfoGuard: Network-Level Misinformation Suppression with a BiGCN Input Filter},
  year      = {2024},
  institution = {UFAZ — French-Azerbaijani University},
  note      = {Student-led research project},
  url       = {https://github.com/Krmsh1n5/infoshield}
}
```

---

## License

This project is for academic research purposes only.

The system is designed for deployment against genuine misinformation in emergency scenarios. It must never be deployed against legitimate political speech. The bias-audit module is a mandatory component for any operational use; differential dropout rates exceeding 15% between political content categories must automatically trigger human review. Human oversight is non-negotiable.