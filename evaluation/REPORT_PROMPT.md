# InfoShield — Report Generation Prompt

---

## HOW TO USE THIS PROMPT

Paste everything from **"YOUR ROLE"** onward into your LLM of choice.
The sections marked `[DATA]` are ground-truth facts — use them verbatim.
Do not invent numbers, citations, or results that are not listed here.

---

## YOUR ROLE

You are a senior machine-learning systems engineer writing a technical project
report for an academic/industry audience. You have built every part of this
system yourself, you understand every trade-off you made, and you write with
precision and confidence. You never pad sentences. You never say something is
"interesting" or "promising" — you say what the numbers show and what they
mean for the next engineering decision.

Your writing style: short declarative sentences, active voice, technical
vocabulary used correctly. You justify every architectural choice with either
an equation, a measured result, or a first-principles argument. When something
is incomplete, you say so plainly and explain what is needed to complete it.

---

## PROJECT CONTEXT

**InfoShield** is a misinformation detection and cascade-suppression system
for social media. The pipeline has four phases that flow into each other:

```
Phase 1 (Exploration)
  └─ Characterize cascade structure across communities (Twitter-15, Weibo, WICO)
  └─ Justify community-specific model parameters statistically

Phase 2 (BiGCN)
  └─ Train a Bi-directional GCN to classify each cascade as
     true / false / unverified / non-rumour (4-class)
  └─ Node features: 768-dim frozen RoBERTa CLS + 4 structural features = 772-dim

Phase 3 (SBM Fitting)
  └─ Build a Stochastic Block Model of the user network:
     b⁺ (k×k) for true content, b⁻ (k×k) for false content
  └─ k=13 polarization classes via Louvain modularity (resolution=2.0)
  └─ b_uv = (transfers observed from class u to class v) / |Cu|   (frequentist MLE)

Phase 4 (LP Optimizer + Algorithm 2)
  └─ At each SIR time step, solve an LP to find the optimal dropout matrix d*
  └─ PRIMARY LP: min ∑ |S^v| |I^u| d_uv b⁻_uv
                  s.t. ∑ |S^v| |I^u| d_uv b⁺_uv ≥ α|I|,  d ∈ [0,1]^{k×k}
  └─ SOFTENED LP (when primary infeasible): min ∑ |S^v||I^u|(b⁻_uv + λb⁺_uv)d_uv
  └─ α = 1.5 (safety margin for true content),  λ = 1.0 (balance weight)
```

The system is evaluated on **WICO** (COVID-19 conspiracy tweets, Twitter English)
as the test community. Twitter-15 (English general rumours) and Weibo (Chinese
Sina Weibo) serve as cross-community comparisons.

---

## [DATA] EMPIRICAL RESULTS — USE EXACTLY THESE NUMBERS

### Phase 1 — Cascade Structure

| Metric | Twitter-15 | Weibo |
|--------|-----------|-------|
| Mean cascade depth | 3.96 | 5.19 |
| Mean cascade width | 309.84 | 530.53 |
| Depth / width ratio | 0.023 | 0.054 |
| Gini (depth distribution) | 0.21 | 0.35 |
| Gini (width distribution) | 0.454 | 0.715 |
| Mean cascade size (nodes) | 402.15 | 790.30 |
| Dataset size | 1,490 cascades | 4,659 cascades |

Statistical significance: All three metrics (depth, width, d/w ratio) differ
significantly between Twitter-15 and Weibo under Mann-Whitney U / Kruskal-Wallis
(p < 0.001 for all). Not a sampling artefact.

Within-Weibo label comparison (rumour vs non-rumour):

| Metric | Rumour | Non-Rumour |
|--------|--------|------------|
| Mean depth | 3.91 | 6.72 |
| Mean width | 566.6 | 494.7 |
| Depth/width ratio | 0.022 | 0.090 |
| Mean size | 721.2 | 876.2 |

The d/w ratio separates the classes by a factor of 4.1x (0.090 / 0.022).
This is a strong structural discriminating feature.

Weibo label balance: 2,278 rumour (48.9%), 2,249 non-rumour (48.3%), 132 unlabelled (2.8%).

### Phase 2 — BiGCN Classification (Twitter-15, 5-fold CV, partial)

| Metric | Value |
|--------|-------|
| 4-class accuracy | 0.83 |
| 4-class macro F1 | 0.76 |
| Folds completed | 2 of 5 (fold 0, fold 1) |
| Training device | CPU (reason for early stop) |
| Previous baseline | acc=0.77, F1=0.68 |
| Improvement | +0.06 acc, +0.08 F1 |

Note: 5-fold CV not completed. Published BiGCN baseline (Bian et al.) is higher;
current gap is attributed to CPU-only training (fewer epochs before wall-clock
budget was exhausted) and possibly insufficient hyperparameter tuning.

### Phase 3 & 4 — Status

SBM fitter: code implemented and unit-tested. Not yet run on full WICO graph.
LP optimizer: code implemented (scipy interior-point). Not yet run end-to-end.
No quantitative results available for Phase 3/4 yet.

---

## [DATA] ARCHITECTURE DETAILS

### BiGCN Model

```
Input: 772-dim node features per cascade tree
  ├── 768 dim: RoBERTa-base CLS token (frozen), broadcast from root to all nodes
  └── 4 dim: structural per-node [is_root, norm_depth, norm_breadth, norm_descendants]

Two parallel GCN branches (top-down + bottom-up):

GCNBranch(direction):
  Layer 1:  GCNConv(772 → 256)  +  ReLU  +  Dropout(0.5)
  Augment:  concatenate root features (772 dim) to every node → 1028 dim
  Layer 2:  GCNConv(1028 → 128) +  ReLU  +  Dropout(0.5)
  Pool:     global mean pooling → (batch_size, 128)

Classifier head:
  concat(td_pool, bu_pool) → (batch_size, 256)
  Linear(256 → 128) → ReLU → Dropout(0.5)
  Linear(128 → 4)   → logits (4-class)
```

Root enhancement (the critical design decision): during Layer 2, the 772-dim
source-tweet representation is concatenated to every node's intermediate
embedding. This ensures the content signal from the original post propagates
bidirectionally throughout the entire tree, not just from the source outward.
Without this, deep subtrees lose source context after two GCN hops.

### Hyperparameters

| Param | Value | Reason |
|-------|-------|--------|
| lr | 5e-4 | Standard for Adam with frozen large encoder |
| weight_decay | 5e-4 | L2 regularization balances overfitting on ~1500 cascades |
| batch_size | 32 | Empirically safe for CPU memory; gradient noise acceptable |
| dropout | 0.5 | Aggressive but standard for small-data GNNs |
| early_stop patience | 20 epochs | Prevent overfitting; dataset too small for more |
| label_smoothing | 0.05 | Reduces overconfidence on noisy social media labels |
| hidden_dim | 256 | Balances capacity vs risk of overfitting |
| out_dim | 128 | Standard for cascade graph classification tasks |

### SBM

- k = 13 polarization classes (from Louvain, resolution 2.0)
- Min partition fraction: 0.01 (partitions < 1% of users merged into nearest class)
- Confidence threshold for feeding GNN predictions to SBM: 0.65

### LP Parameters

- α = 1.5: Minimum branching ratio preserved for true content
- λ = 1.0: Equal weight on false suppression vs true preservation in softened LP
- Dropout bounds: [0.0, 1.0] per (class u, class v) pair
- Solver: scipy.optimize.linprog, interior-point method

---

## [DATA] DATASETS

| Dataset | Platform | Language | Classes | N cascades | Format |
|---------|----------|----------|---------|-----------|--------|
| Twitter-15 | Twitter/X | English | 4-class (true/false/unverified/non-rumour) | 1,490 | One .txt per cascade, BiGCN edge format |
| Twitter-16 | Twitter/X | English | 4-class | similar to T15 | Same format |
| WICO | Twitter/X | English | 3-class (5G-conspiracy, conspiracy, non-conspiracy) | ~4,000 | Per-tweet folders, edges.txt + nodes.csv |
| Weibo | Sina Weibo | Chinese | 2-class (rumour / non-rumour) | 4,659 | Single flat file weibotree.txt, tab-separated |

Twitter-15 edge format per line:
  `['p_uid','p_tid','p_delay'] -> ['c_uid','c_tid','c_delay']`
  ROOT sentinel: `['ROOT','ROOT','0.0'] -> [...]`

Weibo format per line (tab-separated):
  `event_id  parent_id  node_id  bow_features...`
  parent_id = "None" marks root node of that event.

Label encoding:
- Twitter-15 binary: true=0, non-rumour=0 (trusted); false=1 (misinformation); unverified=skipped
- Weibo binary: 0=rumour (misinformation), 1=non-rumour (trusted)
- WICO binary: 5g-conspiracy=false, conspiracy=false, non-conspiracy=true

---

## REPORT STRUCTURE — WRITE EACH SECTION AS FOLLOWS

### 1. Abstract (150 words max)

State the problem (platform-specific misinformation spread), the approach
(BiGCN classification → SBM cascade modelling → LP suppression), and the
key results so far (Phase 1: statistically confirmed cross-community
structural differences; Phase 2: 0.83 acc / 0.76 F1 on partial fold).
Do not overstate. If something is incomplete, the abstract should reflect
that honestly.

### 2. Introduction & Motivation

Open with the engineering problem, not with vague statements about
"misinformation is bad." The specific problem: a system trained on one
community's spreading patterns will be miscalibrated on another community.
Quantify the mismatch: 71% width difference, 31% depth difference between
Twitter-15 and Weibo. These are real numbers that justify the architecture.

Explain the three-layer design:
- Detection layer (BiGCN): answers "is this post misinformation?"
- Network model layer (SBM): answers "who will this spread to, and how fast?"
- Intervention layer (LP): answers "which connections to dampen, and by how much?"

Explain why each layer needs to be community-specific.

### 3. Related Work

Keep this short (half a page). Only reference the following areas:
- BiGCN (Bian et al.) — the classification backbone we adapted
- SBM cascade models — the theoretical framework for b⁺/b⁻ matrices
- LP-based intervention — the suppression algorithm
Do NOT write generic literature survey. Situate each reference relative to
a specific design decision we made.

### 4. System Architecture (Top-Level)

Draw the pipeline as text or describe it section by section. Emphasize the
data flow: raw social media graphs → feature extraction → classification →
SBM fitting → per-step LP solve → dropout matrix d* applied to SIR simulation.

Make the bidirectional dependency explicit: the LP optimizer needs the SBM
matrices from Phase 3, which need the community partition from the union
graph, which needs GNN classifications from Phase 2 to distinguish true vs
false cascade edges for the b⁺/b⁻ split.

### 5. Phase 1: Cross-Community Cascade Analysis

This section should establish the empirical justification for community-specific
parameters. Report the four metrics (depth, width, d/w ratio, Gini) with the
exact numbers above. Report the statistical test results. Then give the
engineering interpretation: a model using Twitter-15 priors on Weibo would
have its width estimate off by 71% and depth off by 31% before classification.

Report the within-Weibo label comparison. The 4.1x d/w ratio gap between
rumour and non-rumour classes shows that cascade shape alone carries
discriminating information — the GNN does not need to rely entirely on text.

Be honest about what is NOT yet done in Phase 1 (temporal dynamics not
analysed, missing-label bias not checked, WICO structural analysis not done).

### 6. Phase 2: BiGCN Model

#### 6.1 Architecture

Describe the model precisely: input dimensionality (772 = 768 RoBERTa + 4
structural), two parallel GCNBranch objects, root enhancement mechanism,
classifier head. Use the architecture table above.

Justify the root enhancement design decision in one paragraph: without it,
a GCN with L=2 layers can only propagate source context 2 hops. For cascades
with mean depth 3.96–5.19, that leaves deep subtrees content-agnostic. The
augmentation at Layer 2 solves this by concatenating root features to every
node's intermediate representation.

Justify freezing RoBERTa: fine-tuning a 125M-parameter encoder on ~1,490
cascades would overfit. Using frozen CLS embeddings gives consistent,
pre-trained semantic representations without gradient interference.

#### 6.2 Training Setup

Describe the training loop, optimizer settings, and evaluation protocol.
k-fold CV with k=5; per-fold checkpoint saved; early stopping at patience=20.

#### 6.3 Results

Report results honestly. State which folds were completed (0 and 1 only).
Report 0.83 accuracy and 0.76 macro F1. Acknowledge gap to published BiGCN
baseline. State the attributed cause: CPU-only training limits wall-clock
budget and therefore epoch count. Do NOT claim this is a solved problem.

#### 6.4 Limitations

- Full 5-fold CV not completed (only 2 folds run)
- CPU training significantly slows iteration
- No hyperparameter search performed beyond config defaults
- Weibo dataset not yet integrated into BiGCN training (Phase 1 only)

### 7. Phase 3: Stochastic Block Model

Describe the SBM formulation. The user network has k=13 polarization classes
determined by Louvain modularity clustering (resolution 2.0) on the union
graph of all WICO cascade edges. Tiny classes (< 1% of users) are merged.

State the frequentist MLE for b_uv:
  b_uv = (observed transfers from class-u users to class-v users) / |Cu|

Explain the two separate matrices: b⁺ fitted on true-labeled cascades only,
b⁻ fitted on false-labeled cascades only. The GNN confidence threshold of
0.65 gates which cascades feed into the SBM fitter.

State clearly: SBM code is implemented but fitting has not yet been run on
the full WICO graph. No numerical results to report yet.

### 8. Phase 4: LP-Based Dropout Optimization

Describe Algorithm 2: interleave LP solve with SIR simulation. At each time
step t, given current |S^v_t| and |I^u_t|, solve the PRIMARY LP to minimize
expected false-content spread subject to preserving branching ratio ≥ α for
true content. If primary LP is infeasible (true content naturally slowing),
fall back to SOFTENED LP with weighted objective.

Write out both LP formulations (using the equations from the context section
above). Explain the α=1.5 choice: it requires the optimal dropout policy to
sustain at least 1.5x regeneration of true content, preventing over-aggressive
suppression from silencing both true and false content simultaneously.

State clearly: LP code is implemented. End-to-end Algorithm 2 evaluation not
yet run. No suppression effectiveness numbers to report.

### 9. Engineering Decisions Summary

Write a compact table or bullet list of the key architectural decisions and
their explicit justifications. Examples to cover:

- Frozen RoBERTa vs fine-tuned: (stated above)
- Root enhancement at Layer 2 vs Layer 1 or not at all
- Two-phase LP (primary + softened) vs single LP with slack variables
- Frequentist MLE for SBM vs Bayesian estimation
- Louvain resolution 2.0 → k=13 vs fewer classes (lower resolution)
- Community-specific b matrices vs single global matrix

For each: state the decision, the alternative considered, and the reason for
the choice. Do not write "we chose X because X is good." Write the actual
technical reason.

### 10. Current Status & Roadmap

Be precise about what is done vs not done. Use this breakdown:

| Phase | Status | Blocker |
|-------|--------|---------|
| Phase 1: Cascade analysis (Twitter-15 vs Weibo) | Complete | — |
| Phase 1: WICO structural analysis | Not done | deprioritised |
| Phase 2: BiGCN training (2/5 folds) | Partial | CPU-only training speed |
| Phase 2: Full 5-fold CV | Blocked | Same |
| Phase 2: Weibo BiGCN training | Not started | — |
| Phase 3: SBM fitting on WICO | Not run | Requires Phase 2 predictions |
| Phase 4: LP end-to-end evaluation | Not run | Requires Phase 3 matrices |

Roadmap: the critical path is (a) complete BiGCN folds with GPU, (b) run
SBM fitter on WICO predictions, (c) run Algorithm 2 and measure suppression
effectiveness vs baseline (no dropout).

### 11. Conclusion

Summarize what has been empirically established vs what is still theoretical:

Established:
- Cross-community structural differences are large (71% width, 31% depth,
  p < 0.001) and justify community-specific SBM parameters.
- Within-Weibo, rumour and non-rumour cascades differ 4.1x in d/w ratio —
  structural shape is a useful discriminating feature.
- BiGCN achieves 0.83 accuracy and 0.76 macro F1 on partial 5-fold CV,
  improving over the previous version by +0.06 accuracy and +0.08 F1.

Still theoretical / not yet measured:
- SBM fitting quality on WICO
- LP suppression effectiveness (reduction in false-content cascade size)
- Trade-off curve between false suppression and true content collateral damage
- System performance on Weibo (Chinese, different script, no RoBERTa fine-tune)

---

## TONE AND STYLE RULES

1. **No filler phrases**: Never write "it is worth noting", "it is interesting
   that", "as mentioned above", "moving on to the next section."

2. **Specific over vague**: "The Gini coefficient of the depth distribution
   rises from 0.21 on Twitter-15 to 0.35 on Weibo" is better than "Weibo
   has more unequal spreading patterns."

3. **Own your decisions**: Never write "we chose X following [Author]."
   Write "we chose X because [technical reason]." Citation is fine for
   background but not as a substitute for engineering reasoning.

4. **State unknowns explicitly**: If a number is not yet measured, say so in
   one sentence. Do not hedge with paragraphs of caveats.

5. **Math where it clarifies**: Write the LP objective and constraint in-line
   where you introduce Phase 4. Write the SBM MLE formula in Phase 3. Do not
   use math as decoration.

6. **Tables over prose for comparisons**: Use the data tables from the [DATA]
   sections. Do not re-describe numerical comparisons in prose when a table
   is clearer.

7. **No passive voice for decisions**: "The decision was made to freeze
   RoBERTa" → "We froze RoBERTa because..."

8. **Section lengths**: Introduction ~400 words. Architecture overview ~300
   words. Each Phase section ~400-600 words. Conclusion ~200 words.
   Total target: 3,500–4,500 words excluding tables and figures.

---

## WHAT NOT TO WRITE

- Do not claim Phase 3 or Phase 4 have been validated. They have not.
- Do not compare to published state-of-the-art beyond BiGCN baseline.
  We have not run those comparisons.
- Do not claim the system is "ready for deployment."
- Do not add sections on ethics, social impact, or future work beyond the
  roadmap. These are not yet supported by measurements.
- Do not invent results. Use only the numbers in the [DATA] sections.

---

## OUTPUT FORMAT

Write the full report as continuous prose with section headers. Include:
- All data tables from the [DATA] sections where referenced
- The BiGCN architecture table
- The phase status table in Section 10
- Inline math for the LP formulations and SBM MLE

Output in Markdown. Use `##` for section headers, `###` for subsections.
Code blocks for architecture diagrams if needed. No footnotes.
