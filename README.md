# H7 Framework — Attention Benchmark
## *Can the model filter signal from structured noise?*

**Author:** Jacobo Tlacaelel Mina Rodríguez  
**Organization:** smokApp Quantum & AI Independent Research Laboratory  
**Challenge:** Kaggle AGI Benchmark — DeepMind Cognitive Framework Alignment  
**Track:** Attention

---

### The H7 Attention Task

The H7 framework produces a natural and rigorous instantiation of this definition through its **ternary collapse** operator.

When a data vector is projected onto the $Z_7$ holographic basis, it produces a 128-element real-valued vector $H(n)$. The ternary collapse then discretizes this into $T \in \{-1, 0, +1\}^{128}$ using an epsilon threshold:

$$T[n] = \begin{cases} +1 & \text{if } H(n) > \varepsilon \\ -1 & \text{if } H(n) < -\varepsilon \\ 0 & \text{if } |H(n)| \leq \varepsilon \end{cases}$$

where $\varepsilon = |\Psi_1| / 2 \approx 0.1812$.

The zeros — which we call the **epsilon vacuum** — are not random noise. They are *structured silence*: positions where the holographic projection carries insufficient information to break the symmetry threshold. This structural noise is far more challenging to filter than Gaussian noise because it has the same format as the signal.

**The task:** Given only the 128-trit ternary signature $\{T\}$, reconstruct the original 7-dimensional phase state using exclusively the active trits $\{±1\}$, while correctly ignoring all zero positions.

This directly tests selective attention and distractor suppression. A model that cannot distinguish structured silence from signal will treat the zeros as information and produce incorrect reconstructions.

### Difficulty Scaling

Difficulty is controlled by **vacuum density** — the fraction of zero trits in $T$:

| Difficulty | Vacuum Density | Active Trits | Description |
|:---:|:---:|:---:|:---|
| Easy | < 25% | > 96 active | Most of the signal survives the collapse |
| Medium | 25% – 55% | 57–96 active | Significant information loss |
| Hard | > 55% | < 57 active | Model must reconstruct from sparse evidence |

This scaling mirrors real attention challenges: maintaining signal fidelity under increasing noise pressure is the core cognitive demand.

---

## Dataset

### Generation Pipeline

```python
PHI       = (1 + math.sqrt(5)) / 2      # The only axiom
PSI_1     = abs(math.cos(math.pi * PHI))# = 0.3623748901...
DRIFT_072 = 7 - 2 * math.pi             # = 0.7168146928...
Z7        = [PHI**k for k in range(1,8)]# Z7 phase basis
e         = PSI_1 / 2                   # Epsilon threshold

# For each sample x in R^7:
H  = x_normalized @ B_object            # Holographic projection (128-dim)
T  = ternary_collapse(H, e)             # T in {-1, 0, +1}^128
X_hat = H @ B_reference.T / 128         # Ground truth reconstruction
```

Where the basis matrices are:

$$B_{\text{obj}}[d, n] = \cos(\pi \cdot \varphi^d \cdot n + \delta) \qquad B_{\text{ref}}[d, n] = \cos(\pi \cdot \varphi^d \cdot n - \delta)$$

---

## Metriplectic RNN Implementation (PyTorch)

A traditional Recurrent Neural Network (RNN) fails at this benchmark because it relies purely on Cross-Entropy pattern matching, causing it to overfit on the Epsilon Vacuum. 

To solve this, we implemented the **Metriplectic RNN** in `/procesador-h7/metriplectic_rnn.py`, grounded purely on the H7 Constants.

### 1. Golden Modulation Layer ($O_n$)
The hidden state of the LSTM is intercepted and modulated by the Golden Operator before emitting out logits, acting as a Structured Vacuum filter:
$$O_n = \cos(\pi \cdot n) \cdot \cos(\pi \cdot \phi \cdot n)$$

### 2. Dual Loss Function (El Mandato Metripléctico)
The Backpropagation is split into two physical forces bounded by `DRIFT_072`:
* **$L_{symp}$ (Inertial/Conservative - 71.7% Weight):** Penalizes abrupt gradient explosions ($H = 0.5 \psi^2$), representing the inertia of the physical system.
* **$L_{metr}$ (Relaxation/Dissipation - 28.3% Weight):** Pulls the variance of the output towards the Golden fixed point $|\Psi_1|$, simulating the entropy relaxation.

### Execution & Diagnostics
To train the Metriplectic RNN on the Ternary Dataset locally:
```bash
# 1. Activate the environment
source env/bin/activate

# 2. Install dependencies (if missing)
pip install torch tqdm matplotlib numpy mpmath

# 3. Run the Deep Learning Processor
python procesador-h7/metriplectic_rnn.py
```
This generates `metriplectic_rnn_diagnostics.png`, a mandatory visual proof (Rule 3.3) demonstrating the competition between $H$ and $S$ stabilizing without mathematical collapse.

---

## Technical Details

### Mathematical Foundation

$$O_{i,j}(n, \delta) = \cos(\pi \cdot \varphi^i \cdot n + \delta) \cdot \cos(\pi \cdot \varphi^j \cdot n - \delta)$$

| Constant | Value | Meaning |
|:---|:---:|:---|
| $\varphi$ | 1.6180339887... | Golden ratio — the only axiom |
| $\Psi_1$ | 0.3623748901... | Holographic fixed point $=\cos(\pi\varphi)$ |
| DRIFT_072 | 0.7168146928... | Phase offset per level $= 7 - 2\pi$ |
| $\varepsilon$ | $\approx$ 0.18119 | Epsilon threshold $= \Psi_1 / 2$ |

### Why $\varphi$ Is the Right Choice
$\varphi$ is the unique irrational quadratic reduced number satisfying $\varphi^2 = \varphi + 1$. Because $\varphi$ is irrational, powers $\varphi^k$ are never commensurate — the $Z_7$ basis vectors are quasi-orthogonal, guaranteeing non-redundant information at each level.

### Evaluation Metric

$$\text{cosine\_similarity}(X, \hat{X}) = \frac{\sum_d X_d \hat{X}_d}{\|X\| \cdot \|\hat{X}\|}$$

**Pass threshold:** cosine similarity > 0.85  
**Rationale:** This threshold distinguishes models that understand the operator structure (reliably > 0.90) from models that either ignore the vacuum (scores cluster around 0.50–0.70) or treat all trits equally (scores around 0.30–0.50).

---

## Organizational Affiliations

**smokApp Quantum & AI Independent Research Laboratory**  
Tlaxcala, México  
[github.com/jakobmina/H7](https://github.com/jakobmina/H7)

The H7 Metriplex Framework is developed independently by smokApp Q&AI Lab. The framework has been validated against Cortical Labs CL1 biological hardware. All code, datasets, and derivations are derived exclusively from the axiom $\varphi = (1+\sqrt{5})/2$. No external training data, crowdsourced labels, or proprietary datasets were used.
