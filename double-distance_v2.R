# First stage experimental
# H7 AGI Benchmark — Predictor (R) 
# Vietoris-Rips inspirado: bola epsilon sobre distancia dual
# Euclidiana + Mahalanobis con expansion adaptativa.

setwd("/kaggle/input/datasets/jakomina/database")
.libPaths(c("/home/runner/R/library", .libPaths()))
suppressMessages({
  library(MASS)
})

# ── Constantes H7 ─────────────────────────────────────────────────────────────
PHI       <- (1 + sqrt(5)) / 2
PSI_1     <- abs(cos(pi * PHI))
DRIFT_072 <- 7 - 2 * pi

# ── Features por track ────────────────────────────────────────────────────────
TRACK_FEATURES <- list(
  learning            = c("phi_i", "phi_j", "level_k", "n_index"),
  metacognition       = c("amplitude", "psi1"),
  attention           = c("epsilon_density", "active_trits"),
  executive_functions = c("level_k", "amplitude", "re_i"),
  social_cognition    = c("amp_a", "amp_b")
)

ALPHA      <- 0.5   # peso Euclidiana vs Mahalanobis
EPS_PCTILE <- 20    # percentil para calibrar epsilon
EPS_EXPAND <- 1.5   # factor de expansion si la bola queda vacia
MAX_EXPAND <- 10    # maximas expansiones antes de fallback


# ── Escalado (StandardScaler equivalente) ─────────────────────────────────────
std_scale <- function(X_train, X_test) {
  mu  <- colMeans(X_train)
  sd  <- apply(X_train, 2, sd) + 1e-9
  list(
    X_tr = sweep(sweep(X_train, 2, mu, "-"), 2, sd, "/"),
    X_te = sweep(sweep(X_test,  2, mu, "-"), 2, sd, "/")
  )
}


# ── Precision matrix robusta (Sigma^-1) ───────────────────────────────────────
precision_mat <- function(X) {
  tryCatch({
    S <- cov(X) + diag(1e-6, ncol(X))
    solve(S)
  }, error = function(e) {
    ginv(cov(X) + diag(1e-6, ncol(X)))
  })
}


# ── Distancia dual (n_test x n_train) ─────────────────────────────────────────
dual_distance <- function(X_tr, X_te, VI, alpha = ALPHA) {
  n_te <- nrow(X_te)
  n_tr <- nrow(X_tr)

  # Euclidiana
  eucl <- matrix(0, n_te, n_tr)
  for (i in seq_len(n_te)) {
    diff     <- sweep(X_tr, 2, X_te[i, ], "-")
    eucl[i,] <- sqrt(rowSums(diff^2))
  }

  # Mahalanobis
  maha <- matrix(0, n_te, n_tr)
  for (i in seq_len(n_te)) {
    diff     <- sweep(X_tr, 2, X_te[i, ], "-")
    maha[i,] <- sqrt(pmax(rowSums((diff %*% VI) * diff), 0))
  }

  alpha * eucl + (1 - alpha) * maha
}


# ── Calibrar epsilon desde distancias train-train ─────────────────────────────
calibrate_eps <- function(D_tt, pctile = EPS_PCTILE) {
  diag(D_tt) <- NA
  as.numeric(quantile(D_tt, pctile / 100, na.rm = TRUE))
}


# ── Bola VR: indices + pesos ──────────────────────────────────────────────────
vr_ball <- function(d_row, eps) {
  current_eps <- eps
  for (k in seq_len(MAX_EXPAND)) {
    idx <- which(d_row <= current_eps)
    if (length(idx) > 0) {
      w <- 1 / (d_row[idx] + 1e-9)
      return(list(idx = idx, w = w / sum(w)))
    }
    current_eps <- current_eps * EPS_EXPAND
  }
  # fallback: vecino mas cercano
  nn <- which.min(d_row)
  list(idx = nn, w = 1.0)
}


# ── Parsers ───────────────────────────────────────────────────────────────────
parse_meta <- function(t) {
  parts <- strsplit(trimws(t), " ")[[1]]
  dev   <- as.numeric(sub("deviation=", "", parts[1]))
  conf  <- sub("confidence=", "", parts[2])
  list(dev = dev, conf = conf)
}

parse_vec7 <- function(t) {
  cleaned <- gsub("[\\[\\]\\s]", "", t, perl = TRUE)
  vals    <- suppressWarnings(as.numeric(strsplit(cleaned, ",")[[1]]))
  vals
}


# ── Voto mayoritario ponderado ────────────────────────────────────────────────
weighted_vote <- function(targets, weights) {
  tab <- tapply(weights, targets, sum)
  names(tab)[which.max(tab)]
}


# ── Predictor por track ───────────────────────────────────────────────────────
predict_track <- function(track, train, test) {
  feats  <- TRACK_FEATURES[[track]]
  tr_sub <- train[train$track == track, ]
  te_sub <- test[test$track   == track, ]

  if (nrow(tr_sub) == 0 || nrow(te_sub) == 0) return(data.frame())

  # Reemplazar NA por 0
  tr_sub[, feats] <- lapply(tr_sub[, feats], function(x) ifelse(is.na(x), 0, as.numeric(x)))
  te_sub[, feats] <- lapply(te_sub[, feats], function(x) ifelse(is.na(x), 0, as.numeric(x)))

  scaled   <- std_scale(as.matrix(tr_sub[, feats]), as.matrix(te_sub[, feats]))
  X_tr     <- scaled$X_tr
  X_te     <- scaled$X_te

  VI  <- precision_mat(X_tr)
  D_tt <- dual_distance(X_tr, X_tr, VI)
  eps  <- calibrate_eps(D_tt, EPS_PCTILE)
  D    <- dual_distance(X_tr, X_te, VI)

  avg_ball <- mean(sapply(seq_len(nrow(X_te)), function(i) sum(D[i,] <= eps)))
  cat(sprintf("    epsilon=%.4f  |  vecinos promedio: %.1f\n", eps, avg_ball))

  preds <- character(nrow(te_sub))

  # ── Learning ────────────────────────────────────────────────────────────
  if (track == "learning") {
    y <- as.numeric(tr_sub$target_numeric)
    for (i in seq_len(nrow(te_sub))) {
      b    <- vr_ball(D[i,], eps)
      pred <- sum(b$w * y[b$idx])
      preds[i] <- sprintf("%.8f", pred)
    }

  # ── Metacognition ────────────────────────────────────────────────────────
  } else if (track == "metacognition") {
    parsed    <- lapply(tr_sub$target, parse_meta)
    devs      <- sapply(parsed, `[[`, "dev")
    confs     <- sapply(parsed, `[[`, "conf")
    for (i in seq_len(nrow(te_sub))) {
      b         <- vr_ball(D[i,], eps)
      pred_dev  <- sum(b$w * devs[b$idx])
      pred_conf <- weighted_vote(confs[b$idx], b$w)
      preds[i]  <- sprintf("deviation=%.6f confidence=%s", pred_dev, pred_conf)
    }

  # ── Attention ────────────────────────────────────────────────────────────
  } else if (track == "attention") {
    vecs <- t(sapply(tr_sub$target, parse_vec7))
    for (i in seq_len(nrow(te_sub))) {
      b      <- vr_ball(D[i,], eps)
      pred_v <- colSums(vecs[b$idx, , drop = FALSE] * b$w)
      preds[i] <- paste0("[", paste(round(pred_v, 4), collapse = ", "), "]")
    }

  # ── Executive functions ──────────────────────────────────────────────────
  } else if (track == "executive_functions") {
    targets <- as.character(tr_sub$target)
    for (i in seq_len(nrow(te_sub))) {
      b        <- vr_ball(D[i,], eps)
      preds[i] <- weighted_vote(targets[b$idx], b$w)
    }

  # ── Social cognition ─────────────────────────────────────────────────────
  } else if (track == "social_cognition") {
    targets <- as.character(tr_sub$target)
    for (i in seq_len(nrow(te_sub))) {
      b        <- vr_ball(D[i,], eps)
      preds[i] <- weighted_vote(targets[b$idx], b$w)
    }
  }

  data.frame(id = te_sub$id, target = preds, stringsAsFactors = FALSE)
}


# ── Evaluacion interna ────────────────────────────────────────────────────────
evaluate <- function(submission) {
  ans_path <- "/kaggle/input/notebooks/jakomina/attention-ipynb/submission"
  if (!file.exists(ans_path)) {
    cat("  (test_with_answers.csv no encontrado)\n")
    return(invisible(NULL))
  }

  answers <- read.csv(ans_path, stringsAsFactors = FALSE)
  m       <- merge(answers, submission, by = "id", suffixes = c("_true", "_pred"))

  cat("\n── Evaluacion interna ──────────────────────────────────────\n")

  # Learning
  sub <- m[m$track == "learning", ]
  if (nrow(sub) > 0) {
    mae <- mean(abs(sub$target_numeric - as.numeric(sub$target_pred)))
    cat(sprintf("  learning            MAE          : %.6f\n", mae))
  }

  # Metacognition
  sub <- m[m$track == "metacognition", ]
  if (nrow(sub) > 0) {
    dt  <- sapply(sub$target_true, function(t) parse_meta(t)$dev)
    dp  <- sapply(sub$target_pred, function(t) parse_meta(t)$dev)
    ct  <- sapply(sub$target_true, function(t) parse_meta(t)$conf)
    cp  <- sapply(sub$target_pred, function(t) parse_meta(t)$conf)
    cat(sprintf("  metacognition       MAE dev      : %.6f  |  conf acc: %.2f%%\n",
                mean(abs(dt - dp)), 100 * mean(ct == cp)))
  }

  # Attention
  sub <- m[m$track == "attention", ]
  if (nrow(sub) > 0) {
    sims <- mapply(function(t, p) {
      vt <- suppressWarnings(parse_vec7(t))
      vp <- suppressWarnings(parse_vec7(p))
      if (any(is.na(vt)) || any(is.na(vp))) return(NA_real_)
      nt <- sqrt(sum(vt^2)); np_ <- sqrt(sum(vp^2))
      if (nt > 0 && np_ > 0) sum(vt * vp) / (nt * np_) else NA_real_
    }, sub$target_true, sub$target_pred)
    cat(sprintf("  attention           cosine sim   : %.4f\n", mean(sims, na.rm = TRUE)))
  }

  # Executive + Social
  for (track in c("executive_functions", "social_cognition")) {
    sub <- m[m$track == track, ]
    if (nrow(sub) > 0) {
      acc <- mean(sub$target_true == sub$target_pred)
      cat(sprintf("  %-22s  exact match  : %.2f%%\n", track, 100 * acc))
    }
  }
}


# ── Main ──────────────────────────────────────────────────────────────────────
main <- function() {
  train <- read.csv("/kaggle/input/datasets/jakomina/database/train.csv", stringsAsFactors = FALSE)
  test  <- read.csv("/kaggle/input/datasets/jakomina/database/test.csv",  stringsAsFactors = FALSE)

  cat(sprintf("Train: %d filas | Test: %d filas\n", nrow(train), nrow(test)))
  cat(sprintf("Metodo: VR-ball  alpha=%.1f  eps-pctile=%d%%  expand x%.1f\n\n",
              ALPHA, EPS_PCTILE, EPS_EXPAND))

  all_preds <- list()
  for (track in names(TRACK_FEATURES)) {
    cat(sprintf("  [%s]\n", track))
    p <- predict_track(track, train, test)
    cat(sprintf("    -> %d predicciones\n\n", nrow(p)))
    all_preds[[track]] <- p
  }

  submission <- do.call(rbind, all_preds)

  # Reordenar segun sample_submission
  sample_sub <- read.csv("/kaggle/input/datasets/jakomina/database/sample_submission.csv",
                         stringsAsFactors = FALSE)
  submission <- merge(sample_sub["id"], submission, by = "id", all.x = TRUE)
  submission$target[is.na(submission$target)] <- "0.00000000"

  out <- "/kaggle/working/submission.csv"
  write.csv(submission, out, row.names = FALSE, quote = TRUE)
  cat(sprintf("Submission -> %s  (%d filas)\n", out, nrow(submission)))

  evaluate(submission)
}

main()

# H7 AGI Benchmark — Predictor

## Executive Summary

This R script implements a **Vietoris-Rips inspired topological predictor** for the H7 AGI Benchmark competition. It leverages dual-distance metrics (Euclidean + Mahalanobis) with adaptive epsilon-ball expansion to perform multi-track cognitive classification and regression across five distinct capability domains.

---

## Table of Contents

1. [Overview](#overview)
2. [Technical Architecture](#technical-architecture)
3. [Core Components](#core-components)
4. [Feature Engineering](#feature-engineering)
5. [Prediction Pipeline](#prediction-pipeline)
6. [Usage Instructions](#usage-instructions)
7. [Dependencies](#dependencies)

---

## Overview

### Purpose

The H7 AGI Benchmark evaluates artificial general intelligence across five cognitive tracks:
- **Learning**: Numerical regression of learning capacity
- **Metacognition**: Deviation estimation + confidence classification
- **Attention**: Multi-dimensional vector prediction
- **Executive Functions**: Discrete state classification
- **Social Cognition**: Categorical relationship classification

This predictor implements a **non-parametric, topology-aware ensemble method** that:
- Combines Euclidean and Mahalanobis distance metrics
- Adapts local neighborhood size via epsilon calibration
- Handles heterogeneous output formats (continuous, categorical, vector-valued)
- Provides weighted voting and ensemble aggregation

### Key Innovations

1. **H7 Mathematical Framework**: Integration of golden ratio (φ) and topological symmetry principles
   - Golden ratio constant: **φ = (1 + √5) / 2 ≈ 1.618**
   - Primary harmonic: **Ψ₁ = |cos(π·φ)| ≈ 0.3624**
   - Drift constant: **DRIFT_072 = 7 − 2π ≈ 0.717**

2. **Dual-Distance Metric**: Weighted combination of:
   - **Euclidean distance** (global structure)
   - **Mahalanobis distance** (covariance-aware local geometry)
   - **Weighting factor α = 0.5** (balanced contribution)

3. **Adaptive Epsilon Calibration**:
   - Percentile-based (20th percentile of train-train distances)
   - Automatic expansion if neighborhoods are empty
   - Maximum 10 expansions before fallback to nearest neighbor

---

## Technical Architecture

### Distance Computation Pipeline

```
Input: X_train (n × p), X_test (m × p)
  ↓
[StandardScaler]
  ↓
[Precision Matrix Estimation] → Σ⁻¹ (covariance inverse)
  ↓
[Dual Distance Matrix]
  ├─ D_eucl = √(Σ(diff²))
  ├─ D_maha = √((diff·Σ⁻¹)·diff)
  └─ D_dual = α·D_eucl + (1−α)·D_maha
  ↓
[Epsilon Calibration] → eps = quantile(D_train, 20%)
  ↓
[VR Ball Expansion] → Local neighborhoods
  ↓
[Weighted Aggregation] → Final predictions
```

### Regularization Strategy

The precision matrix (Σ⁻¹) is estimated with **adaptive ridge regularization**:

```
λ = 1e-4 × max(diag(S))
Σ⁻¹ = solve(S + λI)
```

Fallback to Penrose-Moore pseudo-inverse if singularity occurs.

---

## Core Components

### 1. Feature Extraction (`TRACK_FEATURES`)

Each cognitive track uses domain-specific feature subsets:

| Track | Features | Semantics |
|-------|----------|-----------|
| **learning** | phi_i, phi_j, level_k, n_index | Learning phase indicators |
| **metacognition** | amplitude, psi1 | Self-awareness metrics |
| **attention** | epsilon_density, active_trits | Attention concentration |
| **executive_functions** | level_k, amplitude, re_i | Control & planning capacity |
| **social_cognition** | amp_a, amp_b | Social signal amplitudes |

### 2. Standardization (`std_scale`)

Applies Z-score normalization:
- **Mean centering**: X_centered = X − μ
- **Variance scaling**: X_scaled = X_centered / (σ + ε)
- Numerical stability: ε = 1e-9

### 3. Precision Matrix Estimation (`precision_mat`)

Computes robust inverse covariance via spectrum-aware regularization:
```R
S <- cov(X)                    # Empirical covariance
lambda <- 1e-4 * max(diag(S))  # Adaptive penalty
Σ⁻¹ = solve(S + λI)            # Ridge-regularized inverse
```

### 4. Dual Distance (`dual_distance`)

Blends two complementary metrics:

**Euclidean Component** (global context):
```
d_eucl[i,j] = √(Σₖ (X_test[i,k] − X_train[j,k])²)
```

**Mahalanobis Component** (variance-normalized):
```
d_maha[i,j] = √((X_diff[i,j] · Σ⁻¹) · X_diff[i,j]ᵀ)
```

**Weighted Combination**:
```
d_dual = α·d_eucl + (1−α)·d_maha    (α = 0.5)
```

### 5. Epsilon Calibration (`calibrate_eps`)

Sets local neighborhood radius from empirical distribution:
```R
eps <- quantile(D_train_train, 20%)  # 20th percentile
```

This ensures balanced locality: ~20% of neighbors within radius.

### 6. Vietoris-Rips Ball (`vr_ball`)

Finds neighbors within distance threshold with adaptive expansion:

```
while (neighbors_found == 0) and (expansions < MAX_EXPAND):
  neighbors = {j : d[i,j] <= current_eps}
  if neighbors_found:
    return weighted_neighbors
  else:
    current_eps *= EPS_EXPAND (×1.5)

fallback: return nearest neighbor
```

Weights are inverse-distance normalized:
```
w[j] = (1 / d[i,j]) / Σₖ(1 / d[i,k])
```

---

## Feature Engineering

### Target Preprocessing

**Learning Track** (Continuous):
- Parsed as numeric
- Predicted via weighted average of neighborhood targets

**Metacognition Track** (Structured):
- Format: `"deviation=X.XXX confidence=CLASS"`
- Deviation: weighted mean
- Confidence: weighted majority vote

**Attention Track** (Vector):
- Format: `"[v₁, v₂, v₃, v₄, v₅, v₆, v₇]"`
- Parsed via regex
- Predicted via weighted column-wise mean

**Executive/Social Tracks** (Categorical):
- Discrete classes
- Predicted via weighted majority voting:
  ```
  pred = argmax_class Σⱼ∈neighbors w[j] · [target[j] == class]
  ```

### Missing Value Handling

- **Strategy**: Replace NA with 0 (assumes neutral/resting state)
- **Justification**: H7 framework treats 0 as baseline ternary state {−1, 0, +1}
- **Applied to**: All feature columns independently

---

## Prediction Pipeline

### Per-Track Workflow

For each cognitive domain:

1. **Subset Data**: Filter train/test by track
2. **Feature Selection**: Extract domain-relevant features
3. **Standardize**: Apply Z-score normalization
4. **Estimate Geometry**: Compute precision matrix
5. **Measure Distances**: Calculate dual-distance matrix
6. **Calibrate Locality**: Set epsilon from 20th percentile
7. **Predict**: Apply track-specific aggregation
   - Regression (learning): weighted mean
   - Structured (metacognition): dual-output parsing
   - Vector (attention): coordinate-wise weighted mean
   - Classification (executive/social): weighted vote

### Output Format Consistency

Final submission enforces **identical row order** to `sample_submission.csv`:

```R
final_submission <- data.frame(id = sample_sub$id)
final_submission <- merge(final_submission, submission_raw, 
                         by = "id", all.x = TRUE)
```

Missing predictions are imputed as `"0.0"` (edge case safety).

---

## Usage Instructions

### Prerequisites

```R
install.packages("MASS")
```

### Execution

```bash
Rscript H7_AGI_Benchmark_Predictor_EN.R
```

### Output

- **Location**: `/kaggle/working/submission.csv`
- **Format**: CSV with columns `[id, target]`
- **Validation**: Row count must match `sample_submission.csv`

### Optional: Internal Evaluation

If test labels are available at `/kaggle/input/notebooks/jakomina/attention-ipynb/submission`:

```
── Internal evaluation ──────────────────────────────────────
  learning             MAE          : XXXX.XXXXXX
  metacognition        MAE dev      : XXXX.XXXXXX  |  conf acc: XX.XX%
  attention            cosine sim   : X.XXXX
  executive_functions  exact match  : XX.XX%
  social_cognition     exact match  : XX.XX%
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **MASS** | ≥7.3 | Pseudo-inverse computation (`ginv`) |
| **base R** | ≥4.0 | Core statistics & linear algebra |

**Note**: No external ML packages (caret, tidymodels, etc.) — pure linear algebra implementation.

---

## Computational Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Covariance | **O(n·p²)** | Empirical covariance matrix |
| Precision matrix | **O(p³)** | Gaussian elimination (solve) |
| Dual distance | **O(m·n·p)** | m test × n train samples |
| VR ball | **O(m·n·log n)** | Per-test-point sorting + expansion |
| **Total** | **O(m·n·p + p³)** | Dominated by distance matrix |

For Kaggle competition scale (typically m,n ≤ 10k, p ≤ 50):
- Expected runtime: **< 5 minutes**
- Memory footprint: **≤ 1 GB**

---

## References

### H7 Framework
- **Topological Symmetry & Quantum Stability** (smokApp Lab)
- Golden ratio integration: Discrete phase accumulation via Ψₙ = cos(π·φ·n)
- Ternary state logic: {−1, 0, +1} for cognitive triads

### Vietoris-Rips Complexes
- Edelsbrunner & Harer: *Computational Topology* (2010)
- Application: Local neighborhood-based inference on point clouds

### Mahalanobis Distance
- Classical multivariate statistics
- Covariance-aware dissimilarity for non-spherical clusters

---

## Author

**Jako (Jacobo Tlacaelel Mina Rodríguez)**  
smokApp Quantum & AI Independent Research Laboratory  
Tlaxcala, México

---

## License

**H7 Framework** © smokApp Lab — Proprietary  
**Predictor Implementation** — Competition Entry

---

**Last Updated**: March 2026  
**Script Version**: 1.0 (English Translation)
# ==============================================================================
# H7 AGI Benchmark — Predictor (R) - Ultimate
# ==============================================================================

setwd("/kaggle/input/datasets/jakomina/database")
.libPaths(c("/home/runner/R/library", .libPaths()))
suppressMessages({
  library(MASS)
})

# ── H7 Constants ──────────────────────────────────────────────────────────────
PHI       <- (1 + sqrt(5)) / 2
PSI_1     <- abs(cos(pi * PHI))
DRIFT_072 <- 7 - 2 * pi

TRACK_FEATURES <- list(
  learning            = c("phi_i", "phi_j", "level_k", "n_index"),
  metacognition       = c("amplitude", "psi1"),
  attention           = c("epsilon_density", "active_trits"),
  executive_functions = c("level_k", "amplitude", "re_i"),
  social_cognition    = c("amp_a", "amp_b")
)

ALPHA      <- 0.5   
EPS_PCTILE <- 20    
EPS_EXPAND <- 1.5   
MAX_EXPAND <- 10    

# ── Funciones de Apoyo (Matemáticas y Parsers) ────────────────────────────────
# Support Functions: Mathematical Core and Parsers
std_scale <- function(X_train, X_test) {
  mu  <- colMeans(X_train)
  sd  <- apply(X_train, 2, sd) + 1e-9
  list(
    X_tr = sweep(sweep(X_train, 2, mu, "-"), 2, sd, "/"),
    X_te = sweep(sweep(X_test,  2, mu, "-"), 2, sd, "/")
  )
}

precision_mat <- function(X) {
  S <- cov(X)
  n_col <- ncol(X)
  lambda <- 1e-4 * max(diag(S))
  tryCatch({
    solve(S + diag(lambda, n_col))
  }, error = function(e) {
    ginv(S + diag(lambda, n_col))
  })
}

dual_distance <- function(X_tr, X_te, VI, alpha = ALPHA) {
  n_te <- nrow(X_te); n_tr <- nrow(X_tr)
  eucl <- matrix(0, n_te, n_tr)
  for (i in seq_len(n_te)) {
    diff     <- sweep(X_tr, 2, X_te[i, ], "-")
    eucl[i,] <- sqrt(rowSums(diff^2))
  }
  maha <- matrix(0, n_te, n_tr)
  for (i in seq_len(n_te)) {
    diff     <- sweep(X_tr, 2, X_te[i, ], "-")
    maha[i,] <- sqrt(pmax(rowSums((diff %*% VI) * diff), 0))
  }
  alpha * eucl + (1 - alpha) * maha
}

calibrate_eps <- function(D_tt, pctile = EPS_PCTILE) {
  diag(D_tt) <- NA
  as.numeric(quantile(D_tt, pctile / 100, na.rm = TRUE))
}

vr_ball <- function(d_row, eps) {
  current_eps <- eps
  for (k in seq_len(MAX_EXPAND)) {
    idx <- which(d_row <= current_eps)
    if (length(idx) > 0) {
      w <- 1 / (d_row[idx] + 1e-9)
      return(list(idx = idx, w = w / sum(w)))
    }
    current_eps <- current_eps * EPS_EXPAND
  }
  nn <- which.min(d_row)
  list(idx = nn, w = 1.0)
}

parse_meta <- function(t) {
  parts <- strsplit(trimws(t), " ")[[1]]
  dev   <- as.numeric(sub("deviation=", "", parts[1]))
  conf  <- sub("confidence=", "", parts[2])
  list(dev = dev, conf = conf)
}

parse_vec7 <- function(t) {
  cleaned <- gsub("[\\[\\]\\s]", "", t, perl = TRUE)
  vals    <- suppressWarnings(as.numeric(strsplit(cleaned, ",")[[1]]))
  vals
}

weighted_vote <- function(targets, weights) {
  tab <- tapply(weights, targets, sum)
  names(tab)[which.max(tab)]
}

# ── Predict Track ───────────────────────────────────────────────────────

predict_track <- function(track, train, test) {
  feats  <- TRACK_FEATURES[[track]]
  tr_sub <- train[train$track == track, ]
  te_sub <- test[test$track   == track, ]
  if (nrow(tr_sub) == 0 || nrow(te_sub) == 0) return(data.frame())

  tr_sub[, feats] <- lapply(tr_sub[, feats], function(x) ifelse(is.na(x), 0, as.numeric(x)))
  te_sub[, feats] <- lapply(te_sub[, feats], function(x) ifelse(is.na(x), 0, as.numeric(x)))

  scaled   <- std_scale(as.matrix(tr_sub[, feats]), as.matrix(te_sub[, feats]))
  X_tr     <- scaled$X_tr; X_te <- scaled$X_te

  VI  <- precision_mat(X_tr)
  D_tt <- dual_distance(X_tr, X_tr, VI)
  eps  <- calibrate_eps(D_tt, EPS_PCTILE)
  D    <- dual_distance(X_tr, X_te, VI)

  avg_ball <- mean(sapply(seq_len(nrow(X_te)), function(i) sum(D[i,] <= eps)))
  cat(sprintf("    epsilon=%.4f  |  average neighbors: %.1f\n", eps, avg_ball))

  preds <- character(nrow(te_sub))

  for (i in seq_len(nrow(te_sub))) {
    b <- vr_ball(D[i,], eps)
    if (track == "learning") {
      preds[i] <- sprintf("%.8f", sum(b$w * as.numeric(tr_sub$target_numeric[b$idx])))
    } else if (track == "metacognition") {
      parsed <- lapply(tr_sub$target[b$idx], parse_meta)
      pred_dev <- sum(b$w * sapply(parsed, `[[`, "dev"))
      pred_conf <- weighted_vote(sapply(parsed, `[[`, "conf"), b$w)
      preds[i] <- sprintf("deviation=%.6f confidence=%s", pred_dev, pred_conf)
    } else if (track == "attention") {
      vecs <- t(sapply(tr_sub$target[b$idx], parse_vec7))
      pred_v <- colSums(vecs * b$w)
      preds[i] <- paste0("[", paste(round(pred_v, 4), collapse = ", "), "]")
    } else {
      preds[i] <- weighted_vote(tr_sub$target[b$idx], b$w)
    }
  }
  data.frame(id = te_sub$id, target = preds, stringsAsFactors = FALSE)
}

# ── Función de Evaluación (submission) ─────────────────────────────────────────
# Evaluation Function (Submission & Metrics)
evaluate <- function(submission) {
  ans_path <- "/kaggle/input/notebooks/jakomina/attention-ipynb/submission"
  if (!file.exists(ans_path)) {
    cat("  [!] Error: Please ensure the output directory contains the required response file.\n")
    return(invisible(NULL))
  }
  
  answers <- read.csv(ans_path, stringsAsFactors = FALSE)
  # Crear 'test' para los nombres de los tracks
  path_base <- "/kaggle/input/datasets/jakomina/database/"
  test_meta <- read.csv(paste0(path_base, "test.csv"), stringsAsFactors = FALSE)
  
  m <- merge(answers, submission, by = "id", suffixes = c("_true", "_pred"))
  m <- merge(m, test_meta[, c("id", "track")], by = "id")

  cat("\n── Internal evaluation (Real Output) ──────────────────\n")

  for (tr in names(TRACK_FEATURES)) {
    sub <- m[m$track == tr, ]
    if (nrow(sub) == 0) next
    
    if (tr == "learning") {
      mae <- mean(abs(as.numeric(sub$target_true) - as.numeric(sub$target_pred)), na.rm=T)
      cat(sprintf("  %-20s MAE          : %.6f\n", tr, mae))
    } else if (tr == "attention") {
      sims <- mapply(function(t, p) {
        vt <- parse_vec7(t); vp <- parse_vec7(p)
        if (length(vt) < 7 || length(vp) < 7) return(NA)
        sum(vt * vp) / (sqrt(sum(vt^2)) * sqrt(sum(vp^2)))
      }, sub$target_true, sub$target_pred)
      cat(sprintf("  %-20s Cosine Sim   : %.4f\n", tr, mean(sims, na.rm=T)))
    } else if (tr == "metacognition") {
      dt <- sapply(sub$target_true, function(x) parse_meta(x)$dev)
      dp <- sapply(sub$target_pred, function(x) parse_meta(x)$dev)
      cat(sprintf("  %-20s MAE dev      : %.6f\n", tr, mean(abs(dt - dp), na.rm=T)))
    } else {
      acc <- mean(sub$target_true == sub$target_pred)
      cat(sprintf("  %-20s Exact Match  : %.2f%%\n", tr, 100 * acc))
    }
  }
}

# ── Main Block ───────────────────────────────────────────────────────────────

main <- function() {
  path_base <- "/kaggle/input/datasets/jakomina/database/"
  train <- read.csv(paste0(path_base, "train.csv"), stringsAsFactors = FALSE)
  test  <- read.csv(paste0(path_base, "test.csv"),  stringsAsFactors = FALSE)
  sample_sub <- read.csv(paste0(path_base, "sample_submission.csv"), stringsAsFactors = FALSE)

  cat(sprintf("Processing %d AGI Benchmark tracks...\n", length(TRACK_FEATURES)))

  all_preds <- do.call(rbind, lapply(names(TRACK_FEATURES), function(tr) {
    cat(sprintf("  Executing Track: [%s]\n", tr))
    predict_track(tr, train, test)
  }))

  final_submission <- merge(data.frame(id = sample_sub$id), all_preds, by = "id", all.x = TRUE)
  final_submission$target[is.na(final_submission$target)] <- "0.0"

  write.csv(final_submission, "/kaggle/working/submission.csv", row.names = FALSE, quote = TRUE)
  cat("\n[SUCCESS] File generated at: /kaggle/working/submission.csv\n")
  
  evaluate(final_submission)
}

main()
