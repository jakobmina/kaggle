"""
Tests para metriplectic_rnn.py
Valida: Arquitectura RNN Metripléctica, loss dual,
        constantes H7, generación de datos holográficos.

Regla 2.1: O_n = cos(πn)·cos(πφn) — modulación áurea
Regla 3.1: L_symp y L_metr separados
Regla 1.3: Prohibición de singularidades (no NaN/Inf)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import math
import torch

from metriplectic_rnn import (
    PHI, PSI_1, DRIFT_072, EPSILON,
    W_SYMP, W_METR, W_VAC,
    generate_basis, ternary_collapse,
    HolographicDataset, GoldenModulationLayer,
    MetriplecticRNN, MetriplecticLoss
)


# ==========================================================================
# CONSTANTES H7 AXIOMÁTICAS
# ==========================================================================

class TestH7Constants:
    """Las constantes H7 derivan exclusivamente de φ = (1+√5)/2."""

    def test_phi_value(self):
        expected = (1 + math.sqrt(5)) / 2
        assert abs(PHI - expected) < 1e-12

    def test_psi_1_from_phi(self):
        """Ψ₁ = |cos(πφ)| — punto fijo holográfico."""
        expected = abs(math.cos(math.pi * PHI))
        assert abs(PSI_1 - expected) < 1e-12

    def test_drift_072_value(self):
        """DRIFT_072 = 7 - 2π ≈ 0.7168."""
        expected = 7 - 2 * math.pi
        assert abs(DRIFT_072 - expected) < 1e-12
        assert 0.71 < DRIFT_072 < 0.72

    def test_epsilon_from_psi_1(self):
        """ε = Ψ₁ / 2 — threshold del vacío."""
        assert abs(EPSILON - PSI_1 / 2) < 1e-12

    def test_metriplectic_weights_sum(self):
        """W_SYMP + W_METR + W_VAC = 1.0 — normalización de pesos."""
        total = W_SYMP + W_METR + W_VAC
        assert abs(total - 1.0) < 1e-12, f"Pesos suman {total}, no 1.0"

    def test_w_symp_is_drift(self):
        """El peso conservativo es DRIFT_072 (71.7%)."""
        assert W_SYMP == DRIFT_072

    def test_w_metr_is_complement(self):
        """El peso disipativo es el complemento: 1 - DRIFT - 0.01."""
        assert abs(W_METR - (1.0 - DRIFT_072 - 0.01)) < 1e-12


# ==========================================================================
# GENERADOR DE DATOS HOLOGRÁFICOS Z7
# ==========================================================================

class TestBasisGeneration:
    """Tests para la base holográfica Z_7."""

    def test_basis_shapes(self):
        B_obj, B_ref = generate_basis(128, delta=1.0)
        assert B_obj.shape == (7, 128)
        assert B_ref.shape == (7, 128)

    def test_basis_finite(self):
        """Regla 1.3: Sin NaN/Inf en la base."""
        B_obj, B_ref = generate_basis(128)
        assert np.all(np.isfinite(B_obj))
        assert np.all(np.isfinite(B_ref))

    def test_basis_bounded(self):
        """cos() produce valores en [-1, 1]."""
        B_obj, B_ref = generate_basis(128)
        assert np.all(np.abs(B_obj) <= 1.0 + 1e-10)
        assert np.all(np.abs(B_ref) <= 1.0 + 1e-10)

    def test_basis_obj_ref_differ(self):
        """B_obj y B_ref deben diferir (δ ≠ 0 los distingue)."""
        B_obj, B_ref = generate_basis(128, delta=math.pi / 4)
        assert not np.allclose(B_obj, B_ref), "B_obj = B_ref — delta no funciona"

    def test_basis_quasi_orthogonal(self):
        """Las filas de B_obj no deben ser linealmente dependientes
        (φ es irracional → quasi-ortogonalidad)."""
        B_obj, _ = generate_basis(128)
        # Rank should be 7 (full rank)
        rank = np.linalg.matrix_rank(B_obj, tol=1e-6)
        assert rank == 7, f"Rank(B_obj)={rank}, esperado 7"


class TestTernaryCollapse:
    """Tests para el colapso ternario T ∈ {-1, 0, +1}^128."""

    def test_collapse_values(self):
        """Solo produce {-1, 0, +1}."""
        H = np.random.randn(128)
        T = ternary_collapse(H, EPSILON)
        unique_vals = set(T)
        assert unique_vals.issubset({-1.0, 0.0, 1.0})

    def test_collapse_threshold(self):
        """Valores |H| ≤ ε deben colapsar a 0 (vacío epsilon)."""
        H = np.array([0.1, -0.1, 0.5, -0.5, 0.0])
        T = ternary_collapse(H, 0.2)
        expected = np.array([0, 0, 1, -1, 0])
        np.testing.assert_array_equal(T, expected)

    def test_collapse_symmetry(self):
        """collapse(H) = -collapse(-H) — simetría ante inversión."""
        H = np.random.randn(128)
        T_pos = ternary_collapse(H, EPSILON)
        T_neg = ternary_collapse(-H, EPSILON)
        np.testing.assert_array_equal(T_pos, -T_neg)

    def test_collapse_preserves_shape(self):
        H = np.random.randn(128)
        T = ternary_collapse(H, EPSILON)
        assert T.shape == H.shape

    def test_vacuum_density_varies(self):
        """La densidad de vacío (fracción de ceros) debe existir y ser > 0
        para señales genéricas."""
        np.random.seed(42)
        x = np.random.randn(7)
        x = x / np.linalg.norm(x)
        B_obj, _ = generate_basis(128, delta=math.pi / 4)
        H = x @ B_obj
        T = ternary_collapse(H, EPSILON)
        vacuum_density = np.mean(T == 0)
        assert 0.0 < vacuum_density < 1.0, "Vacío epsilon ausente o total"


# ==========================================================================
# DATASET HOLOGRÁFICO
# ==========================================================================

class TestHolographicDataset:

    def test_dataset_length(self):
        ds = HolographicDataset(num_samples=50)
        assert len(ds) == 50

    def test_dataset_shapes(self):
        ds = HolographicDataset(num_samples=10)
        T, X_hat = ds[0]
        assert T.shape == (128,)       # Firma ternaria
        assert X_hat.shape == (7,)     # Ground truth 7D

    def test_dataset_ternary_indices(self):
        """T_idx debe contener solo {0, 1, 2} (shift de {-1,0,+1})."""
        ds = HolographicDataset(num_samples=10)
        T, _ = ds[0]
        unique_vals = set(T.numpy().tolist())
        assert unique_vals.issubset({0, 1, 2})

    def test_dataset_target_finite(self):
        """Regla 1.3: ground truth sin NaN/Inf."""
        ds = HolographicDataset(num_samples=20)
        for i in range(len(ds)):
            _, X_hat = ds[i]
            assert torch.all(torch.isfinite(X_hat)), f"Muestra {i} tiene NaN/Inf"


# ==========================================================================
# OPERADOR ÁUREO (Regla 2.1)
# ==========================================================================

class TestGoldenModulationLayer:
    """O_n = cos(πn) · cos(πφn) — el fondo estructurado del vacío."""

    def test_golden_modulation_formula(self):
        gm = GoldenModulationLayer()
        x = torch.ones(1, 10)
        for n in [0, 1, 5, 13]:
            result = gm(x, n)
            expected = math.cos(math.pi * n) * math.cos(math.pi * PHI * n)
            # Cada elemento debería ser x * O_n
            assert abs(result[0, 0].item() - expected) < 1e-6

    def test_golden_modulation_n_zero(self):
        """O_0 = cos(0)·cos(0) = 1."""
        gm = GoldenModulationLayer()
        x = torch.ones(1, 5)
        result = gm(x, 0)
        expected = 1.0
        assert abs(result[0, 0].item() - expected) < 1e-6

    def test_golden_modulation_preserves_shape(self):
        gm = GoldenModulationLayer()
        x = torch.randn(4, 64)
        result = gm(x, step_n=7)
        assert result.shape == x.shape

    def test_golden_modulation_finite(self):
        """Regla 1.3: O_n nunca es NaN/Inf."""
        gm = GoldenModulationLayer()
        x = torch.randn(2, 32)
        for n in range(100):
            result = gm(x, n)
            assert torch.all(torch.isfinite(result)), f"O_{n} produjo NaN/Inf"


# ==========================================================================
# ARQUITECTURA MetriplecticRNN
# ==========================================================================

class TestMetriplecticRNN:

    @pytest.fixture
    def model(self):
        return MetriplecticRNN()

    def test_model_output_shape(self, model):
        """Input: (batch, 128) → Output: (batch, 7)."""
        x = torch.randint(0, 3, (4, 128))
        out = model(x, step_n=1)
        assert out.shape == (4, 7)

    def test_model_output_finite(self, model):
        """Regla 1.3: La salida no debe tener NaN/Inf."""
        x = torch.randint(0, 3, (8, 128))
        out = model(x, step_n=5)
        assert torch.all(torch.isfinite(out))

    def test_model_embedding_vocab(self, model):
        """Embedding acepta vocab {0, 1, 2} (3 trits)."""
        assert model.embedding.num_embeddings == 3

    def test_model_output_features(self, model):
        """Capa final produce 7 dimensiones."""
        assert model.fc.out_features == 7

    def test_model_deterministic_same_step(self, model):
        """Misma entrada + mismo step → misma salida (en eval mode)."""
        model.eval()
        x = torch.randint(0, 3, (2, 128))
        with torch.no_grad():
            out1 = model(x, step_n=42)
            out2 = model(x, step_n=42)
        torch.testing.assert_close(out1, out2)


# ==========================================================================
# LOSS METRIPLÉCTICA (Regla 3.1)
# ==========================================================================

class TestMetriplecticLoss:
    """La loss dual: L = w_vac·L_task + w_symp·L_symp + w_metr·L_metr."""

    @pytest.fixture
    def criterion(self):
        return MetriplecticLoss()

    def test_loss_returns_five_values(self, criterion):
        pred = torch.randn(4, 7)
        target = torch.randn(4, 7)
        result = criterion(pred, target)
        assert len(result) == 5  # total, L_task, L_symp, L_metr, cos_sim

    def test_loss_finite(self, criterion):
        """Regla 1.3."""
        pred = torch.randn(8, 7)
        target = torch.randn(8, 7)
        total, L_task, L_symp, L_metr, cos_sim = criterion(pred, target)
        assert torch.isfinite(total)
        assert math.isfinite(L_task)
        assert math.isfinite(L_symp)
        assert math.isfinite(L_metr)
        assert math.isfinite(cos_sim)

    def test_loss_perfect_prediction(self, criterion):
        """Si pred == target, L_task → 0 y cos_sim → 1."""
        target = torch.randn(4, 7)
        total, L_task, L_symp, L_metr, cos_sim = criterion(target, target)
        assert L_task < 1e-5, f"L_task = {L_task} para predicción perfecta"
        assert cos_sim > 0.999

    def test_loss_symp_and_metr_separate(self, criterion):
        """L_symp y L_metr deben ser evaluables independientemente (Regla 3.1)."""
        pred = torch.randn(4, 7)
        target = torch.randn(4, 7)
        _, L_task, L_symp, L_metr, _ = criterion(pred, target)
        # Ambos deben ser no-negativos
        assert L_symp >= 0, "L_symp negativo"
        assert L_metr >= 0, "L_metr negativo"

    def test_loss_weights_applied(self):
        """Los pesos metriplécticos se aplican correctamente."""
        criterion = MetriplecticLoss(w_symp=1.0, w_metr=0.0, w_vac=0.0)
        pred = torch.randn(4, 7)
        target = torch.randn(4, 7)
        total, _, L_symp, _, _ = criterion(pred, target)
        # Con solo w_symp=1.0, total ≈ L_symp
        assert abs(total.item() - L_symp) < 1e-5

    def test_cosine_similarity_range(self, criterion):
        """Cosine similarity ∈ [-1, 1]."""
        pred = torch.randn(10, 7)
        target = torch.randn(10, 7)
        _, _, _, _, cos_sim = criterion(pred, target)
        assert -1.0 <= cos_sim <= 1.0
