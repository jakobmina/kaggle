"""
Tests para simon_improved.py
Valida: Procesador ternario H7, codificación {-1,0,+1},
        detección de estructura, capa neural, pares H7.
Regla 2.1: Modulación phi activa
Regla 1.3: Sin singularidades
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from simon_improved import (
    H7ProcessorConfig, H7TernaryProcessor, H7TernaryLayer
)


class TestH7ProcessorConfig:
    def test_defaults(self):
        c = H7ProcessorConfig()
        assert c.n_features == 3
        assert c.h7_structure == 7
        assert c.ternary_encoding == "balanced"
        assert c.phi_modulation is True


class TestTernaryEncoding:
    @pytest.fixture
    def proc(self):
        return H7TernaryProcessor()

    def test_balanced_values(self, proc):
        x = np.array([[-1.0, 0.0, 1.0]])
        t = proc.encode_to_ternary(x)
        unique = set(t.flatten())
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_balanced_thresholds(self, proc):
        x = np.array([[0.5, -0.5, 0.1]])
        t = proc.encode_to_ternary(x)
        np.testing.assert_array_equal(t, [[1, -1, 0]])

    def test_standard_encoding(self):
        cfg = H7ProcessorConfig(ternary_encoding="standard")
        proc = H7TernaryProcessor(cfg)
        x = np.array([[0.1, 0.5, 0.9]])
        t = proc.encode_to_ternary(x)
        np.testing.assert_array_equal(t, [[0, 1, 2]])

    def test_decode_roundtrip_balanced(self, proc):
        t = np.array([[-1, 0, 1]], dtype=float)
        decoded = proc.decode_from_ternary(t)
        np.testing.assert_array_equal(decoded, t)

    def test_unknown_encoding_raises(self):
        cfg = H7ProcessorConfig(ternary_encoding="unknown")
        proc = H7TernaryProcessor(cfg)
        with pytest.raises(ValueError):
            proc.encode_to_ternary(np.array([[0.5]]))

    def test_preserves_shape(self, proc):
        x = np.random.randn(5, 3)
        t = proc.encode_to_ternary(x)
        assert t.shape == x.shape


class TestH7Pairs:
    def test_complementary_pairs(self):
        proc = H7TernaryProcessor()
        assert proc.h7_pairs == [(1, 6), (2, 5), (3, 4)]

    def test_pairs_sum_to_7(self):
        proc = H7TernaryProcessor()
        for a, b in proc.h7_pairs:
            assert a + b == 7


class TestStructureDetection:
    @pytest.fixture
    def proc(self):
        return H7TernaryProcessor()

    def test_detect_1d_input(self, proc):
        x = np.array([0.5, -0.2, 0.8])
        result = proc.detect_structure(x)
        assert result['ternary_encoded'] is not None
        assert result['h7_embeddings'] is not None
        assert result['structure_scores'] is not None

    def test_detect_batch(self, proc):
        x = np.random.randn(10, 3)
        result = proc.detect_structure(x)
        assert result['h7_embeddings'].shape == (10, 8)
        assert result['complementary_activations'].shape == (10, 3)
        assert result['structure_scores'].shape == (10,)

    def test_embeddings_normalized(self, proc):
        x = np.random.randn(5, 3)
        result = proc.detect_structure(x)
        norms = np.linalg.norm(result['h7_embeddings'], axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_embeddings_finite(self, proc):
        x = np.random.randn(8, 3)
        result = proc.detect_structure(x)
        assert np.all(np.isfinite(result['h7_embeddings']))
        assert np.all(np.isfinite(result['structure_scores']))

    def test_n_processed_counter(self, proc):
        x = np.random.randn(7, 3)
        proc.detect_structure(x)
        assert proc.n_processed == 7
        proc.detect_structure(x)
        assert proc.n_processed == 14


class TestForwardPass:
    @pytest.fixture
    def proc(self):
        return H7TernaryProcessor()

    def test_output_shape(self, proc):
        x = np.random.randn(5, 3)
        out = proc.forward(x)
        # 8 (embeddings) + 3 (pairs) + 1 (score) = 12
        assert out.shape == (5, 12)

    def test_output_finite(self, proc):
        x = np.random.randn(10, 3)
        out = proc.forward(x)
        assert np.all(np.isfinite(out))


class TestH7TernaryLayer:
    def test_layer_output_dim(self):
        layer = H7TernaryLayer(n_features=3)
        assert layer.get_output_dim() == 12

    def test_layer_callable(self):
        layer = H7TernaryLayer(n_features=3)
        x = np.random.randn(4, 3)
        out = layer(x)
        assert out.shape == (4, 12)

    def test_layer_finite(self):
        layer = H7TernaryLayer(n_features=3)
        x = np.random.randn(8, 3)
        out = layer(x)
        assert np.all(np.isfinite(out))


class TestPhiModulation:
    def test_modulation_active(self):
        cfg = H7ProcessorConfig(phi_modulation=True)
        proc = H7TernaryProcessor(cfg)
        x = np.random.randn(3, 3)
        result = proc.detect_structure(x)
        emb = result['h7_embeddings']
        # With phi modulation, not all 8 components equal
        for row in emb:
            assert not np.allclose(row, row[0])

    def test_modulation_off_uniform(self):
        cfg = H7ProcessorConfig(phi_modulation=False)
        proc = H7TernaryProcessor(cfg)
        x = np.random.randn(3, 3)
        result = proc.detect_structure(x)
        emb = result['h7_embeddings']
        # Without modulation, initial superposition is uniform
        # after normalization all components equal
        for row in emb:
            np.testing.assert_allclose(
                row, np.ones(8) / np.sqrt(8), atol=1e-6
            )
