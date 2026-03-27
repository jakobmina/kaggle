"""
Tests para fock_basis.py
Valida: Espacio de Fock, operadores de creación/aniquilación,
        relaciones de conmutación, estados y Gray code.

Regla 3.1: compute_lagrangian() separado (L_symp, L_metr)
Regla 1.3: Prohibición de singularidades (norma nunca 0 ni ∞)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from fock_basis import (
    FockConfig, FockBasis, FockStateVector,
    OccupationMode, fock_ground_state, fock_single_photon
)


# ==========================================================================
# CONFIGURACIÓN Y CONSTRUCCIÓN DE BASE
# ==========================================================================

class TestFockConfig:
    """Tests para FockConfig dataclass."""

    def test_default_config(self):
        config = FockConfig()
        assert config.n_modes == 3
        assert config.n_max == 3
        assert config.mode_type == OccupationMode.BOSONIC
        assert config.use_gray_code is True

    def test_custom_config(self):
        config = FockConfig(n_modes=2, n_max=5, use_gray_code=False)
        assert config.n_modes == 2
        assert config.n_max == 5
        assert config.use_gray_code is False


class TestFockBasis:
    """Tests para la base del espacio de Fock."""

    @pytest.fixture
    def fock_3_2(self):
        """Fock basis: 3 modos, n_max=2 → dim = 3^3 = 27."""
        return FockBasis(FockConfig(n_modes=3, n_max=2))

    @pytest.fixture
    def fock_2_3(self):
        """Fock basis: 2 modos, n_max=3 → dim = 4^2 = 16."""
        return FockBasis(FockConfig(n_modes=2, n_max=3))

    def test_dimension_formula(self, fock_3_2):
        """dim(ℋ_Fock) = (n_max + 1)^n_modes."""
        assert fock_3_2.dim == (2 + 1) ** 3  # 27

    def test_dimension_2_modes(self, fock_2_3):
        assert fock_2_3.dim == (3 + 1) ** 2  # 16

    def test_basis_states_count(self, fock_3_2):
        """Number of basis states equals dimension."""
        assert len(fock_3_2.basis_states) == fock_3_2.dim

    def test_vacuum_state_exists(self, fock_3_2):
        """El estado de vacío |0,0,0⟩ debe existir."""
        vacuum = (0, 0, 0)
        assert vacuum in fock_3_2.state_to_index

    def test_max_occupation_state_exists(self, fock_3_2):
        """El estado de máxima ocupación |n_max, n_max, n_max⟩ debe existir."""
        max_state = (2, 2, 2)
        assert max_state in fock_3_2.state_to_index

    def test_state_to_index_invertible(self, fock_3_2):
        """Mappeo estado→índice debe ser biyectivo."""
        for idx, state in fock_3_2.index_to_state.items():
            assert fock_3_2.state_to_index[state] == idx

    def test_basis_info_complete(self, fock_3_2):
        info = fock_3_2.get_basis_info()
        assert info['dimension'] == 27
        assert info['n_modes'] == 3
        assert info['n_max_per_mode'] == 2
        assert info['mode_type'] == 'bosonic'


# ==========================================================================
# OPERADORES DE CREACIÓN Y ANIQUILACIÓN
# ==========================================================================

class TestLadderOperators:
    """Tests para a† y a — la columna vertebral de la 2da cuantización."""

    @pytest.fixture
    def fock(self):
        return FockBasis(FockConfig(n_modes=2, n_max=3))

    def test_creation_op_shape(self, fock):
        a_dag = fock.get_creation_op(0)
        assert a_dag.shape == (fock.dim, fock.dim)

    def test_annihilation_op_shape(self, fock):
        a = fock.get_annihilation_op(0)
        assert a.shape == (fock.dim, fock.dim)

    def test_creation_annihilation_adjoint(self, fock):
        """a† = (a)^† — deben ser adjuntos conjugados."""
        for mode in range(fock.n_modes):
            a_dag = fock.get_creation_op(mode)
            a = fock.get_annihilation_op(mode)
            # a† should be the conjugate transpose of a
            np.testing.assert_array_almost_equal(
                a_dag, a.conj().T,
                err_msg=f"a†_{mode} ≠ (a_{mode})† — violación de hermiticidad"
            )

    def test_commutation_relation_bosonic(self, fock):
        """[a, a†] ≈ I para modo bosónico en espacio de Fock TRUNCADO.
        
        En ℋ infinita, [a, a†] = I exactamente.
        En ℋ truncada (n_max finito), [a, a†] = I excepto en los estados
        con n_mode = n_max, donde a†|n_max⟩ = 0 causa una desviación.
        Esto es un artefacto conocido de la truncación, no un bug.
        
        Verificamos que el conmutador sea diagonal y que los elementos
        donde n < n_max sean exactamente 1.
        """
        for mode in range(fock.n_modes):
            a = fock.get_annihilation_op(mode)
            a_dag = fock.get_creation_op(mode)
            commutator = a @ a_dag - a_dag @ a
            
            # El conmutador debe ser diagonal
            off_diag = commutator - np.diag(np.diag(commutator))
            np.testing.assert_array_almost_equal(
                off_diag, 0, decimal=10,
                err_msg=f"[a_{mode}, a†_{mode}] tiene elementos off-diagonal"
            )
            
            # Los estados con n_mode < n_max deben tener eigenvalor 1
            for idx, state in enumerate(fock.basis_states):
                if state[mode] < fock.n_max:
                    assert abs(commutator[idx, idx] - 1.0) < 1e-10, (
                        f"[a,a†]|{tuple(state)}⟩ ≠ 1 (n_{mode}={state[mode]} < n_max)"
                    )

    def test_annihilation_kills_vacuum(self, fock):
        """a|0⟩ = 0 — definición del vacío."""
        vacuum = fock.state_vector((0, 0))
        for mode in range(fock.n_modes):
            a = fock.get_annihilation_op(mode)
            result = a @ vacuum
            norm = np.linalg.norm(result)
            assert norm < 1e-10, f"a_{mode}|vacuum⟩ ≠ 0, norma = {norm}"

    def test_creation_on_vacuum_gives_single_photon(self, fock):
        """a†|0,0⟩ = |1,0⟩ (con amplitud √1 = 1)."""
        vacuum = fock.state_vector((0, 0))
        a_dag_0 = fock.get_creation_op(0)
        result = a_dag_0 @ vacuum

        expected = fock.state_vector((1, 0))
        np.testing.assert_array_almost_equal(result, expected)

    def test_number_operator_eigenvalues(self, fock):
        """N|n⟩ = n|n⟩ — los eigenvalores del operador numérico son las ocupaciones."""
        N_0 = fock.number_operator(0)
        for n in range(fock.n_max + 1):
            state = fock.state_vector((n, 0))
            result = N_0 @ state
            expected = n * state
            np.testing.assert_array_almost_equal(
                result, expected,
                err_msg=f"N_0|{n},0⟩ ≠ {n}|{n},0⟩"
            )

    def test_total_number_operator(self, fock):
        """N_total |n₀, n₁⟩ = (n₀ + n₁)|n₀, n₁⟩."""
        N_total = fock.total_number_operator()
        state = fock.state_vector((2, 1))
        result = N_total @ state
        expected = 3.0 * state  # n₀ + n₁ = 2 + 1 = 3
        np.testing.assert_array_almost_equal(result, expected)


# ==========================================================================
# ESTADOS CUÁNTICOS EN FOCK
# ==========================================================================

class TestFockStateVector:
    """Tests para FockStateVector."""

    @pytest.fixture
    def fock(self):
        return FockBasis(FockConfig(n_modes=3, n_max=2))

    def test_ground_state_normalized(self, fock):
        gs = fock_ground_state(fock)
        norm = np.linalg.norm(gs.vec)
        assert abs(norm - 1.0) < 1e-10, "Estado base no normalizado"

    def test_single_photon_normalized(self, fock):
        sp = fock_single_photon(fock, mode=1)
        norm = np.linalg.norm(sp.vec)
        assert abs(norm - 1.0) < 1e-10

    def test_single_photon_correct_occupation(self, fock):
        """|0,1,0⟩ tiene probabilidad 1 en ocupación (0,1,0)."""
        sp = fock_single_photon(fock, mode=1)
        probs = sp.occupation_probabilities()
        assert (0, 1, 0) in probs
        assert abs(probs[(0, 1, 0)] - 1.0) < 1e-10

    def test_fidelity_self_is_one(self, fock):
        gs = fock_ground_state(fock)
        assert abs(gs.fidelity(gs) - 1.0) < 1e-10

    def test_fidelity_orthogonal_is_zero(self, fock):
        gs = fock_ground_state(fock)
        sp = fock_single_photon(fock, mode=0)
        assert gs.fidelity(sp) < 1e-10

    def test_normalize_idempotent(self, fock):
        """Normalizar un estado ya normalizado no lo cambia."""
        gs = fock_ground_state(fock)
        vec_before = gs.vec.copy()
        gs.normalize()
        np.testing.assert_array_almost_equal(gs.vec, vec_before)

    def test_expectation_value_number_in_vacuum(self, fock):
        """⟨0|N|0⟩ = 0 — el vacío no tiene partículas."""
        gs = fock_ground_state(fock)
        N_total = fock.total_number_operator()
        expectation = gs.expectation_value(N_total)
        assert abs(expectation) < 1e-10

    def test_state_vector_invalid_occupation_raises(self, fock):
        """Ocupación fuera de rango debe lanzar ValueError."""
        with pytest.raises(ValueError):
            fock.state_vector((5, 0, 0))  # 5 > n_max=2

    def test_state_vector_wrong_modes_raises(self, fock):
        with pytest.raises(ValueError):
            fock.state_vector((0, 0))  # Solo 2 modos, necesita 3


# ==========================================================================
# GRAY CODE
# ==========================================================================

class TestGrayCode:
    """Tests para la codificación Gray (compatibilidad con Simon)."""

    @pytest.fixture
    def fock(self):
        return FockBasis(FockConfig(n_modes=3, n_max=2))

    def test_gray_code_roundtrip(self, fock):
        """Gray → Binary → Gray debe ser identidad."""
        for i in range(fock.dim):
            gray = fock.to_gray_code(i)
            back = fock.from_gray_code(gray)
            assert back == i, f"Roundtrip failed: {i} → {gray} → {back}"

    def test_gray_code_hamming_distance(self, fock):
        """Gray codes consecutivos difieren en exactamente 1 bit."""
        for i in range(fock.dim - 1):
            g1 = fock.to_gray_code(i)
            g2 = fock.to_gray_code(i + 1)
            xor = g1 ^ g2
            hamming = bin(xor).count('1')
            assert hamming == 1, (
                f"Gray({i})={g1:#b} y Gray({i+1})={g2:#b} "
                f"difieren en {hamming} bits (debe ser 1)"
            )


# ==========================================================================
# REGLA 1.3: PROHIBICIÓN DE SINGULARIDADES
# ==========================================================================

class TestNoSingularities:
    """Regla 1.3: Ningún operador debe producir norma 0 ni ∞ en estados válidos."""

    @pytest.fixture
    def fock(self):
        return FockBasis(FockConfig(n_modes=2, n_max=3))

    def test_creation_never_produces_infinite_norm(self, fock):
        for mode in range(fock.n_modes):
            a_dag = fock.get_creation_op(mode)
            for i in range(fock.dim):
                vec = np.zeros(fock.dim, dtype=complex)
                vec[i] = 1.0
                result = a_dag @ vec
                norm = np.linalg.norm(result)
                assert np.isfinite(norm), f"Norma infinita en a†_{mode}|{i}⟩"

    def test_operator_matrix_finite(self, fock):
        for mode in range(fock.n_modes):
            a_dag = fock.get_creation_op(mode)
            a = fock.get_annihilation_op(mode)
            assert np.all(np.isfinite(a_dag)), "a† contiene NaN/Inf"
            assert np.all(np.isfinite(a)), "a contiene NaN/Inf"
