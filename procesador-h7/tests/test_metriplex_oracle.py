"""
Tests para metriplex_oracle.py
Valida: Oráculo de Momentum, estructura de colisión 2-a-1,
        conservación H7 (XOR=7), perfiles de energía.

Regla 2 (Rigorous Analogy): validación del isomorfismo
  — colisión 2-a-1 requiere simetría real, no ajuste ad-hoc.
Regla 1.3: No singularidades en el perfil de energía.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from metriplex_oracle import (
    MetriplexConfig, MetriplexOracle, EnergyProfile, H7Conservation
)


# ==========================================================================
# CONFIGURACIÓN DEL ORÁCULO
# ==========================================================================

class TestMetriplexConfig:

    def test_default_collision_groups(self):
        config = MetriplexConfig()
        assert config.collision_groups == {'A': [1, 2, 3], 'B': [4, 5, 6]}

    def test_default_energy_profile(self):
        config = MetriplexConfig()
        assert config.energy_profile == EnergyProfile.METRIPLEX

    def test_default_momentum_range(self):
        config = MetriplexConfig()
        assert config.momentum_range == (1, 6)


# ==========================================================================
# ESTRUCTURA DE COLISIÓN 2-A-1
# ==========================================================================

class TestCollisionStructure:
    """La propiedad fundamental: f(x) = f(x ⊕ s) para s oculto."""

    @pytest.fixture
    def oracle(self):
        return MetriplexOracle()

    def test_collision_within_group_A(self, oracle):
        """Momentos en grupo A deben colisionar entre sí."""
        assert oracle.collide_pair(1, 2)
        assert oracle.collide_pair(1, 3)
        assert oracle.collide_pair(2, 3)

    def test_collision_within_group_B(self, oracle):
        """Momentos en grupo B deben colisionar entre sí."""
        assert oracle.collide_pair(4, 5)
        assert oracle.collide_pair(4, 6)
        assert oracle.collide_pair(5, 6)

    def test_no_collision_across_groups(self, oracle):
        """Momentos de grupos diferentes NO deben colisionar."""
        for a in [1, 2, 3]:
            for b in [4, 5, 6]:
                assert not oracle.collide_pair(a, b), (
                    f"p={a} (grupo A) y p={b} (grupo B) no deben colisionar"
                )

    def test_collision_is_symmetric(self, oracle):
        """collide(a,b) == collide(b,a)."""
        for p1 in range(1, 7):
            for p2 in range(1, 7):
                assert oracle.collide_pair(p1, p2) == oracle.collide_pair(p2, p1)

    def test_collision_partners_are_complete(self, oracle):
        """get_collision_partners devuelve todos los miembros del grupo."""
        partners_1 = oracle.get_collision_partners(1)
        assert sorted(partners_1) == [1, 2, 3]

        partners_5 = oracle.get_collision_partners(5)
        assert sorted(partners_5) == [4, 5, 6]


# ==========================================================================
# ORÁCULO FORWARD
# ==========================================================================

class TestOracleForward:
    """Tests para la evaluación oracle.forward(p)."""

    @pytest.fixture
    def oracle(self):
        return MetriplexOracle()

    def test_forward_returns_correct_group(self, oracle):
        for p in [1, 2, 3]:
            group, _, _ = oracle.forward(p)
            assert group == 'A'
        for p in [4, 5, 6]:
            group, _, _ = oracle.forward(p)
            assert group == 'B'

    def test_forward_output_vector_one_hot(self, oracle):
        """El vector de salida debe ser one-hot."""
        for p in range(1, 7):
            _, output, _ = oracle.forward(p)
            assert output.sum() == 1.0
            assert len(output) == 2  # 2 grupos

    def test_forward_group_A_output_vector(self, oracle):
        _, output, _ = oracle.forward(1)
        np.testing.assert_array_equal(output, [1.0, 0.0])

    def test_forward_group_B_output_vector(self, oracle):
        _, output, _ = oracle.forward(4)
        np.testing.assert_array_equal(output, [0.0, 1.0])

    def test_forward_energy_positive(self, oracle):
        """Energía normalizada debe ser positiva."""
        for p in range(1, 7):
            _, _, energy = oracle.forward(p)
            assert energy > 0, f"E({p}) = {energy} ≤ 0"

    def test_forward_energy_finite(self, oracle):
        """Regla 1.3: Sin singularidades en energía."""
        for p in range(1, 7):
            _, _, energy = oracle.forward(p)
            assert np.isfinite(energy), f"E({p}) no finita"

    def test_forward_momentum_out_of_range_raises(self, oracle):
        with pytest.raises(ValueError):
            oracle.forward(0)
        with pytest.raises(ValueError):
            oracle.forward(7)


# ==========================================================================
# CADENA DE SIMETRÍA OCULTA
# ==========================================================================

class TestSymmetryString:

    @pytest.fixture
    def oracle(self):
        return MetriplexOracle()

    def test_symmetry_string_nonzero(self, oracle):
        """s ≠ 0 (la función no es 1-a-1)."""
        s = oracle.symmetry_string()
        assert s != 0, "s = 0 implica que la función es 1-a-1, no 2-a-1"

    def test_symmetry_string_value(self, oracle):
        """Para grupos {1,2,3} y {4,5,6}, s = XOR acumulado."""
        s = oracle.symmetry_string()
        # 1^2=3, 1^3=2, 2^3=1 → OR acumulado = 3|2|1 = 3
        # 4^5=1, 4^6=2, 5^6=3 → OR acumulado |= 1|2|3 = 3
        assert s == 3


# ==========================================================================
# PERFILES DE ENERGÍA
# ==========================================================================

class TestEnergyProfiles:
    """Tests para los distintos perfiles de normalización de energía."""

    def test_linear_profile_monotonic(self):
        config = MetriplexConfig(energy_profile=EnergyProfile.LINEAR)
        oracle = MetriplexOracle(config)
        energies = [oracle.energy_map[p] for p in range(1, 7)]
        # Linear debe ser estrictamente creciente
        for i in range(len(energies) - 1):
            assert energies[i] < energies[i + 1]

    def test_quadratic_profile_monotonic(self):
        config = MetriplexConfig(energy_profile=EnergyProfile.QUADRATIC)
        oracle = MetriplexOracle(config)
        energies = [oracle.energy_map[p] for p in range(1, 7)]
        for i in range(len(energies) - 1):
            assert energies[i] < energies[i + 1]

    def test_metriplex_profile_monotonic(self):
        oracle = MetriplexOracle()
        energies = [oracle.energy_map[p] for p in range(1, 7)]
        for i in range(len(energies) - 1):
            assert energies[i] < energies[i + 1]

    def test_unknown_profile_raises(self):
        config = MetriplexConfig()
        config.energy_profile = EnergyProfile.CUSTOM
        with pytest.raises(ValueError):
            MetriplexOracle(config)


# ==========================================================================
# CONSERVACIÓN H7 (XOR = 7)
# ==========================================================================

class TestH7Conservation:
    """Tests para H7Conservation: x ↔ (7 ⊕ x)."""

    def test_partner_involution(self):
        """partner(partner(x)) = x — debe ser involución."""
        for x in range(8):
            assert H7Conservation.partner_state(
                H7Conservation.partner_state(x)
            ) == x

    def test_partner_xor_is_7(self):
        """x ⊕ partner(x) = 7."""
        for x in range(8):
            partner = H7Conservation.partner_state(x)
            assert x ^ partner == 7

    def test_pairing_table_complete(self):
        table = H7Conservation.pairing_table()
        assert len(table) == 8
        for x, partner in table.items():
            assert 0 <= x <= 7
            assert 0 <= partner <= 7

    def test_verify_pairing_correct(self):
        assert H7Conservation.verify_pairing(0, 7)
        assert H7Conservation.verify_pairing(3, 4)
        assert not H7Conservation.verify_pairing(0, 1)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            H7Conservation.partner_state(8)
        with pytest.raises(ValueError):
            H7Conservation.partner_state(-1)

    def test_conservation_invariant_valid_state(self):
        """Un estado H7-conservado tiene ambos partners activos."""
        # |0⟩ + |7⟩ (partners bajo XOR=7)
        state = np.zeros(8, dtype=complex)
        state[0] = 1 / np.sqrt(2)
        state[7] = 1 / np.sqrt(2)
        assert H7Conservation.verify_conservation_invariant(state)

    def test_conservation_invariant_invalid_state(self):
        """Un estado con |0⟩ pero sin |7⟩ viola conservación."""
        state = np.zeros(8, dtype=complex)
        state[0] = 1.0  # Solo |0⟩, sin partner |7⟩
        assert not H7Conservation.verify_conservation_invariant(state)

    def test_conservation_vacuum(self):
        """El vacío |0...0⟩ trivialmente cumple (no hay nada activo). 
        Pero en la codificación, |0⟩ es un estado → necesita partner |7⟩."""
        # Depende de la implementación: solo |0⟩ activa → necesita |7⟩
        state = np.zeros(8, dtype=complex)
        state[0] = 1.0
        # Esto viola la conservación porque |7⟩ no está activo
        assert not H7Conservation.verify_conservation_invariant(state)


# ==========================================================================
# ORACLE INFO
# ==========================================================================

class TestOracleInfo:

    def test_oracle_info_keys(self):
        oracle = MetriplexOracle()
        info = oracle.get_oracle_info()
        expected_keys = {
            'momentum_range', 'n_groups', 'collision_groups',
            'symmetry_string', 'energy_profile', 'normalization_target',
            'energy_map', 'collision_structure'
        }
        assert set(info.keys()) == expected_keys

    def test_oracle_info_consistency(self):
        oracle = MetriplexOracle()
        info = oracle.get_oracle_info()
        assert info['n_groups'] == 2
        assert info['energy_profile'] == 'metriplex'
