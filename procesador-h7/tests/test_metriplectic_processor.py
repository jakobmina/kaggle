"""
Tests para metriplectic_processor.py
Regla 3.1: compute_lagrangian() devuelve L_symp y L_metr separados
Regla 2.1: O_n = cos(pi*n)*cos(pi*phi*n)
Regla 1.3: Sin singularidades
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import mpmath as mp
from metriplectic_processor import (
    golden_operator, MetriplecticProcessor, QuantumMetriplexOracle,
    PRECISION_DPS, PHI
)


class TestGoldenOperatorMpmath:
    def test_O_0_is_one(self):
        assert abs(float(golden_operator(0)) - 1.0) < 1e-30

    def test_O_1_value(self):
        result = golden_operator(1)
        expected = mp.cos(mp.pi) * mp.cos(mp.pi * PHI)
        assert abs(result - expected) < mp.mpf("1e-40")

    def test_bounded(self):
        for n in range(100):
            assert -1.0 <= float(golden_operator(n)) <= 1.0 + 1e-10

    def test_finite(self):
        for n in range(200):
            assert mp.isfinite(golden_operator(n))

    def test_not_periodic(self):
        vals = [round(float(golden_operator(n)), 12) for n in range(50)]
        assert len(set(vals)) > 40

    def test_precision(self):
        assert PRECISION_DPS >= 50


class TestComputeLagrangian:
    @pytest.fixture
    def proc(self):
        return MetriplecticProcessor(dps=50)

    def test_returns_two(self, proc):
        assert len(proc.compute_lagrangian(mp.mpf("0.5"), mp.mpf("1"))) == 2

    def test_l_symp(self, proc):
        psi = mp.mpf("0.7")
        L_symp, _ = proc.compute_lagrangian(psi, mp.mpf("1"))
        assert abs(L_symp - 0.5 * psi**2) < mp.mpf("1e-40")

    def test_l_metr(self, proc):
        psi, s = mp.mpf("0.5"), mp.mpf("2.0")
        _, L_metr = proc.compute_lagrangian(psi, s)
        assert abs(L_metr - s * mp.log(abs(psi))) < mp.mpf("1e-40")

    def test_psi_zero_no_crash(self, proc):
        L_symp, L_metr = proc.compute_lagrangian(mp.mpf("0"), mp.mpf("1"))
        assert mp.isfinite(L_symp) and mp.isfinite(L_metr)

    def test_l_symp_positive(self, proc):
        for v in ["-3", "0", "0.5", "100"]:
            L, _ = proc.compute_lagrangian(mp.mpf(v), mp.mpf("1"))
            assert L >= 0


class TestProcessorStep:
    @pytest.fixture
    def proc(self):
        return MetriplecticProcessor(dps=50)

    def test_returns_mpf(self, proc):
        assert isinstance(proc.step(mp.mpf("0.5"), 0), mp.mpf)

    def test_history_grows(self, proc):
        psi = mp.mpf("0.5")
        for i in range(5):
            psi = proc.step(psi, i)
        assert len(proc.history_H) == 5

    def test_result_finite(self, proc):
        psi = mp.mpf("0.5")
        for i in range(10):
            psi = proc.step(psi, i)
            assert mp.isfinite(psi)

    def test_both_forces_active(self, proc):
        psi = mp.mpf("0.5")
        for i in range(20):
            psi = proc.step(psi, i)
        assert sum(1 for v in proc.history_H if abs(v) > 1e-10) > 0
        assert sum(1 for v in proc.history_S if abs(v) > 1e-10) > 0


class TestQuantumOracle:
    @pytest.fixture
    def oracle(self):
        return QuantumMetriplexOracle()

    def test_returns_counts(self, oracle):
        counts = oracle.run_oracle()
        assert isinstance(counts, dict) and len(counts) > 0

    def test_total_shots(self, oracle):
        assert sum(oracle.run_oracle().values()) == 1024

    def test_bitstrings_4bit(self, oracle):
        for bs in oracle.run_oracle():
            assert len(bs) == 4 and all(c in '01' for c in bs)

    def test_prn_range(self, oracle):
        prn = oracle.get_prn_influence(oracle.run_oracle())
        assert 0 <= float(prn) <= 1
