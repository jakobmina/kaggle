"""
Conftest para la suite de tests del Procesador H7.
Fixtures compartidas y constantes de validación.
"""
import sys
import os
import pytest
import numpy as np
import math

# Asegurar que procesador-h7 esté en el path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# --- Constantes H7 de referencia (axiomáticas) ---
PHI = (1 + math.sqrt(5)) / 2
PSI_1 = abs(math.cos(math.pi * PHI))
DRIFT_072 = 7 - 2 * math.pi
EPSILON = PSI_1 / 2


@pytest.fixture
def phi():
    """Razón áurea."""
    return PHI


@pytest.fixture
def psi_1():
    """Punto fijo holográfico."""
    return PSI_1


@pytest.fixture
def drift():
    """DRIFT_072 = 7 - 2π."""
    return DRIFT_072


@pytest.fixture
def epsilon():
    """Threshold del vacío epsilon."""
    return EPSILON


@pytest.fixture
def random_state_7d():
    """Vector de estado normalizado en R^7."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(7)
    return x / np.linalg.norm(x)


@pytest.fixture
def random_batch_3d():
    """Batch de 10 muestras, 3 features para simon_improved."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((10, 3))
