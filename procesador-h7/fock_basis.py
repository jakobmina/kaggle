"""
Fock Space Foundation Module
Framework: Second Quantization for Metriplex Dynamics

Core Concepts:
- Fock space as sum of occupation number subspaces: ℋ = ⊕ₙ ℋⁿ
- Truncation to n_max (configurable per hardware constraints)
- Operators: creation (a†) and annihilation (a)
- Gray-code binary mapping for Simon's algorithm compatibility

Author: Jacobo Tlacaelel Mina Rodríguez
License: MIT
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import warnings


class OccupationMode(Enum):
    """Enumeration of occupation number modes in Fock space."""
    BOSONIC = "bosonic"      # [a, a†] = 1
    FERMIONIC = "fermionic"  # {a, a†} = 1


@dataclass
class FockConfig:
    """Configuration for Fock space instantiation."""
    n_modes: int = 3          # Number of independent oscillator modes
    n_max: int = 3            # Maximum occupation per mode
    mode_type: OccupationMode = OccupationMode.BOSONIC
    use_gray_code: bool = True  # Enable Gray-code for Simon compatibility


class FockBasis:
    """
    Fock space representation with second quantization operators.
    
    The Hilbert space is the Fock space: ℋ_Fock = ⊕ₙ ℋⁿ(M)
    where ℋⁿ is the symmetric/antisymmetric n-particle subspace.
    
    In practice, we truncate to finite dimension:
    dim(ℋ_Fock) = (n_max + 1)^n_modes
    """
    
    def __init__(self, config: FockConfig = None):
        """Initialize Fock basis with given configuration."""
        if config is None:
            config = FockConfig()
        
        self.config = config
        self.n_modes = config.n_modes
        self.n_max = config.n_max
        self.dim = (config.n_max + 1) ** config.n_modes
        
        # Build occupation number basis states
        self._build_basis()
        self._precompute_operators()
    
    def _build_basis(self):
        """Build complete set of occupation number states |n₀, n₁, ..., nₘ⟩."""
        basis_states = []
        
        # Generate all occupation combinations
        ranges = [range(self.n_max + 1) for _ in range(self.n_modes)]
        for occupation in np.ndindex(*[self.n_max + 1] * self.n_modes):
            basis_states.append(occupation)
        
        self.basis_states = np.array(basis_states)
        assert len(self.basis_states) == self.dim, "Basis generation failed"
        
        # Create lookup dictionaries for fast access
        self.state_to_index = {tuple(state): i for i, state in enumerate(self.basis_states)}
        self.index_to_state = {i: tuple(state) for i, state in enumerate(self.basis_states)}
    
    def _precompute_operators(self):
        """Precompute all creation and annihilation operators in dense form."""
        self.creation_ops = {}
        self.annihilation_ops = {}
        
        for mode in range(self.n_modes):
            self.creation_ops[mode] = self._build_creation_op(mode)
            self.annihilation_ops[mode] = self._build_annihilation_op(mode)
    
    def _build_creation_op(self, mode: int) -> np.ndarray:
        """
        Build creation operator a†_mode in Fock basis.
        
        Action: a†_mode |n₀, ..., nₘ, ...⟩ = sqrt(nₘ+1) |n₀, ..., nₘ+1, ...⟩
        """
        op = np.zeros((self.dim, self.dim), dtype=complex)
        
        for i, state in enumerate(self.basis_states):
            state_list = list(state)
            
            # Check if we can apply creation (occupation < n_max)
            if state_list[mode] < self.n_max:
                new_occupation = state_list[mode] + 1
                amplitude = np.sqrt(new_occupation)
                
                # Target state
                state_list[mode] = new_occupation
                j = self.state_to_index[tuple(state_list)]
                op[j, i] = amplitude
        
        return op
    
    def _build_annihilation_op(self, mode: int) -> np.ndarray:
        """
        Build annihilation operator a_mode in Fock basis.
        
        Action: a_mode |n₀, ..., nₘ, ...⟩ = sqrt(nₘ) |n₀, ..., nₘ-1, ...⟩
        """
        op = np.zeros((self.dim, self.dim), dtype=complex)
        
        for i, state in enumerate(self.basis_states):
            state_list = list(state)
            
            # Check if we can apply annihilation (occupation > 0)
            if state_list[mode] > 0:
                old_occupation = state_list[mode]
                amplitude = np.sqrt(old_occupation)
                
                # Target state
                state_list[mode] = old_occupation - 1
                j = self.state_to_index[tuple(state_list)]
                op[j, i] = amplitude
        
        return op
    
    def get_creation_op(self, mode: int) -> np.ndarray:
        """Retrieve precomputed creation operator for given mode."""
        if mode not in self.creation_ops:
            raise ValueError(f"Mode {mode} out of range [0, {self.n_modes-1}]")
        return self.creation_ops[mode].copy()
    
    def get_annihilation_op(self, mode: int) -> np.ndarray:
        """Retrieve precomputed annihilation operator for given mode."""
        if mode not in self.annihilation_ops:
            raise ValueError(f"Mode {mode} out of range [0, {self.n_modes-1}]")
        return self.annihilation_ops[mode].copy()
    
    def number_operator(self, mode: int) -> np.ndarray:
        """
        Number operator N_mode = a†_mode * a_mode
        
        Eigenvalues = occupation numbers; eigenstate |n⟩ has eigenvalue n
        """
        a_dag = self.get_creation_op(mode)
        a = self.get_annihilation_op(mode)
        return a_dag @ a
    
    def total_number_operator(self) -> np.ndarray:
        """Total number operator N_total = Σₘ N_mode(m)."""
        N_total = np.zeros((self.dim, self.dim), dtype=complex)
        for mode in range(self.n_modes):
            N_total += self.number_operator(mode)
        return N_total
    
    def state_vector(self, occupation: Tuple[int, ...]) -> np.ndarray:
        """Return state vector |n₀, n₁, ..., nₘ⟩ in computational basis."""
        if len(occupation) != self.n_modes:
            raise ValueError(f"Occupation must have {self.n_modes} elements")
        
        if any(n > self.n_max for n in occupation):
            raise ValueError(f"Occupation exceeds n_max = {self.n_max}")
        
        vec = np.zeros(self.dim, dtype=complex)
        idx = self.state_to_index[tuple(occupation)]
        vec[idx] = 1.0
        return vec
    
    def to_gray_code(self, state_index: int) -> int:
        """
        Convert state index to Gray code binary representation.
        
        This preserves Hamming distance structure for Simon's algorithm.
        Gray code: adjacent integers differ by exactly one bit.
        """
        return state_index ^ (state_index >> 1)
    
    def from_gray_code(self, gray: int) -> int:
        """Convert Gray code back to binary index."""
        mask = gray
        while mask:
            mask >>= 1
            gray ^= mask
        return gray
    
    def occupation_to_binary(self, occupation: Tuple[int, ...]) -> str:
        """Convert occupation tuple to binary string representation."""
        state_index = self.state_to_index[tuple(occupation)]
        if self.config.use_gray_code:
            binary_rep = self.to_gray_code(state_index)
        else:
            binary_rep = state_index
        return format(binary_rep, f'0{self.n_modes * 2}b')
    
    def get_basis_info(self) -> Dict:
        """Return comprehensive basis information."""
        return {
            'dimension': self.dim,
            'n_modes': self.n_modes,
            'n_max_per_mode': self.n_max,
            'total_occupation_max': self.n_max * self.n_modes,
            'basis_states': self.basis_states.tolist(),
            'use_gray_code': self.config.use_gray_code,
            'mode_type': self.config.mode_type.value
        }


class FockStateVector:
    """
    Quantum state in Fock basis with helper methods.
    """
    
    def __init__(self, fock_basis: FockBasis, vector: np.ndarray = None):
        """Initialize state vector in Fock basis."""
        self.fock = fock_basis
        if vector is None:
            self.vec = np.zeros(fock_basis.dim, dtype=complex)
        else:
            if len(vector) != fock_basis.dim:
                raise ValueError(f"Vector dimension {len(vector)} != {fock_basis.dim}")
            self.vec = vector.astype(complex)
    
    def normalize(self):
        """Normalize state vector to unit norm."""
        norm = np.linalg.norm(self.vec)
        if norm > 1e-10:
            self.vec /= norm
        return self
    
    def occupation_probabilities(self) -> Dict[Tuple[int, ...], float]:
        """Return probability distribution over occupation numbers."""
        probs = {}
        for i, state in enumerate(self.fock.basis_states):
            prob = abs(self.vec[i])**2
            if prob > 1e-10:
                probs[tuple(state)] = prob
        return probs
    
    def expectation_value(self, operator: np.ndarray) -> complex:
        """Compute ⟨ψ|O|ψ⟩."""
        return self.vec @ operator @ self.vec
    
    def fidelity(self, other: 'FockStateVector') -> float:
        """Compute fidelity |⟨ψ|φ⟩|² with another state."""
        overlap = np.abs(np.vdot(self.vec, other.vec))**2
        return np.clip(overlap, 0, 1)


# Convenience constructors
def fock_ground_state(fock_basis: FockBasis) -> FockStateVector:
    """Return ground state |0, 0, ..., 0⟩."""
    occupation = tuple(0 for _ in range(fock_basis.n_modes))
    vec = fock_basis.state_vector(occupation)
    return FockStateVector(fock_basis, vec)


def fock_single_photon(fock_basis: FockBasis, mode: int) -> FockStateVector:
    """Return single photon in given mode |0, ..., 1_m, ..., 0⟩."""
    occupation = [0] * fock_basis.n_modes
    occupation[mode] = 1
    vec = fock_basis.state_vector(tuple(occupation))
    return FockStateVector(fock_basis, vec)


if __name__ == "__main__":
    # Example usage
    config = FockConfig(n_modes=3, n_max=2)
    fock = FockBasis(config)
    
    print("Fock Space Basis Information:")
    print(f"  Dimension: {fock.dim}")
    print(f"  Modes: {fock.n_modes}")
    print(f"  Max occupation per mode: {fock.n_max}")
    print(f"\nFirst 5 basis states:")
    for i in range(min(5, fock.dim)):
        state = fock.index_to_state[i]
        print(f"  |{state}⟩ (index {i})")
    
    # Test operators
    print(f"\nCreation operator a†₀ diagonal check (should have zeros):")
    a_dag_0 = fock.get_creation_op(0)
    print(f"  Non-zero diagonal elements: {np.count_nonzero(np.diag(a_dag_0))}")
    
    # Test state vector
    gs = fock_ground_state(fock)
    print(f"\nGround state |0,0,0⟩:")
    print(f"  Norm: {np.linalg.norm(gs.vec):.6f}")
    print(f"  Number operator expectation: {gs.expectation_value(fock.total_number_operator()):.6f}")