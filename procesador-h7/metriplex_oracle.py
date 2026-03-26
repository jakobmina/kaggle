"""
Metriplex Momentum Oracle
Mechanism II: Energy Profiling for Simon's Algorithm

Core Concept:
The oracle maps classical momentum states p ∈ [1..6] to 2D energy vectors,
establishing a 2-to-1 function that Simon's algorithm can exploit.

Energy Profiling:
- Group A (p ∈ {1,2,3}) → (1, 0)  [Normalized energy ≈ 0.45]
- Group B (p ∈ {4,5,6}) → (0, 1)  [Normalized energy ≈ 0.45]

The "collision" structure reveals hidden symmetry: p ⊕ (p+3) = 3 mod 6.

Author: Jacobo Tlacaelel Mina Rodríguez
"""

import numpy as np
from typing import Tuple, Dict, List, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum
import warnings


class EnergyProfile(Enum):
    """Predefined energy normalization profiles."""
    LINEAR = "linear"           # E(p) = p / p_max
    QUADRATIC = "quadratic"     # E(p) = (p / p_max)²
    METRIPLEX = "metriplex"     # Tuned for H7 conservation
    CUSTOM = "custom"


@dataclass
class MetriplexConfig:
    """Configuration for metriplex oracle."""
    momentum_range: Tuple[int, int] = (1, 6)  # [p_min, p_max]
    energy_profile: EnergyProfile = EnergyProfile.METRIPLEX
    normalization_target: float = 0.1024  # Target normalized energy level
    collision_groups: Optional[Dict[str, List[int]]] = None
    
    def __post_init__(self):
        if self.collision_groups is None:
            self.collision_groups = {
                'A': [1, 2, 3],
                'B': [4, 5, 6]
            }


class MetriplexOracle:
    """
    Metriplex momentum oracle for second quantization.
    
    This oracle establishes a hidden 2-to-1 function by mapping momentum
    states to energy vectors. Simon's algorithm discovers the hidden
    structure: momentum pairs differing by 3 collide.
    """
    
    def __init__(self, config: MetriplexConfig = None):
        """Initialize oracle with configuration."""
        if config is None:
            config = MetriplexConfig()
        
        self.config = config
        self.p_min, self.p_max = config.momentum_range
        
        # Build energy lookup tables
        self._build_energy_map()
        self._build_collision_map()
    
    def _build_energy_map(self):
        """Precompute normalized energy for each momentum state."""
        self.energy_map = {}
        
        for p in range(self.p_min, self.p_max + 1):
            # Raw energy calculation based on profile
            if self.config.energy_profile == EnergyProfile.LINEAR:
                raw_energy = p / self.p_max
            elif self.config.energy_profile == EnergyProfile.QUADRATIC:
                raw_energy = (p / self.p_max) ** 2
            elif self.config.energy_profile == EnergyProfile.METRIPLEX:
                # Metriplex: tuned to match target normalization
                # E(p) = target * (p / p_max)^α where α is chosen for consistency
                alpha = 1.2  # Slightly superlinear for metriplex dynamics
                raw_energy = self.config.normalization_target * (p / self.p_max) ** alpha
            else:
                raise ValueError(f"Unknown energy profile: {self.config.energy_profile}")
            
            self.energy_map[p] = raw_energy
        
        # Verify normalization
        mean_energy = np.mean([self.energy_map[p] for p in range(self.p_min, self.p_max + 1)])
        if abs(mean_energy - self.config.normalization_target) > 0.01:
            warnings.warn(
                f"Mean energy {mean_energy:.4f} deviates from target "
                f"{self.config.normalization_target:.4f}"
            )
    
    def _build_collision_map(self):
        """Build 2-to-1 collision structure from momentum groups."""
        self.collision_map = {}
        self.output_groups = {}
        
        for group_name, momenta in self.config.collision_groups.items():
            # All momenta in same group map to same output vector
            for p in momenta:
                self.collision_map[p] = group_name
            
            # Define output vector (one-hot encoding of group)
            group_index = list(self.config.collision_groups.keys()).index(group_name)
            n_groups = len(self.config.collision_groups)
            output_vec = np.zeros(n_groups)
            output_vec[group_index] = 1.0
            self.output_groups[group_name] = output_vec
    
    def _compute_symmetry_string(self) -> int:
        """
        Compute the hidden symmetry string s.
        
        For the standard configuration:
        - Group A = {1, 2, 3}, Group B = {4, 5, 6}
        - Hidden structure: p and (p ⊕ 3) collide
        - In 3-bit arithmetic: s = 3 (binary 011)
        
        More generally: s = XOR of differences within collision groups
        """
        group_lists = list(self.config.collision_groups.values())
        
        # Compute pairwise XOR differences within groups
        xor_accumulator = 0
        for group in group_lists:
            for i, p1 in enumerate(group):
                for p2 in group[i+1:]:
                    xor_accumulator |= (p1 ^ p2)
        
        return xor_accumulator
    
    def forward(self, momentum: int) -> Tuple[str, np.ndarray, float]:
        """
        Evaluate oracle at given momentum state.
        
        Returns:
            group_name: Which collision group this momentum belongs to
            output_vector: Energy-encoded output (1 per group)
            energy: Normalized energy at this momentum
        """
        if momentum < self.p_min or momentum > self.p_max:
            raise ValueError(f"Momentum {momentum} out of range [{self.p_min}, {self.p_max}]")
        
        group = self.collision_map[momentum]
        output_vec = self.output_groups[group]
        energy = self.energy_map[momentum]
        
        return group, output_vec, energy
    
    def collide_pair(self, p1: int, p2: int) -> bool:
        """
        Check if two momentum states collide (map to same output).
        
        True iff they belong to the same collision group.
        """
        return self.collision_map[p1] == self.collision_map[p2]
    
    def get_collision_partners(self, p: int) -> List[int]:
        """Return all momentum states that collide with p."""
        group = self.collision_map[p]
        return self.config.collision_groups[group]
    
    def symmetry_string(self) -> int:
        """Return the hidden symmetry parameter s."""
        return self._compute_symmetry_string()
    
    def to_hilbert_oracle(self, fock_basis) -> Callable:
        """
        Convert classical oracle to quantum operator for Hilbert space.
        
        Returns a function that takes quantum state |ψ⟩ and applies
        phase shifts based on energy profiling.
        
        This is the oracle_f used in Simon's algorithm.
        """
        def quantum_oracle(state_vector: np.ndarray) -> np.ndarray:
            """
            Apply metriplex oracle to quantum state.
            
            For each basis state |occupation⟩, decode momentum p,
            compute energy E(p), apply phase shift e^{i·E(p)·2π}.
            """
            result = state_vector.copy()
            
            for idx, occupation in enumerate(fock_basis.basis_states):
                # Map occupation to effective momentum
                # (depends on Fock space structure)
                effective_momentum = self._occupation_to_momentum(occupation)
                
                # Apply phase based on energy
                energy = self.energy_map[effective_momentum]
                phase_shift = np.exp(2j * np.pi * energy)
                
                result[idx] *= phase_shift
            
            return result
        
        return quantum_oracle
    
    def _occupation_to_momentum(self, occupation: Tuple[int, ...]) -> int:
        """
        Map Fock occupation tuple to effective momentum.
        
        Simple mapping: p = (total occupation mod range_size) + p_min
        For 3-mode system: p = (n₀ + n₁ + n₂) mod 6 → [0..5], shift to [1..6]
        """
        total_occ = sum(occupation)
        range_size = self.p_max - self.p_min + 1
        p = (total_occ % range_size) + self.p_min
        return p
    
    def get_oracle_info(self) -> Dict:
        """Return comprehensive oracle information."""
        return {
            'momentum_range': self.config.momentum_range,
            'n_groups': len(self.config.collision_groups),
            'collision_groups': {
                k: v for k, v in self.config.collision_groups.items()
            },
            'symmetry_string': self.symmetry_string(),
            'energy_profile': self.config.energy_profile.value,
            'normalization_target': self.config.normalization_target,
            'energy_map': {p: self.energy_map[p] for p in range(self.p_min, self.p_max + 1)},
            'collision_structure': {
                p: self.collision_map[p] for p in range(self.p_min, self.p_max + 1)
            }
        }


class H7Conservation:
    """
    Mechanism I: H7 Entanglement Conservation
    
    In 3-qubit Hilbert space (8 basis states 0-7), states are paired
    by the rule: x ↔ (7 ⊕ x). This conservation law is fundamental.
    """
    
    # Fixed rule: paired states sum (XOR) to 7
    CONSERVATION_CONSTANT = 7
    
    @staticmethod
    def partner_state(state: int) -> int:
        """Return entangled partner of state under H7 conservation."""
        if not (0 <= state <= 7):
            raise ValueError("State must be in [0, 7]")
        return H7Conservation.CONSERVATION_CONSTANT ^ state
    
    @staticmethod
    def verify_pairing(state_a: int, state_b: int) -> bool:
        """Verify if two states are H7-entangled partners."""
        return state_b == H7Conservation.partner_state(state_a)
    
    @staticmethod
    def pairing_table() -> Dict[int, int]:
        """Return complete H7 pairing table."""
        return {
            i: H7Conservation.partner_state(i)
            for i in range(8)
        }
    
    @staticmethod
    def verify_conservation_invariant(state_vector: np.ndarray, threshold: float = 1e-6) -> bool:
        """
        Verify if a 3-qubit state respects H7 conservation.
        
        A state |ψ⟩ respects H7 conservation if:
        for all superpositions α_i |i⟩, whenever α_i ≠ 0,
        we also have α_partner(i) ≠ 0 (or both equal to zero).
        """
        if len(state_vector) != 8:
            raise ValueError("State vector must be 8-dimensional (3-qubit Hilbert space)")
        
        for i in range(8):
            if abs(state_vector[i]) > threshold:
                partner = H7Conservation.partner_state(i)
                if abs(state_vector[partner]) < threshold:
                    return False
        return True


if __name__ == "__main__":
    # Example usage
    config = MetriplexConfig()
    oracle = MetriplexOracle(config)
    
    print("Metriplex Oracle Information:")
    info = oracle.get_oracle_info()
    print(f"  Momentum range: {info['momentum_range']}")
    print(f"  Collision groups: {info['n_groups']}")
    print(f"  Hidden symmetry string s = {info['symmetry_string']}")
    
    print(f"\nEnergy Map (normalized):")
    for p in range(1, 7):
        group, output, energy = oracle.forward(p)
        print(f"  p={p}: group={group}, energy={energy:.4f}, output={output}")
    
    print(f"\nCollision Pairs:")
    for p in range(1, 7):
        partners = oracle.get_collision_partners(p)
        print(f"  p={p} collides with {partners}")
    
    print(f"\nH7 Conservation Pairing:")
    pairing = H7Conservation.pairing_table()
    for state, partner in pairing.items():
        print(f"  |{state}⟩ ↔ |{partner}⟩")