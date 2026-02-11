"""
Structural Causal Model (SCM) Implementation

Represents the causal structure of astrophysical systems with:
- Directed Acyclic Graph (DAG) structure
- Causal mechanism functions: X_i = f_i(Pa(X_i), U_i)
- Intervention operators: do(X_j = x*)
- Exogenous noise distributions
"""

from typing import Dict, List, Callable, Optional, Set, Tuple
import torch
import torch.nn as nn
import networkx as nx
import numpy as np


class CausalNode:
    """Represents a single variable in the SCM."""
    
    def __init__(
        self,
        name: str,
        mechanism: Optional[Callable] = None,
        is_latent: bool = False,
        is_observable: bool = True,
    ):
        self.name = name
        self.mechanism = mechanism
        self.is_latent = is_latent
        self.is_observable = is_observable
        self.parents: Set[str] = set()
        self.children: Set[str] = set()
    
    def __repr__(self):
        return f"CausalNode({self.name}, latent={self.is_latent})"


class StructuralCausalModel:
    """
    Structural Causal Model for astronomical systems.
    
    Represents the causal graph G = (V, E) where:
    - V are variables (latent physical states P and observables O)
    - E are directed edges representing causal relationships
    
    Each variable X_i is determined by:
        X_i = f_i(Pa(X_i), U_i)
    where Pa(X_i) are parents and U_i is exogenous noise.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, CausalNode] = {}
        self.mechanisms: Dict[str, nn.Module] = {}
        
    def add_node(
        self,
        name: str,
        mechanism: Optional[nn.Module] = None,
        is_latent: bool = False,
        is_observable: bool = True,
    ):
        """Add a variable to the SCM."""
        node = CausalNode(name, mechanism, is_latent, is_observable)
        self.nodes[name] = node
        self.graph.add_node(name)
        
        if mechanism is not None:
            self.mechanisms[name] = mechanism
    
    def add_edge(self, parent: str, child: str):
        """Add a causal edge from parent to child."""
        if parent not in self.nodes or child not in self.nodes:
            raise ValueError("Both parent and child must exist in the graph")
        
        self.graph.add_edge(parent, child)
        self.nodes[parent].children.add(child)
        self.nodes[child].parents.add(parent)
    
    def get_parents(self, node: str) -> List[str]:
        """Get parents of a node in topological order."""
        return list(self.graph.predecessors(node))
    
    def get_children(self, node: str) -> List[str]:
        """Get children of a node."""
        return list(self.graph.successors(node))
    
    def topological_order(self) -> List[str]:
        """Return nodes in topological order for causal propagation."""
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXError:
            raise ValueError("Graph contains cycles - not a valid DAG")
    
    def intervene(self, interventions: Dict[str, torch.Tensor]) -> "StructuralCausalModel":
        """
        Create a new SCM with interventions applied: do(X_j = x*)
        
        This removes incoming edges to intervened variables and
        fixes their values.
        
        Args:
            interventions: Dict mapping variable names to intervention values
            
        Returns:
            New SCM with interventions applied
        """
        # Create a copy of the SCM
        intervened_scm = StructuralCausalModel()
        
        # Copy all nodes
        for name, node in self.nodes.items():
            intervened_scm.add_node(
                name,
                mechanism=self.mechanisms.get(name),
                is_latent=node.is_latent,
                is_observable=node.is_observable,
            )
        
        # Copy edges, but remove incoming edges to intervened variables
        for parent, child in self.graph.edges():
            if child not in interventions:
                intervened_scm.add_edge(parent, child)
        
        # Store intervention values
        intervened_scm.interventions = interventions
        
        return intervened_scm
    
    def forward(
        self,
        exogenous: Dict[str, torch.Tensor],
        interventions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform forward causal propagation through the SCM.
        
        Args:
            exogenous: Dict of exogenous noise variables U_i
            interventions: Optional interventions to apply
            
        Returns:
            Dict of all variable values
        """
        if interventions is None:
            interventions = {}
        
        values = {}
        
        # Process nodes in topological order
        for node_name in self.topological_order():
            # If intervened, use intervention value
            if node_name in interventions:
                values[node_name] = interventions[node_name]
                continue
            
            # Get parent values
            parents = self.get_parents(node_name)
            parent_values = [values[p] for p in parents]
            
            # Get exogenous noise
            noise = exogenous.get(node_name, torch.zeros(1))
            
            # Apply causal mechanism
            mechanism = self.mechanisms.get(node_name)
            if mechanism is not None:
                if parent_values:
                    parent_tensor = torch.cat(parent_values, dim=-1)
                    values[node_name] = mechanism(parent_tensor, noise)
                else:
                    values[node_name] = mechanism(noise)
            else:
                # Default: identity mechanism (just exogenous noise)
                values[node_name] = noise
        
        return values
    
    def get_markov_blanket(self, node: str) -> Set[str]:
        """
        Get the Markov blanket of a node:
        - Parents
        - Children
        - Parents of children (co-parents)
        """
        if node not in self.nodes:
            raise ValueError(f"Node {node} not in graph")
        
        blanket = set(self.get_parents(node))
        children = self.get_children(node)
        blanket.update(children)
        
        for child in children:
            blanket.update(self.get_parents(child))
        
        blanket.discard(node)
        return blanket
    
    def d_separation(self, X: Set[str], Y: Set[str], Z: Set[str]) -> bool:
        """
        Check if X and Y are d-separated given Z.
        
        Used for conditional independence testing.
        """
        return nx.d_separated(self.graph, X, Y, Z)
    
    def __repr__(self):
        return f"SCM(nodes={len(self.nodes)}, edges={len(self.graph.edges())})"


class AstronomicalSCM(StructuralCausalModel):
    """
    Specialized SCM for astronomical systems with physics structure.
    
    Variables organized as:
    - Latent physical variables P (mass, metallicity, age, environment, etc.)
    - Observable variables O (photometry, spectra, light curves, etc.)
    - Noise/bias variables N (instrument noise, selection bias)
    """
    
    def __init__(
        self,
        latent_dim: int = 2000,
        observable_dim: int = 6000,
        noise_dim: int = 2000,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.observable_dim = observable_dim
        self.noise_dim = noise_dim
        
        # Add latent physical variables
        for i in range(latent_dim):
            self.add_node(f"P_{i}", is_latent=True, is_observable=False)
        
        # Add observable variables
        for i in range(observable_dim):
            self.add_node(f"O_{i}", is_latent=False, is_observable=True)
        
        # Add noise variables
        for i in range(noise_dim):
            self.add_node(f"N_{i}", is_latent=False, is_observable=True)
    
    def add_physics_structure(self, edge_probability: float = 0.01):
        """
        Add causal edges based on physical structure.
        
        Assumes:
        - Latent variables can cause other latent variables
        - Latent variables cause observables
        - Noise affects observables
        """
        np.random.seed(42)  # For reproducibility
        
        # Add edges from latent to observables (sparse)
        for i in range(self.latent_dim):
            # Each latent affects some observables
            n_affected = int(self.observable_dim * edge_probability)
            affected_obs = np.random.choice(
                self.observable_dim, size=n_affected, replace=False
            )
            for j in affected_obs:
                self.add_edge(f"P_{i}", f"O_{j}")
        
        # Add edges from noise to observables (direct)
        for i in range(min(self.noise_dim, self.observable_dim)):
            self.add_edge(f"N_{i}", f"O_{i}")
