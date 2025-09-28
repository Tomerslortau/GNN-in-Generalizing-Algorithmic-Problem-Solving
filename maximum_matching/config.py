from dataclasses import dataclass
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class GraphConfig:
    # Graph type
    graph_type: str  # One of: 'er_bipartite', 'sbm_bipartite', 'powerlaw_bipartite', 
                     # 'er_general', 'sbm_general', 'powerlaw_general', 'sparse_er_bipartite'
    
    # General parameters
    seed: int = 42
    
    def __getitem__(self, key):
        return getattr(self, key)
        
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def set_fixed_n(self, n):
        """Set fixed node count by making min=max for all size parameters"""
        if self.graph_type.endswith('bipartite'):
            self.nU_min = n
            self.nU_max = n
            self.nV_min = n
            self.nV_max = n
        else:
            self.n_min = n
            self.n_max = n
        return self
    
    # General graph parameters
    n_min: int = 100
    n_max: int = 150

    # Bipartite graph parameters
    nU_min: int = n_min
    nU_max: int = n_max
    nV_min: int = n_min
    nV_max: int = n_max

    # Edge probability (for ER models)
    p_min: float = 0.005
    p_max: float = 0.005
    p: float = 0.015

    # SBM parameters
    sbm_blocks: int = 2  # For general graphs
    sbm_blocks_u: int = 2  # For bipartite graphs
    sbm_blocks_v: int = 2  # For bipartite graphs
    sbm_p_in: float = 0.02
    sbm_p_out: float = 0.005
    
    # Power-law parameters
    plaw_exp: float = 2.5  # For general graphs
    plaw_exp_u: float = 2.5  # For bipartite graphs
    plaw_exp_v: float = 2.5  # For bipartite graphs
    plaw_min_deg: int = 1
    
    @classmethod
    def create_er_bipartite(cls, nU_min=20, nU_max=60, nV_min=20, nV_max=60, p=0.01, seed=42):
        return cls(
            graph_type='er_bipartite',
            nU_min=nU_min, nU_max=nU_max,
            nV_min=nV_min, nV_max=nV_max,
            p=p, seed=seed
        )
    
    @classmethod
    def create_sbm_bipartite(cls, nU_min=20, nU_max=60, nV_min=20, nV_max=60, 
                            sbm_blocks_u=2, sbm_blocks_v=2, 
                            sbm_p_in=0.02, sbm_p_out=0.005, seed=42):
        return cls(
            graph_type='sbm_bipartite',
            nU_min=nU_min, nU_max=nU_max,
            nV_min=nV_min, nV_max=nV_max,
            sbm_blocks_u=sbm_blocks_u, sbm_blocks_v=sbm_blocks_v,
            sbm_p_in=sbm_p_in, sbm_p_out=sbm_p_out,
            seed=seed
        )
    
    @classmethod
    def create_powerlaw_bipartite(cls, nU_min=20, nU_max=60, nV_min=20, nV_max=60,
                                 plaw_exp_u=2.5, plaw_exp_v=2.5, plaw_min_deg=1, seed=42):
        return cls(
            graph_type='powerlaw_bipartite',
            nU_min=nU_min, nU_max=nU_max,
            nV_min=nV_min, nV_max=nV_max,
            plaw_exp_u=plaw_exp_u, plaw_exp_v=plaw_exp_v,
            plaw_min_deg=plaw_min_deg,
            seed=seed
        )
    
    @classmethod
    def create_er_general(cls, n_min=100, n_max=200, p=0.01, seed=42):
        return cls(
            graph_type='er_general',
            n_min=n_min, n_max=n_max,
            p=p, seed=seed
        )
    
    @classmethod
    def create_sbm_general(cls, n_min=100, n_max=200,
                          sbm_blocks=2, sbm_p_in=0.02, sbm_p_out=0.005, seed=42):
        return cls(
            graph_type='sbm_general',
            n_min=n_min, n_max=n_max,
            sbm_blocks=sbm_blocks,
            sbm_p_in=sbm_p_in, sbm_p_out=sbm_p_out,
            seed=seed
        )
    
    @classmethod
    def create_powerlaw_general(cls, n_min=100, n_max=200,
                               plaw_exp=2.5, plaw_min_deg=1, seed=42):
        return cls(
            graph_type='powerlaw_general',
            n_min=n_min, n_max=n_max,
            plaw_exp=plaw_exp,
            plaw_min_deg=plaw_min_deg,
            seed=seed
        )
    
    @classmethod
    def create_sparse_er_bipartite(cls, n=100, p=0.01, seed=42):
        return cls(
            graph_type='sparse_er_bipartite',
            nU_max=n,  # Using nU_max as the size parameter
            p=p, seed=seed
        )