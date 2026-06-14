"""Dynamic chunk size strategy based on protein chemical properties and spatial relationships"""

import torch
import torch.nn as nn

class ChunkStrategy(nn.Module):
    def __init__(self, base_chunk_size: int = 64, hidden_dim: int = 128, max_seq_len: int = 256):
        super().__init__()
        self.base_chunk_size = base_chunk_size
        self.max_seq_len = max_seq_len
        self.node_proj = nn.Linear(hidden_dim, 32)
        self.edge_proj = nn.Sequential(
            nn.Linear(1, 64),  
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.importance_proj = nn.Linear(64, 1)
        
    def forward(self, 
                node_features: torch.Tensor,    # [B, L, D]
                edge_features: torch.Tensor,    # [B, L, L, D]
                sequence_mask: torch.Tensor,    # [B, L]
                pocket_mask: torch.Tensor = None 
                ) -> torch.Tensor:
        
        batch_size, seq_len, _ = node_features.shape

        if seq_len > self.max_seq_len:
            adjusted_base_chunk = min(self.base_chunk_size, self.max_seq_len // 4)
        else:
            adjusted_base_chunk = self.base_chunk_size
        
        node_encoding = self.node_proj(node_features)  # [B, L, 32]
        edge_mean = edge_features.mean(dim=-1)  # [B, L, L]
        edge_summary = edge_mean.mean(dim=-1)  # [B, L]
        edge_summary = edge_summary.unsqueeze(-1)  # [B, L, 1]
        edge_encoding = self.edge_proj(edge_summary)  # [B, L, 32]
        combined_features = torch.cat([node_encoding, edge_encoding], dim=-1)  # [B, L, 64]
        
        importance = torch.sigmoid(self.importance_proj(combined_features))  # [B, L, 1]
        importance = importance * sequence_mask.unsqueeze(-1)
        
        distance_core = 6.0
        distance_shell = 10.0
        chunk_sizes = torch.full((batch_size, 1), float(self.base_chunk_size), device=node_features.device)
        for b in range(batch_size):
            if pocket_mask is not None and pocket_mask[b].any():
                pocket_idx = pocket_mask[b].nonzero(as_tuple=True)[0]
                dist_to_pocket = edge_features[b, :, pocket_idx, 0].min(dim=1)[0]  # [L]
                core_mask = dist_to_pocket < distance_core
                shell_mask = (dist_to_pocket >= distance_core) & (dist_to_pocket < distance_shell)
                imp = importance[b].squeeze(-1)
                imp_mean = imp.mean()
                shell_mask = shell_mask & (imp > imp_mean)
                merged_mask = pocket_mask[b].bool() | core_mask | shell_mask
                merged_len = merged_mask.sum().item()
                if merged_len > self.max_seq_len:
                    idx_sorted = torch.argsort(dist_to_pocket + (1 - imp))
                    keep_idx = idx_sorted[:self.max_seq_len]
                    merged_mask = torch.zeros_like(merged_mask)
                    merged_mask[keep_idx] = 1
                    merged_len = self.max_seq_len
                chunk_size = max(merged_len, adjusted_base_chunk)
                chunk_size = min(chunk_size, self.max_seq_len)
                if chunk_size < 1 or not torch.isfinite(torch.tensor(chunk_size)):
                    chunk_size = self.base_chunk_size
                chunk_sizes[b, 0] = float(chunk_size)
            else:
                chunk_importance = importance[b].mean(dim=0)  # [1]
                chunk_size = self.base_chunk_size * chunk_importance
                chunk_size = torch.clamp(chunk_size, 
                                        min=float(adjusted_base_chunk // 2),
                                        max=float(min(adjusted_base_chunk * 2, self.max_seq_len)))
                if chunk_size < 1 or not torch.isfinite(chunk_size):
                    chunk_size = self.base_chunk_size
                chunk_sizes[b, 0] = chunk_size.item()
        
        return chunk_sizes.squeeze(-1), self.max_seq_len  