"""Dynamic TopK strategy based on protein-ligand interaction patterns"""

import torch
import torch.nn as nn

class TopKStrategy(nn.Module):
    def __init__(self, base_topk: int = 4, min_topk: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.base_topk = base_topk
        self.min_topk = min_topk
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.attn_proj = nn.Linear(32, 1)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.3))
        self.ln = nn.LayerNorm(32)
        self.ln_scalar = nn.LayerNorm(1)
        
        self.fuse_mlp = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
        self.gate = nn.Parameter(torch.tensor(0.5))
        
    def forward(self,
                node_features: torch.Tensor,     # [B, L, D]
                edge_features: torch.Tensor,     # [B, L, L, D]
                sequence_mask: torch.Tensor,     # [B, L]
                ligand_len: int = None           
                ) -> torch.Tensor:
        
        batch_size, seq_len, _ = node_features.shape
        contact_threshold = 5.0  
        
        node_encoding = self.feature_encoder(node_features)  # [B, L, 32]
        node_encoding = self.ln(node_encoding)
        
        attn_scores = torch.matmul(node_encoding, node_encoding.transpose(-2, -1))  # [B, L, L]
        attn_scores = attn_scores.mean(dim=2)  # [B, L]
        
        importance = self.attn_proj(node_encoding)  # [B, L, 1]
        importance = importance * sequence_mask.unsqueeze(-1)
        importance = importance.squeeze(-1)  # [B, L]
        importance = self.ln_scalar(importance.unsqueeze(-1)).squeeze(-1)
        
        if ligand_len is not None:
            if isinstance(ligand_len, int):
                ligand_len = torch.full((batch_size,), ligand_len, dtype=torch.long, device=node_features.device)
            idx = torch.arange(seq_len, device=node_features.device).unsqueeze(0).expand(batch_size, -1)  # [B, L]
            ligand_mask = idx < ligand_len.unsqueeze(1)  # [B, L]
            protein_mask = ~ligand_mask  # [B, L]
            dists = edge_features[..., 0]  # [B, L, L]
            contact_matrix = (dists < contact_threshold).float()  # [B, L, L]
            contact_freq = torch.zeros_like(importance)
            for b in range(batch_size):
                l_len = ligand_len[b]
                if l_len > 0:
                    contact_freq[b] = contact_matrix[b, :, :l_len].sum(dim=1) / (l_len + 1e-6)
            contact_freq = contact_freq * protein_mask
            center_score = torch.zeros_like(importance)
            for b in range(batch_size):
                l_len = ligand_len[b]
                if l_len > 0:
                    pl_dists = dists[b, :, :l_len]  # [L, ligand_len]
                    center_score[b] = (1.0 / (pl_dists + 1e-6)).sum(dim=1) / (l_len + 1e-6)
            center_score = center_score * protein_mask
        else:
            contact_freq = torch.zeros_like(importance)
            center_score = torch.zeros_like(importance)
        importance = self.ln_scalar(importance.unsqueeze(-1)).squeeze(-1)
        contact_freq = self.ln_scalar(contact_freq.unsqueeze(-1)).squeeze(-1)
        center_score = self.ln_scalar(center_score.unsqueeze(-1)).squeeze(-1)
        fused = torch.stack([importance, contact_freq, center_score], dim=-1)  # [B, L, 3]
        fused_score = self.fuse_mlp(fused).squeeze(-1)  # [B, L]
        gate = torch.sigmoid(self.gate)
        final_score = gate * fused_score + (1 - gate) * importance
        final_score = final_score * sequence_mask  # [B, L]
        final_score = torch.softmax(final_score, dim=1)
        return final_score  # [B, L]