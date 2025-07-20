import torch
import torch.nn as nn
from .moba_efficient import moba_attn_varlen
from .chunk_strategy import ChunkStrategy
from .topk_strategy import TopKStrategy

class MoBAInterformerAdapter(nn.Module):
    def __init__(self, 
                 hidden_dim: int, 
                 num_heads: int, 
                 moba_chunk_size: int = 64, 
                 moba_topk: int = 4,
                 dropout_rate=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.moba_chunk_size = moba_chunk_size
        self.moba_topk = moba_topk
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.chunk_strategy = ChunkStrategy(
            base_chunk_size=moba_chunk_size,
            hidden_dim=hidden_dim,
            max_seq_len=256
        )
        
        self.topk_strategy = TopKStrategy(
            base_topk=moba_topk,
            min_topk=2,
            hidden_dim=hidden_dim
        )
        
        self.register_buffer('avg_topk', torch.tensor(float(moba_topk)))
        self.register_buffer('avg_chunk_size', torch.tensor(float(moba_chunk_size)))
        self.smooth_factor = 0.8
        self.lambda_soft = nn.Parameter(torch.tensor(5.0))
        self.soft_bias_temp = 1.0
        
        self.fusion_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.task_specific = nn.ModuleDict({
            'energy': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=1e-6),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            'affinity': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=1e-6),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            'affinity_normal': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=1e-6),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        })
        
        self.interaction_enhance = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, 
                node_feats: torch.Tensor, 
                edge_feats: torch.Tensor, 
                pocket_mask: torch.Tensor = None,
                attn_bias: torch.Tensor = None,
                task_type: str = 'energy'):
        if torch.isnan(node_feats).any() or torch.isnan(edge_feats).any():
            print("# [MoBAInterformerAdapter] Input contains NaN values")
            node_feats = torch.nan_to_num(node_feats, nan=0.0)
            edge_feats = torch.nan_to_num(edge_feats, nan=0.0)
        
        batch_size, seq_len, hidden_dim = node_feats.shape
        
        sequence_mask = torch.ones(batch_size, seq_len, device=node_feats.device)
        chunk_sizes, max_seq_len = self.chunk_strategy(
            node_feats,
            edge_feats,
            sequence_mask,
            pocket_mask
        )
        
        node_scores = self.topk_strategy(
            node_feats,
            edge_feats,
            sequence_mask
        )
        
        with torch.no_grad():
            mean_scores = node_scores.mean(dim=1, keepdim=True)
            dynamic_topk_per_batch = (node_scores > mean_scores).sum(dim=1)
            dynamic_topk_per_batch = torch.clamp(dynamic_topk_per_batch, min=2, max=seq_len)
            dynamic_topk = int(dynamic_topk_per_batch.max().item())
            self.avg_topk = self.smooth_factor * self.avg_topk + (1 - self.smooth_factor) * dynamic_topk
            dynamic_topk = int(self.avg_topk.item())
            
            chunk_size = int(chunk_sizes[0].item())
            self.avg_chunk_size = self.smooth_factor * self.avg_chunk_size + (1 - self.smooth_factor) * chunk_size
            chunk_size = int(self.avg_chunk_size.item())
        
        score_min = node_scores.min(dim=1, keepdim=True)[0]
        score_max = node_scores.max(dim=1, keepdim=True)[0]
        score_range = score_max - score_min + 1e-6
        norm_scores = (node_scores - score_min) / score_range
        norm_scores = norm_scores / self.soft_bias_temp
        
        if attn_bias is not None:
            attn_bias = attn_bias.clone()
        else:
            attn_bias = torch.zeros(batch_size, seq_len + 1, seq_len + 1, device=node_feats.device, dtype=torch.float32)
        
        for b in range(batch_size):
            attn_bias[b, 1:, 1:] += self.lambda_soft * norm_scores[b][None, :] + self.lambda_soft * norm_scores[b][:, None]
        
        q = self.q_proj(node_feats)
        k = self.k_proj(node_feats)
        v = self.v_proj(node_feats)
        
        if torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any():
            print("# [MoBAInterformerAdapter] Projection contains NaN values")
            q = torch.nan_to_num(q, nan=0.0)
            k = torch.nan_to_num(k, nan=0.0)
            v = torch.nan_to_num(v, nan=0.0)
        
        cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * seq_len,
            seq_len,
            device=node_feats.device,
            dtype=torch.int32
        )
        
        q = q.reshape(-1, self.num_heads, hidden_dim // self.num_heads)
        k = k.reshape(-1, self.num_heads, hidden_dim // self.num_heads)
        v = v.reshape(-1, self.num_heads, hidden_dim // self.num_heads)
        
        moba_output = moba_attn_varlen(
            q,
            k,
            v,
            cu_seqlens,
            seq_len,
            moba_chunk_size=chunk_size,
            moba_topk=dynamic_topk
        )
        
        processed_node_feats = moba_output.reshape(batch_size, seq_len, hidden_dim)
        
        if torch.isnan(processed_node_feats).any():
            print("# [MoBAInterformerAdapter] MOBA output contains NaN values")
            processed_node_feats = torch.nan_to_num(processed_node_feats, nan=0.0)
        
        if task_type in self.task_specific:
            if task_type == 'energy':
                processed_feats = self.task_specific[task_type](processed_node_feats)
            else:
                fused = torch.cat([processed_node_feats, edge_feats], dim=-1)
                processed_feats = self.task_specific[task_type](fused)
                
                if task_type in ['affinity', 'affinity_normal']:
                    interaction_feats = self.interaction_enhance(fused)
                    processed_feats = processed_feats + interaction_feats
                
                gate = self.fusion_gate(fused)
                processed_feats = processed_feats * gate + processed_node_feats * (1 - gate)
        
        if torch.isnan(processed_feats).any():
            print("# [MoBAInterformerAdapter] Final output contains NaN values")
            processed_feats = torch.nan_to_num(processed_feats, nan=0.0)
        
        processed_node_feats = self.fusion_norm(processed_feats)
        
        return processed_node_feats, edge_feats

    def get_lambda_soft_reg(self):
        return self.lambda_soft.pow(2) 