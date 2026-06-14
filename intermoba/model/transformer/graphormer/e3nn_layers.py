import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

class E3EquivariantLayer(MessagePassing):
    def __init__(self, hidden_dim, edge_dim, dropout_rate=0.2):
        super().__init__(aggr='add', node_dim=0)
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.dropout_rate = dropout_rate
        
        self.scalar_net = nn.Sequential(
            nn.LayerNorm(hidden_dim + edge_dim, eps=1e-6),  
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-6),  
            nn.Dropout(dropout_rate)
        )
        
        # 向量场网络
        self.vector_net = nn.Sequential(
            nn.LayerNorm(hidden_dim + edge_dim, eps=1e-6),  
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Dropout(dropout_rate)
        )
        
        self.pos_dropout = nn.Dropout(p=dropout_rate)
        self.pos_clip_value = 10.0  
        self.x_norm = nn.LayerNorm(hidden_dim, eps=1e-6) 
        self.pos_norm = nn.LayerNorm(3, eps=1e-6)
        
        self.equivariant_attn = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, pos, edge_index, edge_attr):
        
        if torch.isnan(x).any() or torch.isnan(pos).any() or torch.isnan(edge_attr).any():
            print("# [E3EquivariantLayer] Input contains NaN values")
            x = torch.nan_to_num(x, nan=0.0)
            pos = torch.nan_to_num(pos, nan=0.0)
            edge_attr = torch.nan_to_num(edge_attr, nan=0.0)
        
        x = self.x_norm(x)
        pos = self.pos_norm(pos)
        return self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, pos_i, pos_j, edge_attr):
        
        rel_pos = pos_j - pos_i  # [E, 3]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # [E, 1]
        dist = torch.clamp(dist, min=1e-6)  
        rel_pos_norm = rel_pos / dist  # [E, 3]
        
        msg_features = torch.cat([x_j, edge_attr], dim=-1)
        
        attn_weights = self.equivariant_attn(msg_features)  # [E, 1]
        attn_weights = torch.sigmoid(attn_weights)
        
        scalar_update = self.scalar_net(msg_features)  # [E, hidden_dim]
        vector_weight = self.vector_net(msg_features)  # [E, 1]
        
        vector_update = vector_weight * rel_pos_norm * attn_weights  # [E, 3]
        scalar_update = scalar_update * attn_weights  # [E, hidden_dim]
        
        if torch.isnan(scalar_update).any() or torch.isnan(vector_update).any():
            print("# [E3EquivariantLayer] Update contains NaN values")
            scalar_update = torch.nan_to_num(scalar_update, nan=0.0)
            vector_update = torch.nan_to_num(vector_update, nan=0.0)
        
        return scalar_update, vector_update
    
    def aggregate(self, inputs, index, x=None, pos=None):
        scalar_updates, vector_updates = inputs
        N = x.size(0) if x is not None else int(index.max().item()) + 1
        scalar_aggr = scatter(scalar_updates, index, dim=0, reduce='sum', dim_size=N)
        vector_aggr = scatter(vector_updates, index, dim=0, reduce='sum', dim_size=N)
        
        if torch.isnan(scalar_aggr).any() or torch.isnan(vector_aggr).any():
            print("# [E3EquivariantLayer] Aggregation contains NaN values")
            scalar_aggr = torch.nan_to_num(scalar_aggr, nan=0.0)
            vector_aggr = torch.nan_to_num(vector_aggr, nan=0.0)
        
        return scalar_aggr, vector_aggr
    
    def update(self, aggr_out, x, pos):
        scalar_aggr, vector_aggr = aggr_out
        
        gate = self.fusion_gate(torch.cat([x, scalar_aggr], dim=-1))
        x_update = x * (1 - gate) + scalar_aggr * gate
        
        pos_update = pos + vector_aggr
        pos_update = self.pos_dropout(pos_update)
        pos_update = torch.clamp(pos_update, min=-self.pos_clip_value, max=self.pos_clip_value)
        
        if torch.isnan(x_update).any() or torch.isnan(pos_update).any():
            print("# [E3EquivariantLayer] Final update contains NaN values")
            x_update = torch.nan_to_num(x_update, nan=0.0)
            pos_update = torch.nan_to_num(pos_update, nan=0.0)
        
        return x_update, pos_update

class E3EquivariantBlock(nn.Module):
    def __init__(self, hidden_dim, edge_dim, num_layers=3, dropout_rate=0.2):
        super().__init__()
        self.layers = nn.ModuleList([
            E3EquivariantLayer(hidden_dim, edge_dim, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.task_specific = nn.ModuleDict({
            'energy': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            'affinity': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            'affinity_normal': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        })
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, pos, edge_index, edge_attr, task_type='energy'):
        x_orig, pos_orig = x, pos
        
        for layer in self.layers:
            x, pos = layer(x, pos, edge_index, edge_attr)
            
        if task_type in self.task_specific:
            if task_type == 'energy':
                x = self.task_specific[task_type](x)
            else:
                fused = torch.cat([x, x_orig], dim=-1)
                x = self.task_specific[task_type](fused)
                
                gate = self.fusion_gate(fused)
                x = x * gate + x_orig * (1 - gate)
        
        return x, pos 