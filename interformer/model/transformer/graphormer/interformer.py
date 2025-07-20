import os
import pickle
from rdkit import Chem

import torch
import torch.nn as nn
from interformer.model.sbdd_model import SBDD
from interformer.model.transformer.graphormer.graphformer_utils import init_embedding, ComplexEncoder, EncoderLayer, \
    PositionwiseFeedForward

import numpy as np
import torch.nn.functional as F

import shelve
from .MoBA.moba_interformer_adapter import MoBAInterformerAdapter

class AtomHead(nn.Module):
    def __init__(self, hidden_dim, node_feat_size, dropout_rate=0.2):
        super(AtomHead, self).__init__()
        self.node_ln = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.atom_type_proj = PositionwiseFeedForward(hidden_dim, hidden_dim * 4, d_out=node_feat_size,
                                                      dropout=dropout_rate)
        
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=0, label_smoothing=0.1)
        self.init_param()

    def init_param(self):
        self.apply(lambda module: init_embedding(module))

    def forward(self, node_feats, node_x):
        node_x = node_x.unsqueeze(-1).long() 
        node_feats = node_feats[:, 1:, :]  
        
        if torch.isnan(node_feats).any():
            print("# [AtomHead] NaN detected in node_feats, replacing with zeros")
            node_feats = torch.nan_to_num(node_feats, 0.0)
            
        node_feats = self.node_ln(node_feats)
        
        atom_type_hat = self.atom_type_proj(node_feats)
        
        if torch.isnan(atom_type_hat).any():
            print("# [AtomHead] NaN detected in atom_type_hat, replacing with zeros")
            atom_type_hat = torch.nan_to_num(atom_type_hat, 0.0)
            
        if torch.isinf(atom_type_hat).any():
            print("# [AtomHead] Inf detected in atom_type_hat, clipping values")
            atom_type_hat = torch.clamp(atom_type_hat, min=-100, max=100)
            
        atom_type_hat = atom_type_hat.transpose(1, 2)
        
        try:
            if node_x.max() >= atom_type_hat.size(1):
                print(f"# [AtomHead] Invalid label index: max={node_x.max()}, num_classes={atom_type_hat.size(1)}")
                return torch.tensor(0.001, device=atom_type_hat.device)
                
            if (node_x == 0).all():
                print("# [AtomHead] All node_x values are 0, checking data processing")
                print(f"# [AtomHead] node_x shape: {node_x.shape}")
                print(f"# [AtomHead] node_x unique values: {torch.unique(node_x)}")
                print(f"# [AtomHead] node_x statistics: min={node_x.min()}, max={node_x.max()}, mean={node_x.float().mean()}")
                print(f"# [AtomHead] node_feats shape: {node_feats.shape}")
                print(f"# [AtomHead] node_feats statistics: min={node_feats.min()}, max={node_feats.max()}, mean={node_feats.mean()}")
                print(f"# [AtomHead] atom_type_hat shape: {atom_type_hat.shape}")
                print(f"# [AtomHead] atom_type_hat statistics: min={atom_type_hat.min()}, max={atom_type_hat.max()}, mean={atom_type_hat.mean()}")
               
                return torch.tensor(0.001, device=atom_type_hat.device)
                
            log_probs = F.log_softmax(atom_type_hat, dim=1)
            
            ce_loss = self.loss_fn(log_probs, node_x.squeeze(-1))
            
            if torch.isnan(ce_loss) or torch.isinf(ce_loss):
                print("# [AtomHead] Invalid loss value detected")
                print(f"# [AtomHead] log_probs range: [{log_probs.min()}, {log_probs.max()}]")
                print(f"# [AtomHead] node_x range: [{node_x.min()}, {node_x.max()}]")
                print(f"# [AtomHead] node_x unique values: {torch.unique(node_x)}")
                return torch.tensor(0.001, device=atom_type_hat.device)
                
            ce_loss = 0.001 * ce_loss
                
        except RuntimeError as e:
            print(f"# [AtomHead] Error in loss calculation: {e}")
            print(f"# [AtomHead] atom_type_hat shape: {atom_type_hat.shape}, node_x shape: {node_x.shape}")
            print(f"# [AtomHead] atom_type_hat range: [{atom_type_hat.min()}, {atom_type_hat.max()}]")
            print(f"# [AtomHead] node_x range: [{node_x.min()}, {node_x.max()}]")
            print(f"# [AtomHead] node_x unique values: {torch.unique(node_x)}")
            return torch.tensor(0.001, device=atom_type_hat.device)
            
        return ce_loss


class VinaScoreHead(nn.Module):
    def __init__(self, hidden_dim, node_featurizer, dropout=0.2, energy_output_folder='', edge_output_layer=True):
        super(VinaScoreHead, self).__init__()
        self.num_atom_types = 29
        self.train_dist_cut_off = 4.0  # cut_off + 1.9 * 2 = real_distance, approximately 8.0 A
        self.node_featurizer = node_featurizer
        # trainable parameters
        self.pair_ln = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.node_ln = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.final_pair_ln = nn.LayerNorm(hidden_dim, eps=1e-6)
        # Gaussian Heads
        self.meanHead = PositionwiseFeedForward(hidden_dim, hidden_dim * 2, d_out=4, dropout=dropout)
        self.sigmaHead = PositionwiseFeedForward(hidden_dim, hidden_dim * 2, d_out=4, dropout=dropout)
        self.WeightHead = PositionwiseFeedForward(hidden_dim, hidden_dim * 2, d_out=4, dropout=dropout)
        # init
        self.init_param()
        # Vina Section
        self.vdw_table = self.make_vdw_table(node_featurizer)
        self.make_hydro_hbond_table()
        # Output Energy
        self.energy_output_folder = energy_output_folder
        print(f"# [Interformer] energy_output_fodler:{energy_output_folder}")
        # Ablation Study
        self.use_edge_output_layer = edge_output_layer

    def init_param(self):
        self.apply(lambda module: init_embedding(module))

    def make_vdw_table(self, node_featurizer):
        edge_space = 29 ** 2
        table = torch.zeros([edge_space, 1])
        for i in range(28):
            for j in range(28):
                vdw_1 = node_featurizer.atom_type_data[i].xs_radius
                vdw_2 = node_featurizer.atom_type_data[j].xs_radius
                index = (1 + i) * 29 + (1 + j)
                table[index] = vdw_1 + vdw_2
        vdw_embedding = nn.Embedding(edge_space, 1).requires_grad_(False)
        vdw_embedding.weight = torch.nn.Parameter(table)
        vdw_embedding.weight.requires_grad = False
        return vdw_embedding

    def make_hydro_hbond_table(self):
        edge_space = 29 ** 2
        hydro_list = [3, 5, 18, 19, 20, 21]  # CH, ACH, F, CL, B, I
        acceptor_list = [9, 10, 13, 14]  # NDA, NA, ODA, OA
        donnor_list = [8, 9, 12, 13]  # XD, XDA, OD, ODA
        #
        hydro_table = torch.zeros([edge_space, 1])
        hbond_table = torch.zeros([edge_space, 1])
        for h_id in hydro_list:
            for h2_id in hydro_list:
                index = h_id * 29 + h2_id
                hydro_table[index] = 1.
        for a_id in acceptor_list:
            for d_id in donnor_list:
                index = a_id * 29 + d_id
                hbond_table[index] = 1.
                index = d_id * 29 + a_id
                hbond_table[index] = 1.
        # assign to embedding
        hydro_embedding = nn.Embedding(edge_space, 1).requires_grad_(False)
        hydro_embedding.weight = torch.nn.Parameter(hydro_table)
        hydro_embedding.weight.requires_grad = False
        self.hydro_table = hydro_embedding
        hbond_embedding = nn.Embedding(edge_space, 1).requires_grad_(False)
        hbond_embedding.weight = torch.nn.Parameter(hbond_table)
        hbond_embedding.weight.requires_grad = False
        self.hbond_table = hbond_embedding

    def atom_type2pair_type(self, x: torch.Tensor) -> torch.Tensor:
        # [b, n, 1]
        atoms_type_x = torch.where(x >= self.num_atom_types, x - self.num_atom_types,
                                   x)  # 29-th is the 0-idx in pocket-atoms-type
        pair_type = atoms_type_x[:, :, None, :] * self.num_atom_types + atoms_type_x[:, None, :]
        return pair_type

    def edge_output_layer(self, node_feats, pair_feats):
        pair_feats = pair_feats[:, 1:, 1:, :]  # [b, n, n, h]
        pair_feats = self.pair_ln(pair_feats)
        pair_feats = (pair_feats + pair_feats.transpose(1, 2)) * 0.5  # merge from L->P, and P->L
        # 2. node_feats=[b, n, d]
        node_feats = self.node_ln(node_feats[:, 1:])  # exclude vn node
        node_feats = torch.einsum('b i d, b j d-> b i j d', node_feats, node_feats)
        #####
        # Predict the closest pair
        final_pair_feats = node_feats + pair_feats
        final_pair_feats = self.final_pair_ln(final_pair_feats)
        return final_pair_feats

    def gaussian(self, d, mean, width):
        width = torch.clamp(width, min=1e-6) 
        mean = torch.clamp(mean, min=0.1) 
        normal = torch.distributions.Normal(mean, width)
        logik = normal.log_prob(d.expand_as(normal.loc))
        return logik

    def mdn_loss(self, logPro, pi_soft, close_pairs_mask, y):
        close_pairs_mask = close_pairs_mask.squeeze(-1)  # [b, n, n]
        y_mask = (y >= 0.).view(-1)  # [b, 1, 1]
        b, n, _ = close_pairs_mask.shape
        ####
        # Cross-Entropy
        y_mask = y_mask.long().view(b, 1, 1)
        # default
        mix_loglik = torch.log(pi_soft + 1e-9) + logPro 
        loglik = torch.logsumexp(mix_loglik, dim=-1)
        loglik = loglik * close_pairs_mask
        # neg_lik, it is hard to handle
        neg_lik = (pi_soft * torch.exp(logPro)).sum(dim=-1)
        neg_lik = torch.log((1. - neg_lik).clip(1e-9))
        neg_lik = neg_lik * close_pairs_mask
        # ce-loss
        ce_loss = -(y_mask * loglik + (1 - y_mask) * neg_lik)
        ce_mean_loss = ce_loss.mean()
        ### Log
        ce_pos_loss = -(y_mask * loglik).mean()
        return ce_mean_loss, ce_pos_loss

    def GaussianScore(self, d, vdw_pair, pair_type, pair_mask, ligand_mask, pair_emb, batched_data):
        #
        close_pairs_mask = (d < self.train_dist_cut_off) & pair_mask
        hydro_pair = self.hydro_table(pair_type).bool()  # & (d < 1.5)
        hbond_pair = self.hbond_table(pair_type).bool()  # & (d < 0.)
        collision_threshold = 2.0  
        collision_mask = (d < collision_threshold).float()
        pair_emb = pair_emb * (1.0 + collision_mask)
        # vdw-terms
        mean = F.elu(self.meanHead(pair_emb)) + 1.0  
        sigma = F.elu(self.sigmaHead(pair_emb)) + 1.0 + 1e-6  
        all_terms = self.gaussian(d, mean, sigma)  # * vdw_pair_mask
        vdw_term0 = all_terms[:, :, :, 0, None]
        vdw_term1 = all_terms[:, :, :, 1, None]
        # hydro-terms
        hydro_term = all_terms[:, :, :, 2, None]
        hydro_term = hydro_term * hydro_pair
        # hbond-terms
        hbond_term = all_terms[:, :, :, 3, None]
        hbond_term = hbond_term * hbond_pair
        ####
        pi = self.WeightHead(pair_emb)
        # mask by hydro_pair and hbond_pair
        hydro_soft_mask = torch.where(hydro_pair, torch.tensor(0.).to(d), torch.tensor(float('-inf')).to(d))
        hbond_soft_mask = torch.where(hbond_pair, torch.tensor(0.).to(d), torch.tensor(float('-inf')).to(d))
        zero_soft_mask = torch.zeros_like(hydro_soft_mask)
        mask = torch.cat([zero_soft_mask, zero_soft_mask, hydro_soft_mask, hbond_soft_mask], dim=-1)
        pi_masked = pi + mask
        # pi * pro
        pi_soft = torch.softmax(pi_masked, dim=-1) + 1e-9 
        loglik = torch.cat([vdw_term0, vdw_term1, hydro_term, hbond_term], dim=-1)
        # Loss
        mean_mdn_loss, mdn_pos_loss = self.mdn_loss(loglik, pi_soft, close_pairs_mask, batched_data['y'])
       
        collision_threshold = 2.0  
        min_dist_threshold = 1.5  
        collision_loss = torch.relu(collision_threshold - d).mean() 
        min_dist_loss = torch.relu(min_dist_threshold - d).mean()  
        gScore_loss = mean_mdn_loss + 0.1 * collision_loss + 0.1 * min_dist_loss
        self.collision_loss = collision_loss
        self.min_dist_loss = min_dist_loss
        try:
            if self.energy_output_folder:
                target = batched_data['pdb'][0]
                output_f = f'{self.energy_output_folder}/gaussian_predict/{target}_G.db'
                with shelve.open(output_f) as db:
                    last_id = len(db) if len(db) else 0
                    print(f"[Interformer] Gussian Score ->{target}-{last_id}")
                    data = {'ligand_len': batched_data['ligand_len'].cpu().numpy(),
                            'pocket_len': batched_data['pocket_len'].cpu().numpy(),
                            'pi': pi_soft.cpu().float().numpy().squeeze(0),
                            'mean': mean.cpu().float().numpy().squeeze(0),
                            'sigma': sigma.cpu().float().numpy().squeeze(0),
                            'hbond_pair': hbond_pair.cpu().numpy().squeeze(0),
                            'hydro_pair': hydro_pair.cpu().numpy().squeeze(0),
                            'd': d.cpu().numpy().squeeze(0),
                            'vdw_pair': vdw_pair.cpu().numpy().squeeze(0),
                            }
                    db[str(last_id)] = data
        except Exception as e:
            print(f"# [GaussianScore] Error saving gaussian_predict: {e}")
        return gScore_loss, mdn_pos_loss

    def forward(self, node_feats, pair_feats, batched_data):
        D, x, pair_mask, ligand_mask = batched_data['D'], batched_data['node_type'], batched_data['pair_mask'], batched_data[
            'ligand_mask']
        x = x.unsqueeze(-1)
        # Edge output Layer
        if self.use_edge_output_layer:
            pair_emb = self.edge_output_layer(node_feats, pair_feats)
        else:
            pair_emb = pair_feats[:, 1:, 1:, :]
        # Pair-Type
        pair_type = self.atom_type2pair_type(x).squeeze(-1)
        vdw_pair = self.vdw_table(pair_type)  # [b, n, n, 1], (d < 5.)
        d = D - vdw_pair
        min_dist_threshold = 1.5  
        valid_mask = (d > min_dist_threshold).float()
        pair_emb = pair_emb * valid_mask
        # EnergyScore
        gScore_loss, gScore_pos_loss = self.GaussianScore(d, vdw_pair, pair_type, pair_mask, ligand_mask, pair_emb,
                                                          batched_data)
        return 0., gScore_loss, gScore_pos_loss


class Interformer(SBDD):
    def __init__(
        self,
        args
    ):
        super().__init__(args)
        hidden_dim = args['hidden_dim']
        num_heads = args['num_heads']
        attention_dropout_rate = args['attention_dropout_rate']
        dropout_rate = args['dropout_rate']
        intput_dropout_rate = args['intput_dropout_rate']
        ffn_dim = hidden_dim * args['ffn_scale']
        n_layers = args['n_layers']
        node_feat = args['node_feat_size']
        edge_feat = 0 if args[
            'energy_mode'] else args['edge_feat_size']  # disable edge feature while on predicting energy
        angle_feat_size = args['angle_feat_size']
        # RBF
        K = args['rbf_K']
        rbf_cutoff = args['rbf_cutoff']
        self.num_heads = num_heads
        
        self.task_type = args.get('task_type', 'energy')
        
        # Embedding
        self.complex_feat_layer = ComplexEncoder(node_feat, edge_feat, hidden_dim, num_heads, intput_dropout_rate,
                                                 angle_feat_size, K, rbf_cutoff)
        # Transformer-Encoder
        self.intra_encoder = nn.ModuleList(
            [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads, i)
             for i in range(n_layers)])
        self.inter_encoder = nn.ModuleList(
            [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads, i)
             for i in range(n_layers // 2)])
        ##
        # Affinity Output
        final_dim = hidden_dim
        # final FFN
        self.final_ln = nn.LayerNorm(final_dim)
        self.affinity_proj = PositionwiseFeedForward(final_dim, ffn_dim, d_out=1, dropout=dropout_rate)
        # 1. Pose-Selection & Affinity
        if self.pose_sel_mode:
            self.out_pose_sel_proj = PositionwiseFeedForward(final_dim, ffn_dim, d_out=1, dropout=dropout_rate)
        elif self.energy_mode:
            # 2. G-Score training
            ####
            # VinaScore
            energy_output_folder = args['energy_output_folder'] if 'energy_output_folder' in args else ''
            self.VinaScoreHead = VinaScoreHead(hidden_dim, self.hparams['args']['node_featurizer'], dropout_rate,
                                               energy_output_folder=energy_output_folder,
                                               edge_output_layer=args[
                                                   'edge_output_layer'] if 'edge_output_layer' in args else True)
            # Auxiliary Task
            self.AtomTypeHead = AtomHead(hidden_dim, 29 * 2, dropout_rate)
            ######
            # Logger for g_score training
            self.add_metric('g_score')
            self.add_metric('g_score_pos')  # for review only
            self.add_metric('atom_loss')
            # debug set task0's weight to 0
            self.loss_fn[0][1] = 0.  # affinity
            self.loss_fn[2][1] = 0.  # gscore_pos
            
        self.task_specific_norm = nn.ModuleDict({
            'energy': nn.LayerNorm(hidden_dim, eps=1e-6),
            'affinity': nn.LayerNorm(hidden_dim, eps=1e-6),
            'affinity_normal': nn.LayerNorm(hidden_dim, eps=1e-6)
        })
        
        self.task_specific_fuse = nn.ModuleDict({
            'energy': nn.Sequential(
                nn.LayerNorm(hidden_dim * 2, eps=1e-6),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=1e-6)
            ),
            'affinity': nn.Sequential(
                nn.LayerNorm(hidden_dim * 2, eps=1e-6),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=1e-6)
            ),
            'affinity_normal': nn.Sequential(
                nn.LayerNorm(hidden_dim * 2, eps=1e-6),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=1e-6)
            )
        })
        
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2, eps=1e-6),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Ablation Study
        if 'intra_encoder' in args:
            self.with_intra_encoder = args['intra_encoder']
        else:
            self.with_intra_encoder = True
        if 'inter_encoder' in args:
            self.with_inter_encoder = args['inter_encoder']
        else:
            self.with_inter_encoder = True
        if 'attention_mask' in args:
            self.with_attention_mask = args['attention_mask']
        else:
            self.with_attention_mask = True
        # multi-steps
        self.decoder_step = 1
        self.moba_adapter = MoBAInterformerAdapter(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            moba_chunk_size=args.get('moba_chunk_size', 64),
            moba_topk=args.get('moba_topk', 4)
        )
        self.moba_node_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.moba_edge_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp_node_fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2, eps=1e-6), 
            nn.Dropout(0.1),             
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-6)     
        )
        self.mlp_edge_fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2, eps=1e-6),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-6)
        )

    def forward(self, batched_data, perturb=None, istrain=True):
        # Step1. Preprocessing Embedding
        node_feats, intra_edge_feats, inter_attn_bias = self.complex_feat_layer(batched_data)
        if self.energy_mode:
            inter_edge = intra_edge_feats
        else:
            inter_edge = self.complex_feat_layer.edge_feat(batched_data['D'],
                                                           batched_data)
            # [b, h, n, n], Using Pose's inter distance or not
        ####
        if self.with_attention_mask:
            # Step2. Intra-Transformer, # Because the model can't distinguish different between two same type atom
            intra_node, intra_edge, intra_attn_bias = node_feats, intra_edge_feats, inter_attn_bias.clone()
            
            pair_mask = batched_data['pair_mask']
            if pair_mask.dim() == 4:
                pair_mask = pair_mask.squeeze(-1)  # [B, N, N]
            B, N1, N2 = intra_attn_bias.shape
            pair_mask_pad = F.pad(pair_mask, (1, 0, 1, 0), value=0)  # [B, N+1, N+1]
            intra_attn_bias = intra_attn_bias.masked_fill(pair_mask_pad.bool(), float('-inf'))
            
            # Intra-blocks
            if self.with_intra_encoder:
                for i in range(len(self.intra_encoder)):
                    if i % 1 == 0:
                        moba_node, moba_edge = self.moba_adapter(
                            intra_node, 
                            intra_edge, 
                            batched_data.get('attn_bias'),
                            task_type=self.task_type
                        )
                        moba_node = self.moba_node_norm(moba_node)
                        moba_edge = self.moba_edge_norm(moba_edge)
                        fused_node = torch.cat([moba_node, intra_node], dim=-1)
                        fused_edge = torch.cat([moba_edge, intra_edge], dim=-1)
                        intra_node = self.mlp_node_fuse(fused_node)
                        intra_edge = self.mlp_edge_fuse(fused_edge)
                    intra_node, intra_edge = self.intra_encoder[i](intra_node, intra_edge, intra_attn_bias)
            #####
            # Step3. Inter-Transformer
            # Inter-blocks
            inter_node, inter_edge = intra_node, intra_edge
            if self.with_inter_encoder:
                for j in range(self.decoder_step):
                    for i in range(len(self.inter_encoder)):
                        if i % 1 == 0:
                            moba_node, moba_edge = self.moba_adapter(
                                inter_node, 
                                inter_edge, 
                                batched_data.get('attn_bias'),
                                task_type=self.task_type
                            )
                            moba_node = self.moba_node_norm(moba_node)
                            moba_edge = self.moba_edge_norm(moba_edge)
                            fused_node = torch.cat([moba_node, inter_node], dim=-1)
                            fused_edge = torch.cat([moba_edge, inter_edge], dim=-1)
                            inter_node = self.mlp_node_fuse(fused_node)
                            inter_edge = self.mlp_edge_fuse(fused_edge)
                        inter_node, inter_edge = self.inter_encoder[i](inter_node, inter_edge, inter_attn_bias)
            else:
                # at least predict one time of the inter_edge, or it will produce Nan Tensor
                inter_node, inter_edge = self.intra_encoder[i](inter_node, inter_edge, inter_attn_bias)
        else:
            # Normal Graphformer
            intra_node, intra_edge, intra_attn_bias = node_feats, intra_edge_feats, inter_attn_bias.clone()
            for i in range(len(self.intra_encoder)):
                intra_node, intra_edge = self.intra_encoder[i](intra_node, intra_edge, intra_attn_bias)
            inter_node, inter_edge = intra_node, intra_edge
            
        if self.task_type in self.task_specific_norm:
            inter_node = self.task_specific_norm[self.task_type](inter_node)
            fused_node = torch.cat([inter_node, node_feats], dim=-1)
            gate = self.fusion_gate(fused_node)
            inter_node = inter_node * gate + node_feats * (1 - gate)
            inter_node = self.task_specific_fuse[self.task_type](fused_node)
            
        ######
        final = self.task_layer(inter_node, inter_edge, batched_data)
        return final

    def task_layer(self, output_node, output_edge, batched_data):
        # Step4. Pooling Graph-level output
        vn_node = self.final_ln(output_node[:, 0, :])
        affinity = self.affinity_proj(vn_node)  # affinity must be > 0.
        if bool(torch.isnan(torch.sum(affinity))):
            print("# [Interformer-forward] There is Nan output tensor.")
            affinity = torch.ones_like(affinity) * 1e-6
        ######
        final = [affinity]
        # Step5. Pose-Prediction-AUC-task
        if self.pose_sel_mode:
            pose_logits = self.out_pose_sel_proj(vn_node)
            final.append(pose_logits)
        elif self.energy_mode:
            # Step6, G-Score
            _, gscore_loss, gscore_pos_loss = self.VinaScoreHead(output_node, output_edge, batched_data)
            if torch.isnan(gscore_loss):
                print("# [Interformer-forward] G-Score loss is NaN, using default value")
                gscore_loss = torch.tensor(1.0, device=gscore_loss.device)
            if torch.isnan(gscore_pos_loss):
                print("# [Interformer-forward] G-Score pos loss is NaN, using default value")
                gscore_pos_loss = torch.tensor(1.0, device=gscore_pos_loss.device)
            final.append(gscore_loss)
            final.append(gscore_pos_loss)
            # Step7. Auxiliary tasks
            # Predict Atom Type
            atom_loss = self.AtomTypeHead(output_node, batched_data['node_type'])
            if torch.isnan(atom_loss):
                print("# [Interformer-forward] Atom loss is NaN, using default value")
                atom_loss = torch.tensor(1.0, device=atom_loss.device)
            final.append(atom_loss)
        # MARK: Final=[affinity, pose_selection, gscore_loss, gscore_pos_loss, atom_loss]
        return final

    @staticmethod
    def add_model_specific_args(args):
        parser = args.add_argument_group("Interformer")
        # dim
        parser.add_argument('--n_layers', type=int, default=6)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--ffn_scale', type=int, default=4)
        # drop
        parser.add_argument('--intput_dropout_rate', type=float, default=0.)
        parser.add_argument('--dropout_rate', type=float, default=0.2)
        parser.add_argument('--attention_dropout_rate', type=float, default=0.2)
        # special
        parser.add_argument('--attention_kd', type=int, default=0)
        # RBF
        parser.add_argument('--rbf_K', type=int, default=128)
        parser.add_argument('--rbf_cutoff', type=float, default=10.)
        # Ablation study
        parser.add_argument('--intra_encoder', type=bool, default=True)
        parser.add_argument('--inter_encoder', type=bool, default=True)
        parser.add_argument('--attention_mask', type=bool, default=True)
        parser.add_argument('--edge_output_layer', type=bool, default=True)

        # common used parameters
        SBDD.add_lr_parameters(parser)
        return args


if __name__ == '__main__':
    from utils.parser import get_args
    from data.data_process import GraphDataModule

    args = get_args()
    args['debug'] = True
    args['inference'] = True
    # dataset
    args['work_path'] = '/opt/home/revoli/data_worker/interformer/poses'
    args['data_path'] = '/opt/home/revoli/data_worker/interformer/train/100.csv'
    args['energy_mode'] = True

    dm = GraphDataModule(args, istrain=False)
    dm.setup()
    dataloader = dm.test_dataloader()
    # model
    model = Interformer(args)
    for items in dataloader:
        x = items[0]
        b, n = x['x'].size()[:2]
        print(x['target'])
        # forward
        y_hat = model(x)
        ######SE(3)-Test#####
        # added fake coords to input
        # R = rot(*torch.rand(3))
        # T = torch.randn(1, 3)
        # coords = torch.randn([b, n, 3], dtype=torch.float)
        # 1. test
        # x['xyz'] = coords
        # y_hat, x_new1 = model(x)
        # 2. test with rot & translation
        # x['xyz'] = coords @ R + T
        # y_hat2, x_new2 = model(x)
        # # 3. verified SE3
        # print("Coords1", x_new1[0, :10, :])
        # print("Coords2", x_new2[0, :10, :])
        # rot_T_x_new1 = x_new1 @ R + T
        # print("R + T -> x_new1", rot_T_x_new1[0, :10, :])
        break

    print("Test DONE.")