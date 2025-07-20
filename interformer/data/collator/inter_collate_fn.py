import torch
from interformer.data.collator.collate_fn import pad_1D, pad_2D, pad_edge_type_unsqueeze, get_attn_bias, \
    get_pair_mask
from collections import defaultdict

MAX_NODES = 310  

def truncate(x, max_n=MAX_NODES):
    if x.shape[0] > max_n:
        if x.dim() == 2:
            return x[:max_n, :max_n]
        elif x.dim() == 3:
            return x[:max_n, :max_n, ...]
        else:
            return x[:max_n]
    return x

def pad_to(x, max_n):
    # x: [n, n] or [n, n, ...]
    shape = list(x.shape)
    pad_shape = [max_n, max_n] + shape[2:]
    out = torch.zeros(*pad_shape, dtype=x.dtype, device=x.device)
    n = min(x.shape[0], max_n)
    out[:n, :n, ...] = x[:n, :n, ...]
    return out

def get_ligand_mask(D, ligand_len):
    ligand_mask = torch.zeros_like(D)
    for i, l in enumerate(ligand_len):
        ligand_mask[i, :l, :l] = 1.
    ### convert into bool
    ligand_mask = ligand_mask.bool()
    return ligand_mask


def debug_collate(uff_xyz, targets):
    from rdkit import Chem
    from feats.third_rd_lib import rdkit_write_with_new_coords
    # Checked, the uff_ligand will be placed in the center of pocket
    xyz = uff_xyz[0]
    t = targets[0]
    sdf = Chem.SDMolSupplier(f'/opt/home/revoli/data_worker/v2019-docking/ligands/{t}_docked.sdf')[0]
    rdkit_write_with_new_coords(sdf, xyz, f'out/reset_mass_{t}.sdf')


def uff_add_pocket_xyz(uff_xyz, gt_xyz, ligand_len):
    assert uff_xyz.shape == gt_xyz.shape
    # using gt-ligand-conformer
    # uff_xyz = gt_xyz.clone()
    ####
    for i, l_len in enumerate(ligand_len):
        uff_xyz[i, l_len:] = gt_xyz[i, l_len:]
    return uff_xyz


def convert2deltaG(y):
    ic50 = 10 ** -y
    deltaG = 0.59 * torch.log(ic50)
    return deltaG


def unpack_dict(X):
    batch_dict = defaultdict(list)
    for row_dict in X:
        for key in row_dict:
            batch_dict[key].append(row_dict[key])
    return batch_dict


def interformer_collate_fn(data, energy_mode=False):
    names, X, id = list(zip(*data))
    X = unpack_dict(X)
    Y = torch.tensor(X['pIC50']).view(-1, 1)
    # Collecting from batch
    node_lens = [x.shape[0] for x in X['ndata']]
    max_N = min(max(node_lens), MAX_NODES)
    ligand_len = torch.tensor([min(x[0], max_N) for x in X['lens']]).long()
    pocket_len = torch.tensor([min(x[1], max_N) for x in X['lens']]).long()
    for key in ['ndata', 'gt_xyz']:
        if key in X:
            X[key] = [truncate(item, max_N) for item in X[key]]
    if 'edata' in X:
        X['edata'] = [pad_to(truncate(item, max_N), max_N) for item in X['edata']]
    # Packing
    ndata = torch.nn.utils.rnn.pad_sequence(X['ndata'], batch_first=True).float()
    B, max_N, node_feat_dim = ndata.size()
    node_type = ndata[:, :, 0].long() if ndata.size(-1) > 0 else torch.zeros((B, max_N), dtype=torch.long)
    edata = None
    if 'edata' in X and X['edata'][0] is not None:
        edata = torch.stack(X['edata'])  # [B, max_N, max_N, ...]
    gt_xyz = torch.cat([pad_1D(x, max_N) for x in X['gt_xyz']]).float()
    D_mask = (gt_xyz.sum(dim=-1) == 0.)[:, None, :].repeat(1, max_N, 1)
    D = torch.cdist(gt_xyz, gt_xyz)
    D = torch.where(D_mask, torch.tensor(-1.), D)
    attn_bias = get_attn_bias(D, ndata, dist_cut_off=True)
    D = D.unsqueeze(-1)
    pair_mask = get_pair_mask(D, ligand_len)
    ligand_mask = get_ligand_mask(D, ligand_len)
    pocket_mask = ~(ligand_mask + pair_mask)
    if energy_mode:
        xyz = X['uff_xyz']
        attn_bias[:, 1:, 1:][pair_mask.squeeze(-1)] = float(0.)
    else:
        xyz = X['gt_xyz']
    max_N = D.size(1)
    uff_xyz = torch.cat([pad_1D(x, max_N) for x in xyz]).float()
    uff_xyz = uff_add_pocket_xyz(uff_xyz, gt_xyz, ligand_len)
    intra_D = torch.cdist(uff_xyz, uff_xyz)
    intra_D = torch.where(D_mask, torch.tensor(-1.), intra_D).unsqueeze(-1)
    intra_D = intra_D * ~pair_mask
    batch_data = {
        'node_feat': ndata,
        'node_type': node_type,
        'pos': gt_xyz,
        'edata': edata,
        'D': D,
        'ligand_len': ligand_len,
        'pocket_len': pocket_len,
        'pair_mask': pair_mask,
        'attn_bias': attn_bias,
        'ligand_mask': ligand_mask,
        'pocket_mask': pocket_mask,
        'target': X['target'],
        'intra_D': intra_D,
        'y': Y,
        'pdb': X['pdb']
    }
    return batch_data, Y, X['target'], id
