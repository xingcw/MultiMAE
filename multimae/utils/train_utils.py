import torch
from einops import rearrange


def normalize_depth(depth):
    # Flatten depth and remove bottom and top 10% of values
    trunc_depth = torch.sort(rearrange(depth, 'b c h w -> b (c h w)'), dim=1)[0]
    trunc_depth = trunc_depth[:,int(0.1 * trunc_depth.shape[1]): int(0.9 * trunc_depth.shape[1])]
    depth = (depth - trunc_depth.mean(dim=1)[:,None,None,None]) / torch.sqrt(trunc_depth.var(dim=1)[:,None,None,None] + 1e-6)
    
    return depth