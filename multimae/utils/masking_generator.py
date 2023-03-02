# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------
import numpy as np
from itertools import product


class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask  # [196]


def non_overlap_mask_gen(domains, num_tokens, patch_dims):
    nh, nw = patch_dims
    assert num_tokens <= nh * nw, "num_tokens should be smaller than patch_dims"
    
    h_idx = np.arange(nh)
    w_idx = np.arange(nw)
    pos_idx = np.array(list(product(h_idx, w_idx)))
    random_pos_idx = np.random.permutation(pos_idx)
    
    i = len(domains)
    num_masks = []
    num_assigned_tokens = 0
    while i > 1:
        num = np.random.randint(1, num_tokens - num_assigned_tokens - i)
        num_assigned_tokens += num
        num_masks.append(num)
        i -= 1
    
    num_masks.append(num_tokens - sum(num_masks))
    
    assert sum(num_masks) == num_tokens, "num_tokens should be equal to sum of num_masks"
    assert len(num_masks) == len(domains), "num_masks should be equal to len(domains)"
    assert (np.array(num_masks) > 0).all(), "num_masks should be greater than 0"
    
    mask_pos_idx = [random_pos_idx[x:x + num] for x, num in zip(np.cumsum([0] + num_masks[:-1]), num_masks)]

    # generate masks with shape [num_masks, nh, nw] and pos_idx for mask value 1
    masks = []
    for pos_idx in mask_pos_idx:
        mask = np.ones((nh, nw))
        mask[pos_idx[:, 0], pos_idx[:, 1]] = 0
        masks.append(mask)
        
    return {d:m for d, m in zip(domains, masks)}