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
    def __init__(self, input_size, input_domains, mask_ratio, mask_type="random"):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.nh, self.nw = input_size
        self.num_patches = self.nh * self.nw * len(input_domains)
        mask_ratio = min(max(mask_ratio, 0), 1)
        self.num_masks = int(mask_ratio * self.num_patches)
        self.input_domains = input_domains
        self.mask_type = mask_type

    def __repr__(self):
        repr_str = "Mask {}: input domains: {}, total patches {}, mask patches {}".format(
            self.mask_type, "-".join(self.input_domains), self.num_patches, self.num_masks
        )
        return repr_str

    def __call__(self):
        
        if self.mask_type == "random":
            mask = self.random_mask_gen()
        elif self.mask_type == "non-overlap":
            mask = self.non_overlap_mask_gen()
        elif self.mask_type == "gate-oriented":
            mask = self.object_oriented_mask_gen()
        else:
            raise KeyError
        
        return mask  # [196]
    
    def random_mask_gen(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_masks),
            np.ones(self.num_masks),
        ])
        np.random.shuffle(mask)
        masks = np.reshape(mask, (len(self.input_domains), -1))
        return {d: m for d, m in zip(self.input_domains, masks)}

    def non_overlap_mask_gen(self):
        
        h_idx = np.arange(self.nh)
        w_idx = np.arange(self.nw)
        pos_idx = np.array(list(product(h_idx, w_idx)))
        random_pos_idx = np.random.permutation(pos_idx)
        
        i = len(self.input_domains)
        num_masks = []
        num_assigned_tokens = 0
        while i > 1:
            num = np.random.randint(1, self.num_patches - num_assigned_tokens - i)
            num_assigned_tokens += num
            num_masks.append(num)
            i -= 1
        
        num_masks.append(self.num_patches - sum(num_masks))
        
        assert sum(num_masks) == self.num_patches, "num_tokens should be equal to sum of num_masks"
        assert len(num_masks) == len(self.input_domains), "num_masks should be equal to len(domains)"
        assert (np.array(num_masks) > 0).all(), "num_masks should be greater than 0"
        
        mask_pos_idx = [random_pos_idx[x:x + num] for x, num in zip(np.cumsum([0] + num_masks[:-1]), num_masks)]

        # generate masks with shape [num_masks, nh, nw] and pos_idx for mask value 1
        masks = []
        for pos_idx in mask_pos_idx:
            mask = np.ones((self.nh, self.nw))
            mask[pos_idx[:, 0], pos_idx[:, 1]] = 0
            masks.append(mask)
            
        return {d:m for d, m in zip(self.input_domains, masks)}
    
    def object_oriented_mask_gen(self):
        
        if "semseg" not in self.input_domains:
            return self.random_mask_gen()
        