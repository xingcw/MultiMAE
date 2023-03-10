# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------
import torch

from einops import rearrange
from itertools import product
from typing import Dict, List, Union
from torch.distributions.dirichlet import Dirichlet


class MaskGenerator:
    def __init__(self, 
                 input_tokens: Dict[str, torch.Tensor], 
                 num_encoded_tokens: int, 
                 mask_type : str = "random"):
        """random sampling of masks for input tokens

        :param input_tokens: Dictionary of tensors to sample num_encoded_tokens from
        :param num_encoded_tokens: Number of tokens to select
        :param mask_type: Type of masking to use. Can be "random", "non-overlap", "gate-oriented", "dirichlet".
        """
        self.batch_size = list(input_tokens.values())[0].shape[0]
        self.device = list(input_tokens.values())[0].device
        
        self.num_all_tokens = sum([t.shape[1] for t in input_tokens.values()])
        self.num_tasks = len(input_tokens)
        self.num_encoded_tokens = num_encoded_tokens
        self.input_tokens = input_tokens
        self.input_domains = list(input_tokens.keys())
        self.mask_type = mask_type

    def __repr__(self):
        repr_str = "Mask {}: input domains: {}, total patches {}, mask patches {}".format(
            self.mask_type, "-".join(self.input_domains), self.num_all_tokens, self.num_encoded_tokens
        )
        return repr_str

    def __call__(self, **kwargs):
        
        if self.mask_type == "random":
            mask = self.random_mask_gen(**kwargs)
        elif self.mask_type == "non-overlap":
            mask = self.non_overlap_mask_gen(**kwargs)
        elif self.mask_type == "gate-oriented":
            mask = self.object_oriented_mask_gen(**kwargs)
        elif self.mask_type == "dirichlet":
            mask = self.dirichlet_mask_gen(**kwargs)
        else:
            raise KeyError
        
        return mask  # [196]
    
    @staticmethod
    def rand_mask(num_encoded_tokens, num_all_tokens, batch_size=1, device="cuda"):
        if isinstance(num_encoded_tokens, torch.Tensor):
            pos_idx = torch.arange(num_all_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
            masks = torch.where(pos_idx < num_encoded_tokens, 0, 1)
        else:
            masks = torch.hstack([
                torch.ones(batch_size, num_all_tokens - num_encoded_tokens, device=device),
                torch.zeros(batch_size, num_encoded_tokens, device=device)
            ])
        noise = torch.rand(batch_size, num_all_tokens, device=device)  # noise in [0, 1]
        ids_arange_shuffle = torch.argsort(noise, dim=1)               # ascend: small is keep, large is remove
        masks = torch.gather(masks, dim=1, index=ids_arange_shuffle)
        return masks
    
    def random_mask_gen(self):
        masks = self.rand_mask(self.num_encoded_tokens, self.num_all_tokens, self.batch_size, self.device)
        ids_shuffle = torch.argsort(masks, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :self.num_encoded_tokens]

        # split masks to each domain
        num_tokens_per_task = [task_tokens.shape[1] for task_tokens in self.input_tokens.values()]
        task_masks = torch.split(masks, num_tokens_per_task, dim=1)
        task_masks = {domain: mask for domain, mask in zip(self.input_tokens.keys(), task_masks)}
        
        return task_masks, ids_keep, ids_restore

    def non_overlap_mask_gen(self, patch_dims=(14, 14)):
        
        if self.num_tasks == 1:
            return self.random_mask_gen()
        
        # position idxs
        nh, nw = patch_dims
        num_patches = nh * nw
        num_encoded_tokens = min(self.num_encoded_tokens, num_patches)
        pos_idx = torch.arange(self.num_encoded_tokens, device=self.device).unsqueeze(0).expand(self.batch_size, -1)
        
        # assign random number of encoded tokens to each domain
        # sum up to num_encoded_tokens <= num_tokens_per_image
        token_splits = torch.randint(0, num_encoded_tokens, (self.batch_size, self.num_tasks - 1), device=self.device)
        token_splits, _ = torch.sort(token_splits, dim=1)
        token_splits = torch.cat([token_splits, 
                                  torch.tensor(num_encoded_tokens, device=self.device).unsqueeze(0).expand(self.batch_size, 1)], dim=1)
        
        # Use noise to shuffle arange
        noise = torch.rand(self.batch_size, num_patches, device=self.device)  # noise in [0, 1]
        ids_arange_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        random_pos_idx = torch.gather(pos_idx, dim=1, index=ids_arange_shuffle)

        # split tokens to each domain
        task_masks = []
        for i in range(self.num_tasks):
            mask_left = torch.where(random_pos_idx <= token_splits[:, i].unsqueeze(1), 0, 1)
            if i > 0:
                mask_right = torch.where(random_pos_idx > token_splits[:, i - 1].unsqueeze(1), 0, 1)
                mask = torch.logical_or(mask_left, mask_right) * 1
            else:
                mask = mask_left
            task_masks.append(mask)
            
        # concatenate masks and indexing
        masks = torch.cat(task_masks, dim=1)
        # print(torch.sum((masks == 0), dim=1))
        ids_shuffle = torch.argsort(masks, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :self.num_encoded_tokens]

        # split masks to each domain
        num_tokens_per_task = [task_tokens.shape[1] for task_tokens in self.input_tokens.values()]
        task_masks = torch.split(masks, num_tokens_per_task, dim=1)
        task_masks = {domain: mask for domain, mask in zip(self.input_tokens.keys(), task_masks)}
        
        return task_masks, ids_keep, ids_restore
    
    def object_oriented_mask_gen(self, 
                                 inputs, 
                                 patch_dims=(14, 14), 
                                 patch_size=(16, 16), 
                                 px_thresh=0.1, 
                                 semseg_stride=4, 
                                 masked_rgb_gate_only=False):
        
        if "semseg" not in self.input_domains:
            return self.random_mask_gen()
        
        nh, nw = patch_dims
        ph, pw = patch_size[0] // semseg_stride, patch_size[1] // semseg_stride
        num_px_thresh = int(px_thresh * ph * pw)
        # mask for gate from semseg
        semseg = inputs["semseg"].detach().clone()
        fine_masks = torch.zeros_like(semseg, device=self.device)
        fine_masks[semseg == 3] = 1              # 3 -> "gate"
        fine_masks = rearrange(fine_masks, "B (nh h) (nw w) -> B (nh nw) (h w)", B=self.batch_size, nh=nh, nw=nw)
        rgb_mask = (torch.sum(fine_masks, dim=2) > num_px_thresh) * 1
        
        # ensure number of encoded rgb tokens less than num_encoded_tokens
        if not masked_rgb_gate_only:
            num_addition_encoded_tokens = fine_masks.shape[1] - self.num_encoded_tokens
            addition_masks = self.rand_mask(num_addition_encoded_tokens, fine_masks.shape[1], self.batch_size, self.device)
            rgb_mask = torch.logical_or(rgb_mask, addition_masks) * 1      
        
        # sample masks for other domains        
        num_encoded_tokens = torch.ones(self.batch_size, device=self.device) * self.num_encoded_tokens - (rgb_mask == 0).sum(dim=1)
        num_encoded_tokens = torch.maximum(num_encoded_tokens, torch.zeros_like(num_encoded_tokens)).unsqueeze(-1)
        num_all_tokens = sum([v.shape[1] for k, v in self.input_tokens.items() if k != "rgb"])
        rand_masks = self.rand_mask(num_encoded_tokens, num_all_tokens, batch_size=self.batch_size, device=self.device)
        
        # concatenate masks and indexing
        masks = torch.cat([rgb_mask, rand_masks], dim=1)
        ids_shuffle = torch.argsort(masks, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :self.num_encoded_tokens]

        # split masks to each domain
        num_tokens_per_task = [task_tokens.shape[1] for task_tokens in self.input_tokens.values()]
        task_masks = torch.split(masks, num_tokens_per_task, dim=1)
        task_masks = {domain: mask for domain, mask in zip(self.input_tokens.keys(), task_masks)}
        
        return task_masks, ids_keep, ids_restore
    
    def sample_alphas(self, B: int, n_tasks: int, alphas: float = 1.0, eps: float = 1e-5):
        """
        Sample alphas for Dirichlet sampling such that tasks are first uniformly chosen and then Dirichlet sampling
        is performed over the chosen ones.

        :param B: Batch size
        :param n_tasks: Number of input tasks
        :param alphas: Float or list to multiply task choices {0,1} by
        :param eps: Small constant since Dirichlet alphas need to be positive
        """
        valid_task_choices = torch.Tensor([list(i) for i in product([0, 1], repeat=n_tasks)][1:])
        rand_per_sample_choice = torch.randint(0, len(valid_task_choices), (B,))
        alphas_tensor = torch.index_select(valid_task_choices, 0, rand_per_sample_choice)
        alphas_tensor = alphas_tensor * torch.tensor(alphas) + eps
        return alphas_tensor
    
    def sample_per_task(self, 
                        alphas: Union[float, List[float]] = 1.0,
                        sample_tasks_uniformly: bool = False):
        
        alphas = [alphas] * len(self.input_tokens) if isinstance(alphas, float) else alphas
        if sample_tasks_uniformly:
            alphas = self.sample_alphas(self.batch_size, len(self.input_tokens), alphas=alphas)
            task_sampling_dist = Dirichlet(alphas).sample().to(self.device)
        else:
            task_sampling_dist = Dirichlet(torch.Tensor(alphas)).sample((self.batch_size,)).to(self.device)

        samples_per_task = (task_sampling_dist * self.num_encoded_tokens).round().long()
        return samples_per_task

    def dirichlet_mask_gen(self,
                            alphas: Union[float, List[float]] = 1.0,
                            sample_tasks_uniformly: bool = False) :
        """
        Sample a total of num_encoded_tokens from different tasks using Dirichlet sampling.

        :param alphas: Dirichlet distribution parameter alpha. Lower alpha = harder,
            less uniform sampling. Can be float or list of floats.
        :param sample_tasks_uniformly: Set to True to first sample 1-n_tasks uniformly at random
            for each sample in the batch. Dirichlet sampling is then done over selected subsets.
        """
        B = self.batch_size
        samples_per_task = self.sample_per_task(alphas, sample_tasks_uniformly)

        task_masks = []
        num_tokens_per_task = [task_tokens.shape[1] for task_tokens in self.input_tokens.values()]
        for i, num_tokens in enumerate(num_tokens_per_task):
            # Use noise to shuffle arange
            noise = torch.rand(B, num_tokens, device=self.device)  # noise in [0, 1]
            ids_arange_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            mask = torch.arange(num_tokens, device=self.device).unsqueeze(0).expand(B, -1)
            mask = torch.gather(mask, dim=1, index=ids_arange_shuffle)
            # 0 is keep (unmasked), 1 is remove (masked)
            mask = torch.where(mask < samples_per_task[:, i].unsqueeze(1), 0, 1)
            task_masks.append(mask)
            
        mask_all = torch.cat(task_masks, dim=1)
        ids_shuffle = torch.argsort(mask_all + torch.rand_like(mask_all.float()), dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :self.num_encoded_tokens]

        # Update binary mask to adjust for task rounding
        mask_all = torch.ones_like(mask_all)
        mask_all[:, :self.num_encoded_tokens] = 0
        # Unshuffle to get the binary mask
        mask_all = torch.gather(mask_all, dim=1, index=ids_restore)
        # Split to get task masks
        task_masks = torch.split(mask_all, num_tokens_per_task, dim=1)
        # Convert to dict
        task_masks = {domain: mask for domain, mask in zip(self.input_tokens.keys(), task_masks)}

        return task_masks, ids_keep, ids_restore