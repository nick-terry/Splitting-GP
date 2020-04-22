# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 20:59:01 2020

@author: pnter
"""


import gpytorch
import torch
from UtilityFunctions import haversine

def default_postprocess_script(x):
    return x

class Distance(torch.nn.Module):
    def __init__(self, postprocess_script=default_postprocess_script):
        super().__init__()
        self._postprocess = postprocess_script

    def _sq_dist(self, x1, x2, postprocess, x1_eq_x2=False):
        # TODO: use torch squared cdist once implemented: https://github.com/pytorch/pytorch/pull/25799
        
        res = haversine(x1,x2)

        if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
            res.diagonal(dim1=-2, dim2=-1).fill_(0)

        # Zero out negative values
        res.clamp_min_(0)
        return self._postprocess(res) if postprocess else res

    def _dist(self, x1, x2, postprocess, x1_eq_x2=False):
        # TODO: use torch cdist once implementation is improved: https://github.com/pytorch/pytorch/pull/25799
        res = self._sq_dist(x1, x2, postprocess=False, x1_eq_x2=x1_eq_x2)
        res = res.clamp_min_(1e-30).sqrt_()
        return self._postprocess(res) if postprocess else res

class HaversineRBFKernel(gpytorch.kernels.RBFKernel):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
    def covar_dist(
        self,
        x1,
        x2,
        diag=False,
        last_dim_is_batch=False,
        square_dist=False,
        dist_postprocess_func=default_postprocess_script,
        postprocess=True,
        **params,
    ):
        r"""
        This is a helper method for computing the Euclidean distance between
        all pairs of points in x1 and x2.

        Args:
            :attr:`x1` (Tensor `n x d` or `b1 x ... x bk x n x d`):
                First set of data.
            :attr:`x2` (Tensor `m x d` or `b1 x ... x bk x m x d`):
                Second set of data.
            :attr:`diag` (bool):
                Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`.
            :attr:`last_dim_is_batch` (tuple, optional):
                Is the last dimension of the data a batch dimension or not?
            :attr:`square_dist` (bool):
                Should we square the distance matrix before returning?

        Returns:
            (:class:`Tensor`, :class:`Tensor) corresponding to the distance matrix between `x1` and `x2`.
            The shape depends on the kernel's mode
            * `diag=False`
            * `diag=False` and `last_dim_is_batch=True`: (`b x d x n x n`)
            * `diag=True`
            * `diag=True` and `last_dim_is_batch=True`: (`b x d x n`)
        """
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        x1_eq_x2 = torch.equal(x1, x2)

        # torch scripts expect tensors
        postprocess = torch.tensor(postprocess)

        res = None

        # Cache the Distance object or else JIT will recompile every time
        if not self.distance_module or self.distance_module._postprocess != dist_postprocess_func:
            self.distance_module = Distance(dist_postprocess_func)

        if diag:
            # Special case the diagonal because we can return all zeros most of the time.
            if x1_eq_x2:
                res = torch.zeros(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
                if postprocess:
                    res = dist_postprocess_func(res)
                return res
            else:
                res = haversine(x1,x2)
                if square_dist:
                    res = res.pow(2)
            if postprocess:
                res = dist_postprocess_func(res)
            return res

        elif square_dist:
            res = self.distance_module._sq_dist(x1, x2, postprocess, x1_eq_x2)
        else:
            res = self.distance_module._dist(x1, x2, postprocess, x1_eq_x2)

        return res