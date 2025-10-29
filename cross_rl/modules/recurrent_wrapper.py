from __future__ import annotations

import torch
import torch.nn as nn

def recurrent_wrapper(
        module: nn.Module,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        seq_dim: bool,
):
    if isinstance(module, nn.RNNBase):
        x, hidden = x

        if seq_dim:
            return module(x, hidden)
        else:
            n_seq = x.size(0)
            out, hidden = module(x.unsqueeze(0), hidden)
            return out.squeeze(0), hidden

    elif seq_dim:
        n_seq = x.size(0)
        return module(x.flatten(0, 1)).unflatten(0, (n_seq, x.size(1)))
    else:
        return module(x)
