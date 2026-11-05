"""
src/baselines/logit_ensemble.py

Baseline: average the two parents' softmax outputs (protocol Sec. 7).

IMPORTANT FRAMING (Sec. 7 is explicit about this): the logit ensemble is
"an accuracy upper reference, not a fusion competitor, since it costs two
forward passes." It keeps BOTH parents in memory and runs both at
inference, so it is not solving the same problem as weight-space fusion.
It belongs in the results table as a ceiling to contextualize how much
accuracy fusion gives up in exchange for collapsing to a single model --
it should not be described as a method MSOT "beats" or "loses to."
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitEnsemble(nn.Module):
    """
    Wraps two trained parents; forward() returns log-probabilities of the
    averaged softmax, so it can be dropped into the same evaluate() loop
    as a single model.

    Averaging is done in PROBABILITY space (softmax then mean), not logit
    space, since the two parents' logits can sit on different scales and
    averaging those directly would silently weight the more confident
    model higher. Returning log() of the averaged probability keeps the
    output compatible with CrossEntropyLoss-style evaluation, which
    expects unnormalized scores and applies its own log_softmax --
    log(p) is a valid such score and is invariant under that extra
    normalization since the probabilities already sum to 1.
    """

    def __init__(self, parent_A, parent_B, weight_A: float = 0.5):
        super().__init__()
        self.parent_A = parent_A
        self.parent_B = parent_B
        self.weight_A = weight_A

    def forward(self, x):
        p_A = F.softmax(self.parent_A(x), dim=-1)
        p_B = F.softmax(self.parent_B(x), dim=-1)
        p = self.weight_A * p_A + (1.0 - self.weight_A) * p_B
        return torch.log(p.clamp_min(1e-12))


def build_logit_ensemble(parent_A, parent_B, weight_A: float = 0.5) -> LogitEnsemble:
    ens = LogitEnsemble(parent_A, parent_B, weight_A=weight_A)
    ens.eval()
    return ens
