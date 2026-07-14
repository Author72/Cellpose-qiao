"""Test-time adaptation utilities for Cellpose flow predictions.

The adaptation is deliberately restricted to the final readout layer.  It uses
the fact that a valid vector field must transform predictably when an input is
flipped, so it does not require test-image annotations.
"""

from copy import deepcopy
import logging
import math

import torch
import torch.nn.functional as F


tta_logger = logging.getLogger(__name__)


def _undo_flip(prediction, dim):
    """Map a prediction from a flipped image back to the original coordinates."""
    prediction = torch.flip(prediction, dims=(dim,))
    prediction = prediction.clone()
    # Channel 0 is the Y component and channel 1 is the X component.
    prediction[:, 0 if dim == -2 else 1].neg_()
    return prediction


def flow_equivariance_loss(prediction, flipped_prediction, dim, confidence=0.5):
    """Return flip-equivariance loss for Cellpose flow and cellprob outputs.

    A confidence mask, computed from the unflipped cell probability, keeps
    background flow vectors from dominating the self-supervised objective.
    """
    flipped_prediction = _undo_flip(flipped_prediction, dim)
    mask = torch.sigmoid(prediction[:, 2:3]).detach() >= confidence
    if not torch.any(mask):
        # Empty images still need a well-defined, finite loss.
        mask = torch.ones_like(mask, dtype=torch.bool)
    flow_error = F.smooth_l1_loss(prediction[:, :2], flipped_prediction[:, :2],
                                  reduction="none")
    prob_error = F.mse_loss(torch.sigmoid(prediction[:, 2:3]),
                            torch.sigmoid(flipped_prediction[:, 2:3]),
                            reduction="none")
    return ((flow_error * mask).sum() / (mask.sum() * 2) +
            (prob_error * mask).sum() / mask.sum())


def _patches(images, patch_size, max_patches):
    """Make deterministic, padded crops of NHWC normalized test images."""
    images = torch.as_tensor(images).permute(0, 3, 1, 2)
    _, _, height, width = images.shape
    pad_y, pad_x = max(0, patch_size - height), max(0, patch_size - width)
    if pad_y or pad_x:
        images = F.pad(images, (0, pad_x, 0, pad_y))
    _, _, height, width = images.shape

    grid_size = math.ceil(math.sqrt(max_patches))
    ys = torch.linspace(0, height - patch_size, steps=grid_size).round().long()
    xs = torch.linspace(0, width - patch_size, steps=grid_size).round().long()
    locations = [(y, x) for y in ys for x in xs][:max_patches]
    return torch.cat([images[i:i + 1, :, y:y + patch_size, x:x + patch_size]
                      for i in range(images.shape[0])
                      for y, x in locations], dim=0)


def adapt_flow_head(net, images, steps=8, lr=1e-4, patch_size=256,
                    max_patches=1, confidence=0.5, anchor_weight=0.05):
    """Adapt ``net.out`` to one test image and return its original state.

    The caller must restore the returned state after running the actual
    inference. Only the flow readout is optimized; all encoder parameters stay
    frozen.  The small anchor to the initial prediction prevents an
    under-constrained consistency loss from changing a well calibrated model.
    """
    if steps < 0:
        raise ValueError("tta_steps must be non-negative")
    if steps == 0:
        return None
    if not hasattr(net, "out"):
        raise ValueError("flow test-time adaptation requires a network with an 'out' readout")
    if max_patches < 1 or patch_size < 1:
        raise ValueError("tta_patch_size and tta_batch_size must be positive")
    if lr <= 0:
        raise ValueError("tta_lr must be positive")
    if not 0 <= confidence <= 1:
        raise ValueError("tta_confidence must be between 0 and 1")

    original_state = deepcopy(net.out.state_dict())
    requires_grad = {name: parameter.requires_grad for name, parameter in net.named_parameters()}
    try:
        net.eval()
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        for parameter in net.out.parameters():
            parameter.requires_grad_(True)

        patches = _patches(images, patch_size, max_patches).to(net.device, dtype=net.dtype)
        optimizer = torch.optim.Adam(net.out.parameters(), lr=lr)
        with torch.no_grad():
            teacher = net(patches)[0].detach()

        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = net(patches)[0]
            horizontal = net(torch.flip(patches, dims=(-1,)))[0]
            vertical = net(torch.flip(patches, dims=(-2,)))[0]
            equivariance = (flow_equivariance_loss(prediction, horizontal, -1, confidence) +
                             flow_equivariance_loss(prediction, vertical, -2, confidence)) / 2
            anchor = F.smooth_l1_loss(prediction, teacher)
            loss = equivariance + anchor_weight * anchor
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss during flow test-time adaptation")
            loss.backward()
            optimizer.step()
        tta_logger.info("adapted flow readout for %d step(s)", steps)
        return original_state
    except Exception:
        net.out.load_state_dict(original_state)
        raise
    finally:
        for name, parameter in net.named_parameters():
            parameter.requires_grad_(requires_grad[name])


def restore_flow_head(net, state):
    """Restore the transient flow-readout state returned by ``adapt_flow_head``."""
    if state is not None:
        net.out.load_state_dict(state)
