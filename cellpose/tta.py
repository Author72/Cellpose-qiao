"""Region-aware, test-time refinement of Cellpose flow fields.

This module intentionally adapts an image's predicted flow field, not the
network weights. Candidate regions are connected components of cellprob; each
region has its own endpoint-compactness loss, preventing the all-to-one-centre
failure of a global compactness objective.
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


tta_logger = logging.getLogger(__name__)


def _candidate_regions(cellprob, threshold, min_size, max_points):
    """Return deterministic pixel samples for valid cellprob components."""
    labels, nregions = ndimage.label(np.asarray(cellprob) > threshold)
    regions = []
    for label in range(1, nregions + 1):
        points = np.argwhere(labels == label)
        if len(points) < min_size:
            continue
        if len(points) > max_points:
            # Evenly-spaced deterministic sampling makes TTA reproducible.
            points = points[np.linspace(0, len(points) - 1, max_points).astype(int)]
        regions.append(points)
    return regions


def _follow_flow(flow, points, niter):
    """Differentiably follow 2D Cellpose flow from ``points`` (YX order)."""
    height, width = flow.shape[-2:]
    positions = points.to(dtype=flow.dtype)
    for _ in range(niter):
        grid_x = positions[:, 1] * (2.0 / max(width - 1, 1)) - 1.0
        grid_y = positions[:, 0] * (2.0 / max(height - 1, 1)) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1).view(1, -1, 1, 2)
        # Cellpose dynamics uses dP / 5. Use border values outside the image.
        displacement = F.grid_sample(flow.unsqueeze(0) / 5.0, grid,
                                     align_corners=True, padding_mode="border")
        positions = positions + displacement[0, :, :, 0].transpose(0, 1)
        positions = torch.stack((positions[:, 0].clamp(0, height - 1),
                                 positions[:, 1].clamp(0, width - 1)), dim=1)
    return positions


def _flow_smoothness(flow, foreground):
    """Squared finite differences, evaluated only inside candidate regions."""
    vertical = (flow[:, 1:] - flow[:, :-1]).square().mean(dim=0)
    horizontal = (flow[:, :, 1:] - flow[:, :, :-1]).square().mean(dim=0)
    vmask = foreground[1:] & foreground[:-1]
    hmask = foreground[:, 1:] & foreground[:, :-1]
    losses = []
    if torch.any(vmask):
        losses.append(vertical[vmask].mean())
    if torch.any(hmask):
        losses.append(horizontal[hmask].mean())
    return torch.stack(losses).mean() if losses else flow.new_zeros(())


def refine_flow_regions(dP, cellprob, *, steps=8, lr=5e-2, niter=32,
                        region_threshold=0.0, min_region_size=50,
                        max_points=256, flow_weight=0.1,
                        smooth_weight=0.01, device=None):
    """Refine a 2D ``[2, Y, X]`` flow field using region-aware TTA.

    The optimized objective is
    ``L_compact + flow_weight * L_anchor + smooth_weight * L_smooth``, where
    ``L_compact`` is the mean endpoint variance *within each candidate region*.
    The returned array is independent of model weights and is safe to use only
    for the current image's post-processing.
    """
    if steps < 0 or niter < 1 or min_region_size < 1 or max_points < 1:
        raise ValueError("invalid Region-aware Flow-TTA settings")
    if steps == 0:
        return np.asarray(dP, dtype=np.float32)
    if lr <= 0 or flow_weight < 0 or smooth_weight < 0:
        raise ValueError("tta_lr must be positive and regularization weights non-negative")
    if np.asarray(dP).shape[0] != 2:
        raise ValueError("Region-aware Flow-TTA supports 2D flow fields with shape [2, Y, X]")

    regions_np = _candidate_regions(cellprob, region_threshold, min_region_size, max_points)
    if not regions_np:
        tta_logger.info("skipping Region-aware Flow-TTA: no candidate cellprob regions")
        return np.asarray(dP, dtype=np.float32)

    target_device = torch.device(device) if device is not None else torch.device("cpu")
    initial = torch.as_tensor(dP, dtype=torch.float32, device=target_device)
    flow = initial.detach().clone().requires_grad_(True)
    regions = [torch.as_tensor(points, dtype=torch.float32, device=target_device)
               for points in regions_np]
    foreground = torch.zeros(initial.shape[-2:], dtype=torch.bool, device=target_device)
    for points in regions:
        foreground[points[:, 0].long(), points[:, 1].long()] = True

    optimizer = torch.optim.Adam((flow,), lr=lr)
    scale = float(max(initial.shape[-2:]) ** 2)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        compact_losses = []
        for points in regions:
            endpoints = _follow_flow(flow, points, niter)
            compact_losses.append((endpoints - endpoints.mean(dim=0)).square().sum(dim=1).mean() / scale)
        compact = torch.stack(compact_losses).mean()
        anchor = (flow - initial).square().mean()
        smooth = _flow_smoothness(flow, foreground)
        loss = compact + flow_weight * anchor + smooth_weight * smooth
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss during Region-aware Flow-TTA")
        loss.backward()
        optimizer.step()

    tta_logger.info("refined flow for %d region(s), %d step(s)", len(regions), steps)
    return flow.detach().cpu().numpy().astype(np.float32, copy=False)
