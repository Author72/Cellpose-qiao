"""Foreground-aware Maximum Mean Discrepancy test-time adaptation.

The source domain is represented by a bank of encoder-neck features sampled
from ground-truth foreground. At inference, high-confidence foreground is
frozen from the unadapted prediction and a small network module is optimized
episodically against a multi-kernel MMD objective.
"""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import transforms


mmd_tta_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MMDTTAConfig:
    """Settings for conservative, episodic MMD-TTA.

    ``steps=0`` disables adaptation. The defaults update only the CPSAM encoder
    neck, use MMD as a weak regularizer, and add photometric consistency plus a
    parameter anchor to reduce drift.
    """

    steps: int = 3
    learning_rate: float = 1e-5
    foreground_threshold: float = 0.8
    bandwidths: tuple = (0.5, 1.0, 2.0, 4.0)
    mmd_weight: float = 0.1
    consistency_weight: float = 0.1
    anchor_weight: float = 1e-4
    noise_std: float = 0.02
    max_source_samples: int = 2048
    max_target_samples: int = 2048
    min_samples: int = 16
    gradient_clip: float = 1.0
    adaptable_module: str = "encoder.neck"
    random_seed: int = 0

    def __post_init__(self):
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.foreground_threshold < 1.0:
            raise ValueError("foreground_threshold must be between 0 and 1")
        if not self.bandwidths or any(float(value) <= 0 for value in self.bandwidths):
            raise ValueError("bandwidths must contain positive values")
        if min(self.mmd_weight, self.consistency_weight, self.anchor_weight,
               self.noise_std) < 0:
            raise ValueError("loss weights and noise_std must be non-negative")
        if min(self.max_source_samples, self.max_target_samples,
               self.min_samples) < 1:
            raise ValueError("sample counts must be positive")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")


@dataclass
class MMDSourceBank:
    """Detached source-domain foreground features and their metadata."""

    features: torch.Tensor
    feature_layer: str = "encoder.neck"
    foreground_pixels: int = 0

    def __post_init__(self):
        self.features = torch.as_tensor(self.features, dtype=torch.float32,
                                        device="cpu").detach().contiguous()
        if (self.features.ndim != 2 or self.features.shape[0] == 0 or
                self.features.shape[1] == 0):
            raise ValueError("source features must have shape [N, C] with N,C > 0")
        if not torch.isfinite(self.features).all():
            raise ValueError("source features contain non-finite values")

    def save(self, path):
        """Save the bank without model weights or source images."""
        payload = {
            "features": self.features,
            "feature_layer": self.feature_layer,
            "foreground_pixels": int(self.foreground_pixels),
            "format_version": 1,
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path):
        """Load a bank created by :meth:`save`."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise ValueError("unsupported MMD source-bank format")
        return cls(features=payload["features"],
                   feature_layer=payload.get("feature_layer", "encoder.neck"),
                   foreground_pixels=payload.get("foreground_pixels", 0))


def multi_kernel_mmd(source, target, bandwidths=(0.5, 1.0, 2.0, 4.0)):
    """Return biased multi-kernel MMD squared for two ``[N, C]`` tensors."""
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target must have shape [N, C]")
    if source.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError("source and target must contain at least one sample")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target feature dimensions differ")
    source = F.normalize(source.float(), p=2, dim=1)
    target = F.normalize(target.float(), p=2, dim=1)
    ss_distance = torch.cdist(source, source).square()
    tt_distance = torch.cdist(target, target).square()
    st_distance = torch.cdist(source, target).square()
    loss = source.new_zeros(())
    for bandwidth in bandwidths:
        denominator = 2.0 * float(bandwidth)**2
        loss = loss + (torch.exp(-ss_distance / denominator).mean() +
                       torch.exp(-tt_distance / denominator).mean() -
                       2.0 * torch.exp(-st_distance / denominator).mean())
    return loss / len(bandwidths)


def _sample_rows(values, maximum, generator):
    if values.shape[0] <= maximum:
        return values
    indices = torch.randperm(values.shape[0], generator=generator)[:maximum]
    return values[indices.to(values.device)]


def _foreground_features(features, probability, threshold, maximum, generator):
    probability = F.interpolate(probability.float(), size=features.shape[-2:],
                                mode="bilinear", align_corners=False)
    selected = features.permute(0, 2, 3, 1)[probability[:, 0] > threshold]
    return _sample_rows(selected, maximum, generator)


def _mask_features(features, masks, maximum, generator):
    masks = F.interpolate(masks.float(), size=features.shape[-2:], mode="nearest")
    selected = features.permute(0, 2, 3, 1)[masks[:, 0] > 0.5]
    return _sample_rows(selected, maximum, generator)


def _image_tiles(images, bsize, tile_overlap, rescale=1.0, masks=None):
    """Tile normalized channel-last images like :func:`cellpose.core.run_net`."""
    if bsize < 1 or rescale <= 0:
        raise ValueError("bsize and rescale must be positive")
    if len(images) == 0:
        raise ValueError("at least one image is required")
    all_images, all_masks, all_valid = [], [], []
    for index, image in enumerate(images):
        image = (transforms.resize_image(image, rsz=rescale)
                 if rescale != 1.0 else image)
        height, width = image.shape[:2]
        ypad1, ypad2, xpad1, xpad2 = transforms.get_pad_yx(
            height, width, min_size=(bsize, bsize))
        image_chw = np.pad(image.transpose(2, 0, 1),
                           ((0, 0), (ypad1, ypad2), (xpad1, xpad2)),
                           mode="constant")
        image_tiles, ysub, xsub, _, _ = transforms.make_tiles(
            image_chw, bsize=bsize, augment=False, tile_overlap=tile_overlap)
        all_images.append(image_tiles.reshape(-1, *image_tiles.shape[-3:]))
        valid = np.pad(np.ones((height, width), dtype=bool),
                       ((ypad1, ypad2), (xpad1, xpad2)), mode="constant")
        all_valid.append(np.stack([
            valid[y0:y1, x0:x1]
            for (y0, y1), (x0, x1) in zip(ysub, xsub)
        ]))

        if masks is not None:
            mask = np.asarray(masks[index])
            if mask.ndim != 2:
                raise ValueError("each source mask must be a 2D label image")
            if mask.shape != (height, width):
                if rescale == 1.0:
                    raise ValueError("source image and mask spatial shapes differ")
                mask = transforms.resize_image(mask, Ly=height, Lx=width,
                                               interpolation=0,
                                               no_channels=True)
            mask = np.pad(mask, ((ypad1, ypad2), (xpad1, xpad2)),
                          mode="constant")
            all_masks.append(np.stack([
                mask[y0:y1, x0:x1]
                for (y0, y1), (x0, x1) in zip(ysub, xsub)
            ]))
    image_array = np.concatenate(all_images).astype(np.float32, copy=False)
    if masks is None:
        return image_array, np.concatenate(all_valid)
    return image_array, np.concatenate(all_masks)


def build_source_bank(net, images, masks, *, bsize=256, tile_overlap=0.1,
                      max_features=10000, batch_size=8, random_seed=0):
    """Build a source bank from normalized channel-last images and true masks."""
    if max_features < 1 or batch_size < 1:
        raise ValueError("max_features and batch_size must be positive")
    if isinstance(images, np.ndarray):
        images = [images] if images.ndim == 3 else list(images)
    else:
        images = list(images)
    if isinstance(masks, np.ndarray):
        masks = [masks] if masks.ndim == 2 else list(masks)
    else:
        masks = list(masks)
    if len(images) != len(masks):
        raise ValueError("source images and masks must contain the same number of items")
    image_tiles, mask_tiles = _image_tiles(images, bsize, tile_overlap, masks=masks)
    generator = torch.Generator(device="cpu").manual_seed(random_seed)
    chunks = []
    was_training = net.training
    net.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(image_tiles), batch_size):
                tile = torch.from_numpy(image_tiles[start:start + batch_size]).to(
                    net.device, dtype=net.dtype)
                feature = net.forward_features(tile)
                mask = torch.from_numpy(mask_tiles[start:start + batch_size, None]).to(
                    feature.device)
                selected = _mask_features(feature, mask, max_features, generator)
                if selected.numel():
                    chunks.append(selected.float().cpu())
    finally:
        net.train(was_training)
    if not chunks:
        raise ValueError("source masks contain no foreground at the selected feature scale")
    features = _sample_rows(torch.cat(chunks), max_features, generator)
    return MMDSourceBank(features=features,
                         foreground_pixels=sum(int((np.asarray(mask) > 0).sum())
                                               for mask in masks))


def _resolve_module(net, dotted_name):
    module = net
    for component in dotted_name.split("."):
        if not hasattr(module, component):
            raise ValueError(f"network has no adaptable module {dotted_name!r}")
        module = getattr(module, component)
    if not isinstance(module, torch.nn.Module):
        raise ValueError(f"adaptable object {dotted_name!r} is not a module")
    return module


class MMDTTAAdapter:
    """Apply and restore one episodic model adaptation."""

    def __init__(self, net, source_bank, config=None):
        self.net = net
        self.source_bank = source_bank
        self.config = config or MMDTTAConfig()
        if source_bank.feature_layer != "encoder.neck":
            raise ValueError("source bank feature layer is incompatible with CPSAM")
        if self.config.adaptable_module != "encoder.neck":
            raise ValueError("CPSAM MMD-TTA currently adapts encoder.neck only")
        if source_bank.features.shape[0] < self.config.min_samples:
            raise ValueError(
                "source bank contains fewer features than config.min_samples")
        self._restore_data = None

    def adapt(self, images, *, bsize=256, tile_overlap=0.1, batch_size=8,
              rescale=1.0):
        """Adapt to normalized images and leave weights changed until ``restore``."""
        if self._restore_data is not None:
            raise RuntimeError("restore the previous MMD-TTA episode before adapting again")
        if self.config.steps == 0:
            return []
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if images.ndim == 3:
            images = images[np.newaxis]
        image_tiles, valid_tiles = _image_tiles(
            images, bsize, tile_overlap, rescale=rescale)
        module = _resolve_module(self.net, self.config.adaptable_module)
        parameters = list(module.parameters())
        if not parameters:
            raise ValueError("adaptable module has no parameters")

        original_training = self.net.training
        original_requires_grad = [parameter.requires_grad
                                  for parameter in self.net.parameters()]
        original_state = {name: value.detach().clone()
                          for name, value in module.state_dict().items()}
        original_dtype = next(module.parameters()).dtype
        self._restore_data = (module, original_state, original_dtype,
                              original_requires_grad, original_training)

        for parameter in self.net.parameters():
            parameter.requires_grad_(False)
        # Float32 master parameters are important: a 1e-5 update can disappear
        # when applied directly to bfloat16 weights.
        module.float()
        for parameter in parameters:
            parameter.requires_grad_(True)
        self.net.eval()  # disables CPSAM stochastic layer dropping, not gradients

        generator = torch.Generator(device="cpu").manual_seed(
            self.config.random_seed)
        fixed_probabilities = []
        with torch.no_grad():
            for start in range(0, len(image_tiles), batch_size):
                tile = torch.from_numpy(image_tiles[start:start + batch_size]).to(
                    self.net.device, dtype=self.net.dtype)
                prediction = self.net(tile)[0]
                probability = torch.sigmoid(prediction[:, -1:]).float().cpu()
                valid = torch.from_numpy(
                    valid_tiles[start:start + batch_size, None])
                fixed_probabilities.append(probability * valid)

        optimizer = torch.optim.Adam(parameters, lr=self.config.learning_rate)
        initial_parameters = [parameter.detach().clone() for parameter in parameters]
        history = []
        source_dimension = self.source_bank.features.shape[1]
        skipped_batches = 0
        for _ in range(self.config.steps):
            optimizer.zero_grad(set_to_none=True)
            mmd_total = 0.0
            consistency_total = 0.0
            target_sample_total = 0
            valid_batches = 0
            for batch_index, start in enumerate(range(0, len(image_tiles), batch_size)):
                tile = torch.from_numpy(image_tiles[start:start + batch_size]).to(
                    self.net.device, dtype=self.net.dtype)
                prediction, feature = self.net.forward_with_features(tile)
                if feature.shape[1] != source_dimension:
                    raise ValueError("source-bank and target feature dimensions differ")
                target = _foreground_features(
                    feature, fixed_probabilities[batch_index].to(feature.device),
                    self.config.foreground_threshold,
                    self.config.max_target_samples, generator)
                if target.shape[0] < self.config.min_samples:
                    skipped_batches += 1
                    continue
                source = _sample_rows(self.source_bank.features,
                                      self.config.max_source_samples,
                                      generator).to(feature.device)
                mmd_loss = multi_kernel_mmd(source, target,
                                            self.config.bandwidths)

                if self.config.consistency_weight:
                    noise = torch.randn(tile.shape, generator=generator,
                                        dtype=torch.float32).to(tile.device)
                    augmented = (tile.float() + self.config.noise_std * noise).to(
                        dtype=tile.dtype)
                    augmented_prediction = self.net(augmented)[0]
                    consistency = F.mse_loss(augmented_prediction.float(),
                                             prediction.detach().float())
                else:
                    consistency = mmd_loss.new_zeros(())
                batch_loss = (self.config.mmd_weight * mmd_loss +
                              self.config.consistency_weight * consistency)
                if not torch.isfinite(batch_loss):
                    raise RuntimeError("non-finite loss during MMD-TTA")
                batch_loss.backward()
                mmd_total += float(mmd_loss.detach())
                consistency_total += float(consistency.detach())
                target_sample_total += int(target.shape[0])
                valid_batches += 1
            if not valid_batches:
                break

            # Average the independently backpropagated tile gradients so the
            # effective step size does not grow with image dimensions.
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(valid_batches)
            anchor = torch.stack([
                (parameter - initial).float().square().mean()
                for parameter, initial in zip(parameters, initial_parameters)
            ]).mean()
            if self.config.anchor_weight:
                (self.config.anchor_weight * anchor).backward()
            mean_mmd = mmd_total / valid_batches
            mean_consistency = consistency_total / valid_batches
            mean_loss = (self.config.mmd_weight * mean_mmd +
                         self.config.consistency_weight * mean_consistency +
                         self.config.anchor_weight * float(anchor.detach()))
            if not np.isfinite(mean_loss):
                raise RuntimeError("non-finite loss during MMD-TTA")
            torch.nn.utils.clip_grad_norm_(parameters,
                                           self.config.gradient_clip)
            optimizer.step()
            history.append({
                "loss": mean_loss,
                "mmd": mean_mmd,
                "consistency": mean_consistency,
                "anchor": float(anchor.detach()),
                "target_samples": target_sample_total,
            })
        if not history:
            mmd_tta_logger.warning(
                "skipping MMD-TTA: no tile had at least %d confident foreground features",
                self.config.min_samples)
            self.restore()
        else:
            mmd_tta_logger.info(
                "MMD-TTA completed %d update(s); final MMD %.6f; skipped %d batch(es)",
                len(history), history[-1]["mmd"], skipped_batches)
        return history

    def restore(self):
        """Restore source weights, dtypes, gradient flags, and train/eval mode."""
        if self._restore_data is None:
            return
        module, state, dtype, requires_grad, training = self._restore_data
        module.load_state_dict(state)
        module.to(dtype=dtype)
        for parameter, required in zip(self.net.parameters(), requires_grad):
            parameter.requires_grad_(required)
        self.net.train(training)
        self._restore_data = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore()


def config_dict(config):
    """Return a JSON-friendly representation useful for experiment logging."""
    values = asdict(config)
    values["bandwidths"] = list(values["bandwidths"])
    return values
