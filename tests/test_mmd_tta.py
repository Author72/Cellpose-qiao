from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from cellpose.cli import get_arg_parser
from cellpose.models import CellposeModel
from cellpose.mmd_tta import (MMDSourceBank, MMDTTAAdapter, MMDTTAConfig,
                              build_source_bank, multi_kernel_mmd)


class TinyNet(nn.Module):
    """Small network exposing the same TTA interface as CPSAM."""

    def __init__(self, foreground_logit=4.0):
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 1)
        self.encoder = SimpleNamespace(neck=nn.Sequential(
            nn.Conv2d(4, 4, 1), nn.ReLU(), nn.Conv2d(4, 4, 1)))
        # Register the namespace module's neck on the actual module tree.
        self.add_module("neck", self.encoder.neck)
        self.out = nn.Conv2d(4, 3, 1)
        nn.init.constant_(self.out.bias, 0.0)
        self.out.bias.data[-1] = foreground_logit
        self._dtype = torch.float32

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return self._dtype

    def forward_features(self, x):
        return self.encoder.neck(self.stem(x))

    def forward_with_features(self, x):
        feature = self.forward_features(x)
        return self.out(feature), feature

    def forward(self, x):
        prediction, feature = self.forward_with_features(x)
        return prediction, torch.zeros((len(feature), 256), device=feature.device)


def test_multi_kernel_mmd_is_zero_for_identical_samples():
    generator = torch.Generator().manual_seed(3)
    source = torch.randn((12, 5), generator=generator)
    identical = multi_kernel_mmd(source, source)
    different = multi_kernel_mmd(source, -source + 0.2)
    assert identical.item() == pytest.approx(0.0, abs=1e-6)
    assert different > identical


def test_source_bank_build_and_roundtrip(tmp_path):
    net = TinyNet()
    image = np.random.default_rng(2).normal(size=(16, 16, 3)).astype("float32")
    mask = np.zeros((16, 16), dtype="uint16")
    mask[3:13, 4:12] = 1
    bank = build_source_bank(net, image, mask, bsize=16, batch_size=2,
                             max_features=40)
    assert bank.features.shape == (40, 4)
    assert bank.foreground_pixels == 80
    assert bank.features.device.type == "cpu"

    path = tmp_path / "bank.pt"
    bank.save(path)
    loaded = MMDSourceBank.load(path)
    assert torch.equal(loaded.features, bank.features)
    assert loaded.foreground_pixels == bank.foreground_pixels


def test_adapter_updates_then_restores_neck():
    torch.manual_seed(4)
    net = TinyNet()
    bank = MMDSourceBank(torch.randn(32, 4))
    config = MMDTTAConfig(
        steps=1,
        learning_rate=1e-2,
        foreground_threshold=0.8,
        bandwidths=(0.5, 1.0),
        mmd_weight=1.0,
        consistency_weight=0.0,
        anchor_weight=0.0,
        max_source_samples=16,
        max_target_samples=16,
        min_samples=4,
    )
    original = {name: value.detach().clone()
                for name, value in net.encoder.neck.state_dict().items()}
    target = np.random.default_rng(5).normal(size=(1, 16, 16, 3)).astype("float32")

    adapter = MMDTTAAdapter(net, bank, config)
    history = adapter.adapt(target, bsize=16, batch_size=2)
    assert history
    assert any(not torch.equal(value, original[name])
               for name, value in net.encoder.neck.state_dict().items())

    adapter.restore()
    assert all(torch.equal(value, original[name])
               for name, value in net.encoder.neck.state_dict().items())
    assert all(parameter.requires_grad for parameter in net.parameters())


def test_adapter_safely_falls_back_without_confident_foreground():
    net = TinyNet(foreground_logit=-10.0)
    bank = MMDSourceBank(torch.randn(16, 4))
    config = MMDTTAConfig(steps=1, foreground_threshold=0.9,
                          min_samples=4, consistency_weight=0.0)
    original = {name: value.detach().clone()
                for name, value in net.encoder.neck.state_dict().items()}
    target = np.zeros((1, 16, 16, 3), dtype="float32")

    adapter = MMDTTAAdapter(net, bank, config)
    assert adapter.adapt(target, bsize=16, batch_size=2) == []
    assert all(torch.equal(value, original[name])
               for name, value in net.encoder.neck.state_dict().items())


def test_cellpose_eval_integration_is_episodic():
    torch.manual_seed(7)
    net = TinyNet()
    model = CellposeModel.__new__(CellposeModel)
    model.net = net
    model.device = torch.device("cpu")
    model.mmd_source_bank = MMDSourceBank(torch.randn(32, 4))
    model.last_mmd_tta_history = []
    original = {name: value.detach().clone()
                for name, value in net.encoder.neck.state_dict().items()}
    config = MMDTTAConfig(
        steps=1, learning_rate=1e-2, consistency_weight=0.0,
        anchor_weight=0.0, min_samples=4, max_source_samples=16,
        max_target_samples=16,
    )
    image = np.random.default_rng(8).normal(size=(16, 16, 3)).astype("float32")

    masks, flows, _ = model.eval(
        image, normalize=False, compute_masks=False, bsize=16,
        batch_size=2, mmd_tta=config)
    assert masks.size == 0
    assert flows[1].shape == (2, 16, 16)
    assert model.last_mmd_tta_history
    assert all(torch.equal(value, original[name])
               for name, value in net.encoder.neck.state_dict().items())


def test_config_and_cli_validation():
    with pytest.raises(ValueError, match="foreground_threshold"):
        MMDTTAConfig(foreground_threshold=1.0)
    with pytest.raises(ValueError, match="fewer features"):
        MMDTTAAdapter(TinyNet(), MMDSourceBank(torch.randn(2, 4)),
                      MMDTTAConfig(min_samples=3))
    with pytest.raises(ValueError, match="encoder.neck only"):
        MMDTTAAdapter(TinyNet(), MMDSourceBank(torch.randn(4, 4)),
                      MMDTTAConfig(min_samples=3, adaptable_module="out"))

    args = get_arg_parser().parse_args([
        "--mmd_source_bank", "bank.pt", "--mmd_tta_steps", "2",
        "--mmd_tta_bandwidths", "0.25", "1.0",
    ])
    assert args.mmd_source_bank == "bank.pt"
    assert args.mmd_tta_steps == 2
    assert args.mmd_tta_bandwidths == [0.25, 1.0]
