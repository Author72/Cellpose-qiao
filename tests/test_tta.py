import torch
from torch import nn

from cellpose import tta


class TinyFlowNet(nn.Module):
    """Small flow-readout network used to test TTA without SAM weights."""
    def __init__(self):
        super().__init__()
        self.out = nn.Conv2d(3, 3, kernel_size=1)
        self._dtype = torch.float32

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        return self.out(x), torch.zeros((x.shape[0], 256), device=x.device)


def test_flow_equivariance_loss_is_zero_for_matching_flows():
    prediction = torch.zeros((1, 3, 8, 8))
    prediction[:, 2] = 10  # mark all pixels confident
    flipped = torch.flip(prediction, dims=(-1,))
    flipped[:, 1].neg_()
    assert tta.flow_equivariance_loss(prediction, flipped, -1).item() < 1e-8


def test_flow_tta_restores_readout_and_parameter_flags():
    net = TinyFlowNet()
    original = {key: value.clone() for key, value in net.out.state_dict().items()}
    images = torch.randn(1, 32, 48, 3).numpy()
    state = tta.adapt_flow_head(net, images, steps=1, patch_size=64, max_patches=1)
    assert any(not torch.equal(value, original[key]) for key, value in net.out.state_dict().items())
    tta.restore_flow_head(net, state)
    assert all(torch.equal(value, original[key]) for key, value in net.out.state_dict().items())
    assert all(parameter.requires_grad for parameter in net.parameters())
