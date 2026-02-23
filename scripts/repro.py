import torch
import coremltools as ct

class EinsumRepro(torch.nn.Module):
    def forward(self, rotations, freq_ranges):
        return torch.einsum('..., f -> ... f', rotations, freq_ranges)

model = EinsumRepro().eval()

rotations = torch.randn(2, 3)
freq_ranges = torch.randn(3)

traced = torch.jit.trace(model, (rotations, freq_ranges), check_trace=False)

ct.convert(
    traced,
    inputs=[
        ct.TensorType(name='rotations', shape=rotations.shape),
        ct.TensorType(name='freq_ranges', shape=freq_ranges.shape),
    ],
)
