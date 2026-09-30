import torch

from projects.evoseg.factorized_temporal_spatial.protocol import (
    MeanPoolProbeFactory,
    count_parameters,
    fixed_derangement,
)
from projects.evoseg.temporal_grounding_mechanism.train_probe import ProbeFactory


def test_mean_pool_is_parameter_matched():
    temporal = count_parameters(ProbeFactory.build())
    mean_pool = count_parameters(MeanPoolProbeFactory.build())
    assert abs(mean_pool - temporal) / temporal < 0.01


def test_mean_pool_shape():
    model = MeanPoolProbeFactory.build()
    output = model(torch.randn(3, 256), torch.randn(3, 4, 256), torch.randn(3, 4, 1152))
    assert output.shape == (3,)


def test_shuffle_is_fixed_and_non_identity():
    first = fixed_derangement("long_rvos/video/1/2")
    second = fixed_derangement("long_rvos/video/1/2")
    assert first == second
    assert first != [0, 1, 2, 3]
    assert sorted(first) == [0, 1, 2, 3]
