import numpy as np
import torch

from projects.evoseg.temporal_dense_prompt.protocol import DensePromptHead, count_parameters, local_extrema_points, one_point, stage_endpoints


def test_head_shape_and_budget():
    model=DensePromptHead();output=model(torch.randn(2,256),torch.randn(2,256,12,16));assert output.shape==(2,12,16);assert count_parameters(model)<2_000_000


def test_points_fixed_counts_and_disjoint():
    value=np.linspace(0,1,100,dtype=np.float32).reshape(10,10);points,labels=local_extrema_points(value);assert labels.count(1)==4;assert labels.count(0)==4;assert len({tuple(x) for x in points})==8
    points,labels=one_point(value);assert points==[[.95,.95]];assert labels==[1]


def test_four_stage_endpoints():
    assert stage_endpoints(235)==[58,117,176,234]
