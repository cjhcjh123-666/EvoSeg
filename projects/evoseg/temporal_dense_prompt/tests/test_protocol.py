import numpy as np
import torch

from projects.evoseg.temporal_dense_prompt.protocol import DensePromptHead, count_parameters, local_extrema_points, one_point, resize_mask, stage_endpoints
from projects.evoseg.temporal_dense_prompt.summarize import pixel_coverage


def test_head_shape_and_budget():
    model=DensePromptHead();output=model(torch.randn(2,256),torch.randn(2,256,12,16));assert output.shape==(2,12,16);assert count_parameters(model)<2_000_000


def test_points_fixed_counts_and_disjoint():
    value=np.linspace(0,1,100,dtype=np.float32).reshape(10,10);points,labels=local_extrema_points(value);assert labels.count(1)==4;assert labels.count(0)==4;assert len({tuple(x) for x in points})==8
    points,labels=one_point(value);assert points==[[.95,.95]];assert labels==[1]


def test_four_stage_endpoints():
    assert stage_endpoints(235)==[58,117,176,234]


def test_absent_official_mask_is_empty(tmp_path):
    value=resize_mask(tmp_path/'absent.png',(72,72));assert value.shape==(72,72);assert value.sum()==0


def test_pixel_coverage_requires_every_condition():
    dense=[{'identity':'dataset/video/1/expression'}]
    pixel=[{'identity':'dataset/video/1/expression','condition':condition} for condition in ('static_p1','temporal_p1','static_p8')]
    coverage=pixel_coverage(dense,pixel)
    assert not coverage['complete']
    assert coverage['missing_identity_condition_pairs']==1
    pixel.append({'identity':'dataset/video/1/expression','condition':'temporal_p8'})
    assert pixel_coverage(dense,pixel)['complete']
