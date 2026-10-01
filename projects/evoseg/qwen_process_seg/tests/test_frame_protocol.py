import torch

from projects.evoseg.qwen_process_seg.frame_protocol import (
    gather_frame_tokens,
    recover_frame_token_spans,
    uniform_frame_indices,
)


def test_uniform_sampling_covers_full_video() -> None:
    indices = uniform_frame_indices(101, 16)
    assert len(indices) == 16
    assert indices[0] == 0
    assert indices[-1] == 100
    assert indices == sorted(set(indices))


def test_short_video_uses_each_frame_once() -> None:
    assert uniform_frame_indices(4, 16) == [0, 1, 2, 3]


def test_recover_frame_token_spans() -> None:
    input_ids = torch.tensor([[9, 7, 7, 7, 7, 8, 7, 7, 7, 7, 10]])
    grids = torch.tensor([[1, 4, 4], [1, 4, 4]])
    spans = recover_frame_token_spans(input_ids, grids, [0, 12], 7, 2)
    assert [(span.token_start, span.token_end) for span in spans] == [(1, 5), (6, 10)]
    hidden = torch.arange(11 * 3, dtype=torch.float32).reshape(1, 11, 3)
    tokens, summaries = gather_frame_tokens(hidden, spans)
    assert [len(frame) for frame in tokens] == [4, 4]
    assert summaries.shape == (2, 3)


def test_span_mismatch_is_not_silent() -> None:
    input_ids = torch.tensor([7, 7, 7])
    grids = torch.tensor([[1, 4, 4]])
    try:
        recover_frame_token_spans(input_ids, grids, [0], 7, 2)
    except ValueError as error:
        assert "visual-token mismatch" in str(error)
    else:
        raise AssertionError("mismatch must fail")
