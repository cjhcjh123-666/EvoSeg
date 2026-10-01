from projects.evoseg.qwen_process_seg.qwen_frame_audit import find_subsequence


def test_find_subsequence() -> None:
    assert find_subsequence([1, 2, 3, 2, 3], [2, 3]) == (1, 3)


def test_missing_subsequence_fails() -> None:
    try:
        find_subsequence([1, 2], [3])
    except ValueError as error:
        assert "not found" in str(error)
    else:
        raise AssertionError("missing query must fail")
