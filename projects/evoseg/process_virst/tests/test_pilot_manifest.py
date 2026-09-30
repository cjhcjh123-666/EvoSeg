from projects.evoseg.process_virst.prepare_pilot_manifest import select_videos


def test_pilot_samples_videos_before_retaining_all_objects():
    manifest = {
        "objects": [
            {"video_id": "v1", "object_id": 1},
            {"video_id": "v1", "object_id": 2},
            {"video_id": "v2", "object_id": 3},
            {"video_id": "v3", "object_id": 4},
        ]
    }
    selected = select_videos(manifest, count=2, seed=42)
    chosen = set(selected["process_virst_pilot"]["selected_videos"])
    assert len(chosen) == 2
    assert {item["video_id"] for item in selected["objects"]} == chosen
    for video_id in chosen:
        expected = [item for item in manifest["objects"] if item["video_id"] == video_id]
        actual = [item for item in selected["objects"] if item["video_id"] == video_id]
        assert actual == expected
