from PIL import Image

from projects.evoseg.ftg.compare_qualitatives import _sample_key, compose_pair


def test_contact_sheet_filename_recovers_public_sample_key():
    assert _sample_key("002_long_rvos_hybrid_17aeae3bea_8.jpg") == (
        "long_rvos", "17aeae3bea", "8"
    )
    assert _sample_key("001_mevis_v2_motion_4f90b1ea9a10_3.jpg") == (
        "mevis_v2", "4f90b1ea9a10", "3"
    )


def test_compose_pair_adds_two_labeled_rows(tmp_path):
    frame = tmp_path / "frame.jpg"
    ftg = tmp_path / "ftg.jpg"
    output = tmp_path / "comparison.jpg"
    Image.new("RGB", (80, 40), "red").save(frame)
    Image.new("RGB", (80, 40), "blue").save(ftg)

    compose_pair(frame, ftg, output, "the moving object")

    comparison = Image.open(output)
    assert comparison.width == 80
    assert comparison.height > 2 * 40
