from projects.evoseg.process_virst.groundmore_adapter import action_interval, clip_start, object_ids


def test_groundmore_official_time_conversion():
    assert clip_start("Yuo-RMNZviA_0415_0430") == 4 * 60 + 15
    assert action_interval("Yuo-RMNZviA_0415_0430", "4:15", "4:23") == (0, 47)


def test_groundmore_multi_object_ids_are_preserved():
    assert object_ids("1, 3") == [1, 3]
