from argparse import Namespace

from projects.evoseg.process_virst.train_sft import validate_resume


def arguments() -> Namespace:
    return Namespace(
        seed=11,
        warmup_steps=128,
        ordered_steps=256,
        frames=8,
        learning_rate=1e-5,
        alignment_mode="monotonic",
        disable_order_loss=False,
    )


def state() -> dict:
    return {
        "seed": 11,
        "total_steps": 384,
        "warmup_steps": 128,
        "ordered_steps": 256,
        "frames": 8,
        "learning_rate": 1e-5,
        "process_config": {
            "alignment_mode": "monotonic",
            "order_loss_enabled": True,
        },
    }


def test_resume_configuration_matches() -> None:
    validate_resume(arguments(), state())


def test_resume_rejects_changed_budget() -> None:
    value = state()
    value["ordered_steps"] = 255
    try:
        validate_resume(arguments(), value)
    except ValueError as error:
        assert "configuration mismatch" in str(error)
    else:
        raise AssertionError("changed training budget was accepted")


def test_resume_rejects_changed_method() -> None:
    value = state()
    value["process_config"] = {
        "alignment_mode": "global",
        "order_loss_enabled": True,
    }
    try:
        validate_resume(arguments(), value)
    except ValueError as error:
        assert "ProcessVIRST configuration mismatch" in str(error)
    else:
        raise AssertionError("changed ProcessVIRST method was accepted")
