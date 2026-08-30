from capx.envs.trial import _filter_console_for_multiturn


def test_filter_console_for_multiturn_keeps_structured_tactile_lines() -> None:
    raw = "\n".join(
        [
            "Step     1(   N/A%), action     0, FPS   1.57, Running   74.86s",
            "[univtac-franka] adaptive_close stable=False reason=one_sided_high_force force=0.455 width=0.3910 steps=62",
            "[univtac-tactile] hand=both contact=True left=False right=True force=0.455 slip=0.100 event=one_hand_contact",
            "CAPX_FAILURE object=object_a phase=close reason=one_sided_contact action=retry_lower_z stable=false contact=true left=false right=true force=0.455 slip=0.100",
            "ordinary print line",
        ]
    )

    filtered = _filter_console_for_multiturn(raw)

    assert "Step     1" not in filtered
    assert "[console-filter]" in filtered
    assert "[univtac-franka] adaptive_close" in filtered
    assert "[univtac-tactile]" in filtered
    assert "CAPX_FAILURE object=object_a" in filtered
    assert "ordinary print line" in filtered


def test_filter_console_for_multiturn_limits_recent_other_lines() -> None:
    raw = "\n".join(
        [f"ordinary {i}" for i in range(8)]
        + ["CAPX_EVENT object=object_a phase=close reason=checked action=continue stable=true contact=true left=true right=true force=0.8 slip=0.0"]
    )

    filtered = _filter_console_for_multiturn(raw, max_lines=5, keep_recent_other=2)

    assert "CAPX_EVENT object=object_a" in filtered
    assert "ordinary 0" not in filtered
    assert "ordinary 6" in filtered
    assert "ordinary 7" in filtered
    assert "omitted_other=6" in filtered
