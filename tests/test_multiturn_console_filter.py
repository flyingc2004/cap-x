from capx.envs.trial import (
    _clip_multiturn_code,
    _clip_multiturn_console,
    _filter_console_for_multiturn,
)


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


def test_filter_console_for_multiturn_bounds_one_huge_error_line() -> None:
    raw = "CAPX_FAILURE " + ("diagnostic=" + "x" * 5000) + " final_reason=budget"

    filtered = _filter_console_for_multiturn(raw, max_lines=4, max_chars=600)

    assert len(filtered) <= 600
    assert "CAPX_FAILURE" in filtered
    assert "final_reason=budget" in filtered
    assert "line clipped" in filtered


def test_hard_console_clip_applies_after_filtering() -> None:
    clipped = _clip_multiturn_console("before" + "x" * 4000 + "after", 500, "stdout")

    assert len(clipped) <= 560
    assert "console clipped" in clipped
    assert clipped.endswith("after")


def test_clip_multiturn_code_keeps_recent_suffix() -> None:
    clipped = _clip_multiturn_code("old\n" * 600 + "latest", 1000, "history")
    assert "latest" in clipped
    assert "clipped" in clipped
