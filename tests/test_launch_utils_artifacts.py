from __future__ import annotations

import json

from capx.utils.launch_utils import _save_in_progress_trial_artifacts


def test_save_in_progress_trial_artifacts_writes_generated_code(tmp_path) -> None:
    prompt = [{"content": [{"text": "task prompt"}]}]
    code_path = _save_in_progress_trial_artifacts(
        {"output_dir": str(tmp_path)},
        1,
        final_code="# Code block 0\nprint('hi')\n",
        raw_code="```python\nprint('hi')\n```",
        all_responses=[
            {
                "decision": "initial",
                "initial_prompt": prompt,
                "code_blocks": ["print('hi')"],
            }
        ],
    )

    trial_dir = tmp_path / "trial_01_in_progress"
    assert code_path == str(trial_dir / "code.py")
    assert (trial_dir / "code.py").read_text() == "# Code block 0\nprint('hi')\n"
    assert "print('hi')" in (trial_dir / "raw_response.sh").read_text()
    assert (trial_dir / "initial_prompt.txt").read_text() == "task prompt"
    assert json.loads((trial_dir / "all_responses.json").read_text())[0]["decision"] == "initial"
