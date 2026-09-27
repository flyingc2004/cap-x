# UniVTAC CaP-X Configs

## Current Entry Point

Use `tactile_memory_match_response_selection_easy_gt.yaml` for the current
composable tactile-response selection smoke.

- It uses public reset/grasp anchors only for coarse motion and does not launch
  SAM.
- It validates the public probe, task-local trial memory, and candidate
  selection. Transport is deliberately out of scope for this engineering demo.
- It requires a generated `tactile_response_expression.v1.json`, passed with
  `CAPX_TACTILE_RESPONSE_EXPRESSION` when it is not in the UniVTAC default
  location.

## Baselines

- `lift_can_tactile.yaml` is the native tactile regression baseline.
- `lift_can_no_tactile.yaml`, `lift_can_tactile_controller_only.yaml`, and the
  `grasp_classify_*` files remain compact comparison configurations.

## Memory Scope

The current response-selection configuration enables only bounded trial-local
memory. It does not inject a persistent code-memory or strategy-memory bank.
Those generic modules remain available for other experiments but are not part
of this UniVTAC entry point.

## Archived Configurations

`legacy/lift-can-memory-v1/` contains the earlier lift-can memory-stage
workflow. `legacy/tactile-transfer-v1/` contains the retired two-cylinder
transfer variants. They are preserved for history, but are not default entry
points and their former root paths are intentionally unavailable.
