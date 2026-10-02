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

## Memory Scope

The current response-selection configuration enables only bounded trial-local
memory. It does not inject a persistent code-memory or strategy-memory bank.
Those generic modules remain available for other experiments but are not part
of this UniVTAC entry point.

## ViTaForge OpenTac Tension-Strap Control

`tension_strap_stage_memory_control.yaml` is the active one-seed CaP-X control
entry for ViTaForge's public `tension_strap` task. The companion
`tension_strap_stage_memory_probe.yaml` is retained for probe/retrieval
debugging. Both use the frozen external 12N/18N response-memory sidecar:
`tactile_response_memory/tension_strap_12n_18n.v2.json`. The v2 snapshot
uses the public EE-z plateau/transition alignment from the expert replay; the
retained v1 sidecar is an audit-only adjacent-window snapshot and is not used
by the active control configuration.

The sidecar contains only calibration-derived public depth/marker response
prototypes and a pooled calibration IQR for comparable cross-stage distances.
At runtime, the ViTaForge-specific `OpenTacApi` exposes the frozen memory, a
public `tactile_stage_response.v1` capture, and a marker-RGB calibrated tension
estimate. Generated code owns local Z feedback, response distance, and stage
transitions. `OpenTacApi` never returns true tension, actor state, reward,
success, or an action recommendation. `UniVTACTactileApi` remains unchanged
for UniVTAC tasks.

The selected `UNIVTAC_ROOT` must point to the matching ViTaForge checkout and
each force-task seed must run in its own process. The control entry is an
integration target, not yet a reported success-rate result.

## Archived Configurations

`legacy/lift-can-memory-v1/` contains the earlier lift-can memory-stage
workflow. `legacy/tactile-transfer-v1/` contains the retired two-cylinder
transfer variants. `legacy/capx-baselines-v1/` contains the former
`lift_can`/`grasp_classify` CaP-X comparisons. They are preserved for history,
but are not default entry points and their former root paths are intentionally
unavailable.
