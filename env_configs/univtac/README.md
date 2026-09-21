# UniVTAC CaP-X Configs

## Current Entry Point

Use `tactile_memory_match_easy_sam_gt.yaml` for the current public-v4
response-memory selection smoke.

- Despite its historical filename, it is **Easy-GT**, not SAM. It uses public
  reset/grasp anchors for coarse motion and does not launch SAM.
- It is a fixed, selection-only engineering smoke: CaP-X performs the public
  probe for reference/left/right, writes trial memory, and selects a
  candidate. It intentionally does not transport an object to the slot.
- It requires a generated `tactile_response_expression.v1.json`, provided at
  runtime through `CAPX_TACTILE_RESPONSE_EXPRESSION`.

## Memory Terminology

Every tactile-memory-match YAML currently contains:

```yaml
tactile_memory:
  trial:
    enabled: true
    include_in_multiturn: true
  persistent:
    enabled: false
```

`trial.enabled` means generated code can write/read `trial_memory.v1` records
within one trial, including a failure-only regeneration. `persistent.enabled`
is false, so no configuration carries tactile evidence or a code/skill bank
across trials. Therefore the `_memory` filename suffix does **not** mean that
memory is enabled; it is a historical name only.

## File Status

| File | Status | Localization | Memory protocol | Use it now? |
| --- | --- | --- | --- | --- |
| `tactile_memory_match_easy_sam_gt.yaml` | Current | Easy-GT public anchors | `tactile_probe.v4` plus frozen `tactile_response_expression.v1` | Yes |

The prior compatibility, Hard-SAM, and `*_memory` YAMLs were deleted because
they described obsolete full-transport experiments and were frequently
mistaken for distinct memory modes. Git history preserves them if a prior run
must be reproduced. A future no-memory baseline must use an explicitly named
config with `tactile_memory.trial.enabled: false` and no
`write_trial_memory` / `read_trial_memory` APIs.
