# Shadow Hand 钢琴实验：成功音数与动作步数

当前索引：137 次运行；118 次有可读原生计分和动作审计。其余保留在表中，缺失值用“—”表示，不按 0 处理。

## 统计口径

- **成功音数**取 `shadow_piano_native_audit_summary.json` 的 `native_notes_completed`，分母取该摘要中的请求音数；它是原生任务的逐音计分，不是生成代码打印的信心值，也不是 `Task Completed`。启用失败后继续尝试的运行会将成功音数和尝试数分开记录。
- **总原生动作步**精确计数 `shadow_native_events.jsonl` 中 `kind=action` 的行数。每行是一次 `Task.step()`；旧配置推进 8 个 240 Hz 物理步，`4phys` 配置推进 4 个。批量 `steps=n` 计为 n 步。包括 reset 后的自动视觉清场动作，不包括 reset、键位映射、尝试结束或 END。
- **生成代码动作步**＝总原生动作步减去同次 `run.log` 记录的 `keyboard clear at action_step`。没有清场记录但总数为 0 时记 0；其余缺少清场记录的运行记“—”，避免把适配器清场算成模型动作。
- 本次 `random6_online_skills` 运行另有公开 reset 记录明确记载清场步数，故缓存命中时清场为 0，生成动作数可直接与总数核对；不将这一规则追溯用于缺少记录的旧实验。
- **END**取私有摘要 `end_submitted`。终端退出码只在 `run.log` 存在时记录；退出码 0 本身不表示钢琴成功。若审计无权限、缺失或运行在启动阶段中断，不虚构成功音数或动作数。
- 历史运行的控制配置和故障类型不同。本表是完整实验账本；论文计算成功率或平均动作数时，须先固定纳入标准、配置版本及是否包含未完成运行，再从本表筛选，不能直接平均全部行。

## 逐次记录（UTC）

| 时间 | 实验版本 | Seed | 原生成功音数 | 尝试数 | 生成代码动作步 | 总原生动作步 | END | 退出码 | 原始证据 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | --- |
| 2026-10-20 09:49:36 | `visual_force_smoke` | 3150 | 0/8 | — | 0 | 0 | 否 | — | — · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_visual_force_seed3150_smoke_rep1_20261020T094936Z_8e6a63058e6747dea692aa2d27c7d724/shadow_piano_native_audit_summary.json) |
| 2026-10-20 13:41:24 | `visual_force_smoke` | 3150 | 0/8 | — | 0 | 0 | 否 | — | — · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_visual_force_seed3150_smoke_rep1_20261020T134124Z_d0c3fd99cc64469fb233a4831263fcfd/shadow_piano_native_audit_summary.json) |
| 2026-10-20 14:21:03 | `visual_force_utf8_smoke` | 3150 | 0/8 | — | 0 | 0 | 是 | — | — · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_visual_force_seed3150_utf8_smoke_rep1_20261020T142103Z_85efac73dc7c4e90b8c748f023551987/shadow_piano_native_audit_summary.json) |
| 2026-10-20 16:45:04 | `visual_force_mirrored_smoke` | 3150 | 0/8 | — | 0 | 0 | 否 | — | — · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_visual_force_mirrored_seed3150_smoke_rep1_20261020T164504Z_13edc06c9edd4931b97316b952f9ce32/shadow_piano_native_audit_summary.json) |
| 2026-10-20 18:03:00 | `visual_force_mirrored_smoke` | 3150 | 0/8 | — | 0 | 0 | 否 | — | — · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_visual_force_mirrored_seed3150_smoke_rep1_20261020T180300Z_eab3afa218b749538609c8994ee87d38/shadow_piano_native_audit_summary.json) |
| 2026-10-20 18:30:54 | `clear_view_smoke_smoke` | 3150 | 0/8 | — | — | 632 | 是 | — | — · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_smoke_smoke_rep1_20261020T183054Z_6991641085ed4fdd929f02a3a7631d55/shadow_piano_native_audit_summary.json) |
| 2026-10-20 18:56:06 | `clear_view_v2_smoke` | 3150 | 0/8 | — | — | 120 | 否 | — | — · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v2_smoke_rep1_20261020T185606Z_5d076ab1fbce4678a473fef640874e8c/shadow_piano_native_audit_summary.json) |
| 2026-10-20 19:37:27 | `clear_view_v2_smoke` | 3150 | 0/8 | — | — | 174 | 否 | — | — · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v2_smoke_rep1_20261020T193727Z_394246ea63524b11b11fbcc5c45e191b/shadow_piano_native_audit_summary.json) |
| 2026-10-21 03:06:16 | `clear_view_v2_smoke` | 3150 | 0/8 | — | — | 120 | 否 | — | — · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v2_smoke_rep1_20261021T030616Z_4455cbb2ad5f4d76b40a7edd9d82870b/shadow_piano_native_audit_summary.json) |
| 2026-10-21 03:40:31 | `clear_view_v2_smoke` | 3150 | 0/8 | — | — | 267 | 否 | — | — · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v2_smoke_rep1_20261021T034031Z_6ca401142f6540818499d749d007ce98/shadow_piano_native_audit_summary.json) |
| 2026-10-21 06:15:33 | `clear_view_v2_smoke` | 3150 | 0/8 | — | — | 259 | 否 | — | — · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v2_smoke_rep1_20261021T061533Z_8089081df70544cb923e4329482f8054/shadow_piano_native_audit_summary.json) |
| 2026-10-21 06:25:40 | `clear_view_v2_smoke` | 3150 | — | — | — | — | — | — | — · —（审计无权限或缺失） |
| 2026-10-21 06:26:42 | `clear_view_v2_smoke` | 3150 | 0/8 | — | — | 110 | 否 | — | — · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v2_smoke_rep1_20261021T062642Z_45c3e6b6fee0495cb620750715d68e14/shadow_piano_native_audit_summary.json) |
| 2026-10-21 06:33:01 | `clear_view_v2_smoke` | 3150 | — | — | — | — | — | — | — · —（审计无权限或缺失） |
| 2026-10-21 06:35:02 | `clear_view_v2_smoke` | 3150 | 0/8 | — | — | 110 | 否 | — | — · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v2_smoke_rep1_20261021T063502Z_1d987e2659f74b1a9993a62d1a06d4a1/shadow_piano_native_audit_summary.json) |
| 2026-10-21 06:42:09 | `clear_view_v2_smoke` | 3150 | 0/8 | — | 211 | 321 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v2_smoke_rep1_20261021T064209Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v2_smoke_rep1_20261021T064209Z_1ce38005a069492d80cbfebcea2ad9d1/shadow_piano_native_audit_summary.json) |
| 2026-10-21 08:37:50 | `clear_view_v2_smoke` | 3150 | 0/8 | — | 58 | 168 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v2_smoke_rep1_20261021T083750Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v2_smoke_rep1_20261021T083750Z_aa5dab62d3f74e64a4b531e97243a400/shadow_piano_native_audit_summary.json) |
| 2026-10-21 15:20:11 | `clear_view_v3_smoke` | 3150 | 1/8 | — | 287 | 397 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261021T152011Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v3_smoke_rep1_20261021T152011Z_f8d9d8dd268745ebb9cea75709b113ae/shadow_piano_native_audit_summary.json) |
| 2026-10-21 19:13:28 | `clear_view_v3_smoke` | 3150 | 4/8 | — | 407 | 517 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261021T191328Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v3_smoke_rep1_20261021T191328Z_c91f55470da946c8b9ccd95cfb443ae7/shadow_piano_native_audit_summary.json) |
| 2026-10-21 19:28:51 | `clear_view_v3_smoke` | 3150 | 1/8 | — | 182 | 292 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261021T192851Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v3_smoke_rep1_20261021T192851Z_4ba58de4aa17490eb23efc94e0c06300/shadow_piano_native_audit_summary.json) |
| 2026-10-21 19:38:34 | `clear_view_v3_smoke` | 3150 | 0/8 | — | 0 | 110 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261021T193834Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v3_smoke_rep1_20261021T193834Z_c653835b50fd4b5ea4c3992a6a33aa74/shadow_piano_native_audit_summary.json) |
| 2026-10-22 00:09:38 | `clear_view_v3_smoke` | 3150 | 0/8 | — | 0 | 110 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261022T000938Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v3_smoke_rep1_20261022T000938Z_007a1bd943a347f2aa52af64b9da532c/shadow_piano_native_audit_summary.json) |
| 2026-10-22 00:12:45 | `clear_view_v3_smoke` | 3150 | 2/8 | — | 215 | 325 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261022T001245Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v3_smoke_rep1_20261022T001245Z_2f91943ff71847a6886e55281d7bc80c/shadow_piano_native_audit_summary.json) |
| 2026-10-22 00:18:09 | `clear_view_v3_smoke` | 3150 | 1/8 | — | 240 | 350 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261022T001809Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v3_smoke_rep1_20261022T001809Z_ab775fcf48c24e1cb6ab1cba8cb68c28/shadow_piano_native_audit_summary.json) |
| 2026-10-22 03:22:37 | `clear_view_v3_smoke` | 3150 | — | — | — | — | — | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261022T032237Z/run.log) · —（原生审计缺失） |
| 2026-10-22 14:44:24 | `clear_view_v3_smoke` | 3150 | 1/8 | — | 278 | 388 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261022T144424Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v3_smoke_rep1_20261022T144424Z_4d5d413daf6a455fbc709cdfb1846f82/shadow_piano_native_audit_summary.json) |
| 2026-10-22 18:52:37 | `clear_view_v3_smoke` | 3150 | 7/8 | — | 680 | 790 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261022T185237Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v3_smoke_rep1_20261022T185237Z_0abe5c32992a4410a7cdf699d95f2179/shadow_piano_native_audit_summary.json) |
| 2026-10-22 19:30:56 | `clear_view_v3_smoke` | 3150 | 4/8 | — | 402 | 512 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261022T193056Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v3_smoke_rep1_20261022T193056Z_5704e82f2b23417fb40f0e82f878eb85/shadow_piano_native_audit_summary.json) |
| 2026-10-22 19:53:30 | `clear_view_v3_smoke` | 3150 | 4/8 | 8 | 774 | 884 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261022T195330Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v3_smoke_rep1_20261022T195330Z_3f6d5514c86841f8a588a0029f86ee4d/shadow_piano_native_audit_summary.json) |
| 2026-10-23 02:53:52 | `clear_view_v3_smoke` | 3150 | — | — | — | — | — | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261023T025352Z/run.log) · —（原生审计缺失） |
| 2026-10-23 03:06:22 | `clear_view_v3_smoke` | 3150 | 7/8 | 8 | 830 | 940 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261023T030622Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v3_smoke_rep1_20261023T030622Z_f3ce06834da34c878d483bcb5eceb6eb/shadow_piano_native_audit_summary.json) |
| 2026-10-23 03:17:44 | `clear_view_v3_smoke` | 3150 | — | — | — | — | — | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261023T031744Z/run.log) · —（原生审计缺失） |
| 2026-10-23 03:51:11 | `clear_view_v3_smoke` | 3150 | 4/8 | 8 | 732 | 842 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261023T035111Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v3_smoke_rep1_20261023T035111Z_c7b56fe9090641a0be4f125f169e5ed7/shadow_piano_native_audit_summary.json) |
| 2026-10-23 06:29:52 | `clear_view_v3_smoke` | 3150 | 7/8 | 8 | 825 | 935 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261023T062952Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v3_smoke_rep1_20261023T062952Z_b68ca2ae99cb4dc4983fc16b4302e34c/shadow_piano_native_audit_summary.json) |
| 2026-10-23 07:30:40 | `clear_view_v3_smoke` | 3150 | 4/8 | 8 | 772 | 882 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261023T073040Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v3_smoke_rep1_20261023T073040Z_1ab2401212c244afb30af9b55e00a6bd/shadow_piano_native_audit_summary.json) |
| 2026-10-23 08:12:46 | `4phys_smoke` | 3150 | 0/8 | 0 | — | 150 | 否 | — | — · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_4phys_seed3150_smoke_rep1_20261023T081246Z_99ebe56ab2914c7f9354df69e0ce7098/shadow_piano_native_audit_summary.json) |
| 2026-10-23 08:23:31 | `4phys_gpu0_smoke` | 3150 | 6/8 | 8 | 1419 | 1639 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_4phys_seed3150_gpu0_smoke_rep1_20261023T082331Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_4phys_seed3150_gpu0_smoke_rep1_20261023T082331Z_b4cf934bbd644c08b29db9f05f1ae547/shadow_piano_native_audit_summary.json) |
| 2026-10-23 09:17:00 | `4phys_gpu0_smoke` | 3150 | 1/8 | 1 | 408 | 628 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_4phys_seed3150_gpu0_smoke_rep1_20261023T091700Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_4phys_seed3150_gpu0_smoke_rep1_20261023T091700Z_19cb148d613c496a847aa0fa0bd4b0f6/shadow_piano_native_audit_summary.json) |
| 2026-10-23 09:33:50 | `4phys_gpu0_smoke` | 3150 | 6/8 | 8 | 1674 | 1894 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_4phys_seed3150_gpu0_smoke_rep1_20261023T093350Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_4phys_seed3150_gpu0_smoke_rep1_20261023T093350Z_bbd181b46e4e403ebc7cdfb23399804c/shadow_piano_native_audit_summary.json) |
| 2026-10-23 18:37:39 | `5d_area_rate_4phys_gpu0_smoke` | 3150 | — | — | — | — | — | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_5d_area_rate_4phys_seed3150_gpu0_smoke_rep1_20261023T183739Z/run.log) · —（原生审计缺失） |
| 2026-10-24 08:26:05 | `5d_area_rate_4phys_gpu0_smoke` | 3150 | 0/8 | 8 | 1234 | 1454 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_5d_area_rate_4phys_seed3150_gpu0_smoke_rep1_20261024T082605Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_5d_area_rate_4phys_seed3150_gpu0_smoke_rep1_20261024T082605Z_1a2d6f5593804d15a3d5c2d6addcb52f/shadow_piano_native_audit_summary.json) |
| 2026-10-24 09:34:16 | `5d_quality_fix_4phys_gpu0_smoke` | 3150 | 0/8 | 8 | 1287 | 1507 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_5d_quality_fix_4phys_seed3150_gpu0_smoke_rep1_20261024T093416Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_5d_quality_fix_4phys_seed3150_gpu0_smoke_rep1_20261024T093416Z_b08794b4d509457a9bc37ba2daa0b406/shadow_piano_native_audit_summary.json) |
| 2026-10-24 09:49:30 | `5d_quality_nocontact_fix_4phys_gpu7_smoke` | 3150 | 0/8 | 8 | 1770 | 1990 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_5d_quality_nocontact_fix_4phys_seed3150_gpu7_smoke_rep1_20261024T094930Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_5d_quality_nocontact_fix_4phys_seed3150_gpu7_smoke_rep1_20261024T094930Z_d7985ae4e67a4b7ba9a646564d8e16e0/shadow_piano_native_audit_summary.json) |
| 2026-10-24 16:22:18 | `5d_matched8_4phys_gpu7_smoke` | 3150 | 7/8 | 8 | 1615 | 1835 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_5d_matched8_4phys_seed3150_gpu7_smoke_rep1_20261024T162218Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_5d_matched8_4phys_seed3150_gpu7_smoke_rep1_20261024T162218Z_277d90e009d642189c3e12d3f09dbf5c/shadow_piano_native_audit_summary.json) |
| 2026-10-24 17:05:12 | `5d_matched8_repeat_4phys_gpu1_smoke` | 3150 | 6/8 | 8 | 1381 | 1601 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_5d_matched8_repeat_4phys_seed3150_gpu1_smoke_rep1_20261024T170512Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_5d_matched8_repeat_4phys_seed3150_gpu1_smoke_rep1_20261024T170512Z_41d0cd8c4b4d49458c4a457e5f024776/shadow_piano_native_audit_summary.json) |

| 2026-10-24 19:09:46 | `random6_online_skills` | 3300 | 6/6 | 6 | 1140 | 1360 | 是 | 0 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3300_random6_3300_3309_20261025_v1_smoke_rep1_20261024T190946Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3300_random6_3300_3309_20261025_v1_smoke_rep1_20261024T190946Z_658a08a508be470b95c15062f0f01052/shadow_piano_native_audit_summary.json) |
| 2026-10-24 19:24:38 | `random6_online_skills` | 3301 | 4/6 | 6 | 1160 | 1160 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3301_random6_3300_3309_20261025_v1_smoke_rep1_20261024T192438Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3301_random6_3300_3309_20261025_v1_smoke_rep1_20261024T192438Z_b4e0be99064f4697a5cb0bd1d1d90e20/shadow_piano_native_audit_summary.json) |
| 2026-10-24 19:30:33 | `random6_online_skills` | 3302 | 6/6 | 6 | 958 | 958 | 是 | 0 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3302_random6_3300_3309_20261025_v1_smoke_rep1_20261024T193033Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3302_random6_3300_3309_20261025_v1_smoke_rep1_20261024T193033Z_924a2b172ed448e1a60dcd6404881204/shadow_piano_native_audit_summary.json) |
| 2026-10-24 19:35:40 | `random6_online_skills` | 3303 | 4/6 | 6 | 1127 | 1127 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3303_random6_3300_3309_20261025_v1_smoke_rep1_20261024T193540Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3303_random6_3300_3309_20261025_v1_smoke_rep1_20261024T193540Z_e64c7a47f4004234a7989c3b49a0feec/shadow_piano_native_audit_summary.json) |
| 2026-10-24 19:41:23 | `random6_online_skills` | 3304 | 4/6 | 6 | 1234 | 1234 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3304_random6_3300_3309_20261025_v1_smoke_rep1_20261024T194123Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3304_random6_3300_3309_20261025_v1_smoke_rep1_20261024T194123Z_0b89d303622448e19263d29883ca60ec/shadow_piano_native_audit_summary.json) |
| 2026-10-24 19:47:23 | `random6_online_skills` | 3305 | 4/6 | 6 | 1061 | 1061 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3305_random6_3300_3309_20261025_v1_smoke_rep1_20261024T194723Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3305_random6_3300_3309_20261025_v1_smoke_rep1_20261024T194723Z_a922757b11c54667bd6d4b8b650be7d4/shadow_piano_native_audit_summary.json) |
| 2026-10-24 19:52:43 | `random6_online_skills` | 3306 | 6/6 | 6 | 838 | 838 | 是 | 0 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3306_random6_3300_3309_20261025_v1_smoke_rep1_20261024T195243Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3306_random6_3300_3309_20261025_v1_smoke_rep1_20261024T195243Z_22410ad3fc354d9d9b90964115911328/shadow_piano_native_audit_summary.json) |
| 2026-10-24 19:58:19 | `random6_online_skills` | 3307 | 4/6 | 6 | 1335 | 1335 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3307_random6_3300_3309_20261025_v1_smoke_rep1_20261024T195819Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3307_random6_3300_3309_20261025_v1_smoke_rep1_20261024T195819Z_bc520199faa844b0bfd91e9ff3cbd824/shadow_piano_native_audit_summary.json) |
| 2026-10-24 20:04:50 | `random6_online_skills` | 3308 | 5/6 | 6 | 1188 | 1188 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3308_random6_3300_3309_20261025_v1_smoke_rep1_20261024T200450Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3308_random6_3300_3309_20261025_v1_smoke_rep1_20261024T200450Z_14a3a0a83a794cc28da09096132616f3/shadow_piano_native_audit_summary.json) |
| 2026-10-24 20:10:48 | `random6_online_skills` | 3309 | 6/6 | 6 | 993 | 993 | 是 | 0 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3309_random6_3300_3309_20261025_v1_smoke_rep1_20261024T201048Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3309_random6_3300_3309_20261025_v1_smoke_rep1_20261024T201048Z_117fae6989084a1bb0c39518d03561d8/shadow_piano_native_audit_summary.json) |

| 2026-10-25 05:53:15 | `random6_empty_online_skills` | 3300 | 5/6 | 6 | 942 | 1162 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3300_random6_empty_3300_3309_20261025T055314Z_smoke_rep1_20261025T055315Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3300_random6_empty_3300_3309_20261025T055314Z_smoke_rep1_20261025T055315Z_0dfcbafad7d140fb931eaf6c9fffa43d/shadow_piano_native_audit_summary.json) |
| 2026-10-25 06:00:04 | `random6_empty_online_skills` | 3301 | 5/6 | 6 | 1115 | 1115 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3301_random6_empty_3300_3309_20261025T055314Z_smoke_rep1_20261025T060004Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3301_random6_empty_3300_3309_20261025T055314Z_smoke_rep1_20261025T060004Z_6a119f52ec9e4e298543e717af1dd13b/shadow_piano_native_audit_summary.json) |
| 2026-10-25 06:05:47 | `random6_empty_online_skills` | 3302 | 0/6 | 6 | 823 | 823 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3302_random6_empty_3300_3309_20261025T055314Z_smoke_rep1_20261025T060547Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3302_random6_empty_3300_3309_20261025T055314Z_smoke_rep1_20261025T060547Z_2c659827ca8047cfa04e6288e50000c5/shadow_piano_native_audit_summary.json) |
| 2026-10-25 06:11:18 | `random6_empty_online_skills` | 3303 | 4/6 | 6 | 1300 | 1300 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3303_random6_empty_3300_3309_20261025T055314Z_smoke_rep1_20261025T061118Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3303_random6_empty_3300_3309_20261025T055314Z_smoke_rep1_20261025T061118Z_fdf786e7189b41eaac6f369f765b2808/shadow_piano_native_audit_summary.json) |
| 2026-10-25 06:17:44 | `random6_empty_online_skills` | 3304 | 4/6 | 6 | 1267 | 1267 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3304_random6_empty_3300_3309_20261025T055314Z_smoke_rep1_20261025T061744Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3304_random6_empty_3300_3309_20261025T055314Z_smoke_rep1_20261025T061744Z_c86b5be13a2e451093d6e459cbe8f4a5/shadow_piano_native_audit_summary.json) |
| 2026-10-25 06:24:36 | `random6_empty_online_skills` | 3305 | 4/6 | 6 | 1135 | 1135 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3305_random6_empty_3300_3309_20261025T055314Z_smoke_rep1_20261025T062436Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3305_random6_empty_3300_3309_20261025T055314Z_smoke_rep1_20261025T062436Z_f180cc1ab13a43199edd40890b17da31/shadow_piano_native_audit_summary.json) |
| 2026-10-25 06:30:21 | `random6_empty_online_skills` | 3306 | 5/6 | 6 | 1156 | 1156 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3306_random6_empty_3300_3309_20261025T055314Z_smoke_rep1_20261025T063021Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3306_random6_empty_3300_3309_20261025T055314Z_smoke_rep1_20261025T063021Z_1a392f0bc1474ddf971fad584bde8b25/shadow_piano_native_audit_summary.json) |
| 2026-10-25 06:36:07 | `random6_empty_online_skills` | 3307 | 6/6 | 6 | 1257 | 1257 | 是 | 0 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3307_random6_empty_3300_3309_20261025T055314Z_smoke_rep1_20261025T063607Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3307_random6_empty_3300_3309_20261025T055314Z_smoke_rep1_20261025T063607Z_956a790384df46959f3f766b2730c58d/shadow_piano_native_audit_summary.json) |
| 2026-10-25 06:42:43 | `random6_empty_online_skills` | 3308 | 5/6 | 6 | 1149 | 1149 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3308_random6_empty_3300_3309_20261025T055314Z_smoke_rep1_20261025T064243Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3308_random6_empty_3300_3309_20261025T055314Z_smoke_rep1_20261025T064243Z_4cbd200a96784ee88a938c9b29b32c6c/shadow_piano_native_audit_summary.json) |
| 2026-10-25 06:48:39 | `random6_empty_online_skills` | 3309 | 5/6 | 6 | 990 | 990 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3309_random6_empty_3300_3309_20261025T055314Z_smoke_rep1_20261025T064839Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3309_random6_empty_3300_3309_20261025T055314Z_smoke_rep1_20261025T064839Z_974d3086f848431d86783b579536f2b4/shadow_piano_native_audit_summary.json) |

| 2026-10-25 09:43:18 | `executable_code_memory` | 3300 | 5/6 | 6 | 1375 | 1595 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_code_memory_seed3300_code_memory_3300_3309_20261025T094300Z_smoke_rep1_20261025T094318Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_code_memory_seed3300_code_memory_3300_3309_20261025T094300Z_smoke_rep1_20261025T094318Z_97abaf863ff84cde9217a250e86fa18e/shadow_piano_native_audit_summary.json) |
| 2026-10-25 09:57:28 | `executable_code_memory` | 3301 | 5/6 | 6 | 1157 | 1157 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_code_memory_seed3301_code_memory_3300_3309_20261025T094300Z_smoke_rep1_20261025T095728Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_code_memory_seed3301_code_memory_3300_3309_20261025T094300Z_smoke_rep1_20261025T095728Z_50c25eb3bc654b028263a6ab61e3903e/shadow_piano_native_audit_summary.json) |
| 2026-10-25 10:04:35 | `executable_code_memory` | 3302 | 6/6 | 6 | 1159 | 1159 | 是 | 0 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_code_memory_seed3302_code_memory_3300_3309_20261025T094300Z_smoke_rep1_20261025T100435Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_code_memory_seed3302_code_memory_3300_3309_20261025T094300Z_smoke_rep1_20261025T100435Z_43efbbae081b415582ae357ad4baf42f/shadow_piano_native_audit_summary.json) |
| 2026-10-25 10:10:42 | `executable_code_memory` | 3303 | 5/6 | 6 | 1272 | 1272 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_code_memory_seed3303_code_memory_3300_3309_20261025T094300Z_smoke_rep1_20261025T101042Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_code_memory_seed3303_code_memory_3300_3309_20261025T094300Z_smoke_rep1_20261025T101042Z_c85716e8b3e64f96aa26d290a3c40803/shadow_piano_native_audit_summary.json) |
| 2026-10-25 10:17:23 | `executable_code_memory` | 3304 | 6/6 | 6 | 1242 | 1242 | 是 | 0 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_code_memory_seed3304_code_memory_3300_3309_20261025T094300Z_smoke_rep1_20261025T101723Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_code_memory_seed3304_code_memory_3300_3309_20261025T094300Z_smoke_rep1_20261025T101723Z_4df0235b84cf42c08d5c70f071bee7f9/shadow_piano_native_audit_summary.json) |
| 2026-10-25 10:23:49 | `executable_code_memory_interrupted` | 3305 | — | — | — | — | — | — | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_code_memory_seed3305_code_memory_3300_3309_20261025T094300Z_smoke_rep1_20261025T102349Z/run.log) · 用户暂停中断，缺少已关闭的原生审计，不按0计 |
| 2026-10-25 10:53:13 | `seed3302_readonly_diagnostic` | 3302 | 6/6 | 6 | 1267 | 1267 | 是 | 0 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_3302_readonly_diag_20261025T105313Z_smoke_rep1_20261025T105313Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_3302_readonly_diag_20261025T105313Z_smoke_rep1_20261025T105313Z_047a314839b747b3a67f9042a8e18974/shadow_piano_native_audit_summary.json) |

| 2026-10-25 13:23:31 | `paired_memory_A` | 3300 | 1/6 | 1 | 215 | 215 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/piano_abc_A_seed3300_paired_3300_3309_20261025T132326Z_smoke_rep1_20261025T132331Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/piano_abc_A_seed3300_paired_3300_3309_20261025T132326Z_smoke_rep1_20261025T132331Z_63bbebdb5fa84091bbdf2029c7c83d44/shadow_piano_native_audit_summary.json)；API故障中断，余下音未尝试 |
| 2026-10-25 13:27:09 | `paired_memory_B` | 3300 | 5/6 | 6 | 1334 | 1334 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/piano_abc_B_seed3300_paired_3300_3309_20261025T132326Z_smoke_rep1_20261025T132709Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/piano_abc_B_seed3300_paired_3300_3309_20261025T132326Z_smoke_rep1_20261025T132709Z_6bb46a6356034fe3937866d068855425/shadow_piano_native_audit_summary.json) |
| 2026-10-25 13:33:51 | `paired_memory_C` | 3300 | 1/6 | 1 | 215 | 215 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/piano_abc_C_seed3300_paired_3300_3309_20261025T132326Z_smoke_rep1_20261025T133351Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/piano_abc_C_seed3300_paired_3300_3309_20261025T132326Z_smoke_rep1_20261025T133351Z_fe096c9c7ce84120b636b7c3b768f004/shadow_piano_native_audit_summary.json)；API故障中断，余下音未尝试 |

| 2026-10-25 14:35:31 | `model_override_aborted` | 3300 | —/6 | — | 131 | 131 | — | 130 | [日志](/mnt/data/ljz/opentacbench/capx-runs/piano_abc_A_seed3300_rerun3300_gpt61sol_20261025T143526Z_smoke_rep1_20261025T143531Z/run.log)；启动模型覆盖错误，已中止并排除正式对照 |
| 2026-10-25 14:40:46 | `model_override_aborted` | 3300 | —/6 | — | 580 | 580 | — | 130 | [日志](/mnt/data/ljz/opentacbench/capx-runs/piano_abc_A_seed3300_rerun3300_gpt61sol_20261025T144041Z_smoke_rep1_20261025T144046Z/run.log)；启动模型覆盖错误，已中止并排除正式对照 |
| 2026-10-25 14:53:50 | `paired_gpt61sol_A` | 3300 | 1/6 | 1 | 331 | 331 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/piano_abc_A_seed3300_rerun3300_gpt61sol_20261025T145345Z_smoke_rep1_20261025T145350Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/piano_abc_A_seed3300_rerun3300_gpt61sol_20261025T145345Z_smoke_rep1_20261025T145350Z_30a7ef66a75e46b9bf08da6628cfb961/shadow_piano_native_audit_summary.json)；生成代码异常，剩余音未尝试 |
| 2026-10-25 14:57:08 | `paired_gpt61sol_B` | 3300 | 1/6 | 1 | 185 | 185 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/piano_abc_B_seed3300_rerun3300_gpt61sol_20261025T145345Z_smoke_rep1_20261025T145708Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/piano_abc_B_seed3300_rerun3300_gpt61sol_20261025T145345Z_smoke_rep1_20261025T145708Z_498461a1b5d842638f1ef1011ce29cb3/shadow_piano_native_audit_summary.json)；API响应超时，剩余音未尝试 |
| 2026-10-25 15:00:46 | `paired_gpt61sol_C` | 3300 | 1/6 | 1 | 185 | 185 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/piano_abc_C_seed3300_rerun3300_gpt61sol_20261025T145345Z_smoke_rep1_20261025T150046Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/piano_abc_C_seed3300_rerun3300_gpt61sol_20261025T145345Z_smoke_rep1_20261025T150046Z_6f0ad7ab918d410fbffddf71fb27b9c1/shadow_piano_native_audit_summary.json)；生成代码异常，剩余音未尝试 |

完整音序、token、技能库版本和视频参见 [本次十组报告](/mnt/a/ljz/cap/t-cap/cap-x/docs/opentac-shadow-piano-random6-online-skills.md)。

空库起点对照及失败诊断参见 [本轮报告](/mnt/a/ljz/cap/t-cap/cap-x/docs/opentac-shadow-piano-empty-memory-results.md)。

3302单次诊断与修复详见 [报告](/mnt/a/ljz/cap/t-cap/cap-x/docs/opentac-shadow-piano-seed3302-diagnostic.md)。

三组配对实验逐seed表与API故障说明：[实时结果](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/paired_3300_3309_20261025T132326Z/results.md)。首个seed已记录三组，剩余27次因API可用性问题尚未启动。

3300三组改用gpt-6.1-sol的重跑：[结果表](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/rerun3300_gpt61sol_20261025T145345Z/results.md)。三组均仅第一个音通过；A/C为生成代码错误，B为API超时。两次模型覆盖错误的启动单独保留并排除。

## 更新方式

在现有 CaP-X 工作区运行：

```bash
python3 scripts/opentac/build_shadow_piano_experiment_ledger.py
```

默认读取 `/mnt/data/ljz/opentacbench/capx-runs/shadow_*/run.log` 和 `/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1` 下的审计目录，重新生成本文件。此脚本只读实验产物，不运行仿真，不把私有逐帧力标签写入文档或运行时提示词。

| 2026-10-25 17:32:36 | `independent_v2_A_initial_API_timeout` | 3300 | 0/6（未尝试） | 0 | 0 | 0 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/piano_abc_A_seed3300_independent_3300_3301_gpt61_20261025T173235Z_smoke_rep1_20261025T173236Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/piano_abc_A_seed3300_independent_3300_3301_gpt61_20261025T173235Z_smoke_rep1_20261025T173236Z_774be8a797834f0d989e5e2c72e53321/shadow_piano_native_audit_summary.json) |

本轮独立初始化协议 v2 原计划3300/3301各A/B/C共六组。A/3300 首次请求无代理 `gpt-6.1-sol` 超时，无生成代码、无按键动作；API请求1次，用量未知。其余五组未启动，不计为失败或零分。队列暂停，无自动补跑。不得与此前共享A初始源码的三组记录合并为同一协议。

| 2026-10-25 18:43:07 | `independent_v2_A_preflight_ok_initial120_timeout` | 3300 | 0/6（未尝试） | 0 | 0 | 0 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/piano_abc_A_seed3300_independent_preflight_3300_3301_20261025T184303Z_smoke_rep1_20261025T184307Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/piano_abc_A_seed3300_independent_preflight_3300_3301_20261025T184303Z_smoke_rep1_20261025T184307Z_15f86cc6463f40478865f08e0b8f61d0/shadow_piano_native_audit_summary.json) |

该运行原脚本preflight通过，但首次生成仍被代码记忆模块共用的120秒修复限时打断。没有源码返回，无控制动作，不属于策略失败。后续新实验只将首次生成限时与原启动脚本300秒对齐，修复仍限120秒；不使用该运行挑选源码。

| 2026-10-25 18:47:56 | `independent_v2_initial300_A` | 3300 | 6/6 | 6 | 1186 | 1186 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/piano_abc_A_seed3300_independent_initial300_3300_3301_20261025T184748Z_smoke_rep1_20261025T184756Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/piano_abc_A_seed3300_independent_initial300_3300_3301_20261025T184748Z_smoke_rep1_20261025T184756Z_7bf6f9c9c0124e8ba85c338d4e783607/shadow_piano_native_audit_summary.json) |

| 2026-10-25 18:59:18 | `independent_v2_initial300_B` | 3300 | 5/6 | 6 | 1306 | 1306 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/piano_abc_B_seed3300_independent_initial300_3300_3301_20261025T184748Z_smoke_rep1_20261025T185918Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/piano_abc_B_seed3300_independent_initial300_3300_3301_20261025T184748Z_smoke_rep1_20261025T185918Z_9beabdb0efc049a8a36097bfc2000883/shadow_piano_native_audit_summary.json) |

| 2026-10-25 19:11:51 | `independent_v2_initial300_C` | 3300 | 0/6（未尝试） | 0 | 0 | 0 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/piano_abc_C_seed3300_independent_initial300_3300_3301_20261025T184748Z_smoke_rep1_20261025T191151Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/piano_abc_C_seed3300_independent_initial300_3300_3301_20261025T184748Z_smoke_rep1_20261025T191151Z_b66ff3e427ab4610ab9c7bb9c6cd2d4c/shadow_piano_native_audit_summary.json) |

该批次A/B/C独立初始化，首次生成等待300秒，修复120秒，原脚本preflight开启；因C修复超时停在3300结束后，3301未运行。A六音合格但END缺失；B错第二音，原因是释放恢复时重弹F4；C无动作。详见[批次分析](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/independent_initial300_3300_3301_20261025T184748Z/analysis.json)。


### 独立配置v4：A3300、A3301（UTC）

| 时间 | 实验版本 | Seed | 原生成功音数 | 尝试数 | 生成代码动作步 | 总原生动作步 | END | 退出码 | 原始证据 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | --- |
| 2026-10-25 20:07:21 | `abc_v4_A_separate_config` | 3300 | 5/6 | 6 | 1303 | 1303 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/piano_abc_A_seed3300_abc_separate_3300_3301_20261025T200709Z_smoke_rep1_20261025T200721Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/piano_abc_A_seed3300_abc_separate_3300_3301_20261025T200709Z_smoke_rep1_20261025T200721Z_498ab9fcd0094e5f9178ca49b749025b/shadow_piano_native_audit_summary.json) |
| 2026-10-25 20:18:56 | `abc_v4_A_separate_config` | 3301 | 1/6 | 1 | 201 | 201 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/piano_abc_A_seed3301_abc_separate_3300_3301_20261025T200709Z_smoke_rep1_20261025T201856Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/piano_abc_A_seed3301_abc_separate_3300_3301_20261025T200709Z_smoke_rep1_20261025T201856Z_0cd1cbdd1d5a43358b42328bb074ff94/shadow_piano_native_audit_summary.json) |

A3301只尝试首音并原生合格，之后释放计数差一拍触发护栏，修复API超时。剩余5音未尝试，不归类为5次控制失败。A3300完成六次尝试，第二音wrong_key。两轮结束后整理调用均为0；B/C尚未运行。详见[本轮诊断](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/abc_separate_3300_3301_20261025T200709Z/diagnosis.md)。


### 原八音请求回归（UTC）

| 时间 | 实验版本 | Seed | 原生成功音数 | 尝试数 | 生成代码动作步 | 总原生动作步 | END | 退出码 | 原始证据 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | --- |
| 2026-10-26 01:49:43 | `old8_prompt_check_gpt61sol_matched5d` | 3150 | 5/8 | 8 | 1575 | 1795 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/piano_old8_prompt_check_seed3150_20261026T014940Z_smoke_rep1_20261026T014943Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/piano_old8_prompt_check_seed3150_20261026T014940Z_smoke_rep1_20261026T014943Z_60c67bb2d0924f189adec9b159a8eb06/shadow_piano_native_audit_summary.json) |

仅一条原八音普通生成入口：无代理，初次请求62.89秒、21717服务商报告token，1代码块、0修复/超时。失败为F4重、D#4重、E4重的原生incomplete_note。含220清场动作；不是与ABC严格同条件的成功率对照。详见[请求与结果对照](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/old8_prompt_check_20261026T014940Z/result.md)。


### 直接run_responses：A/B/C独立首轮（UTC）

| 时间 | 实验版本 | Seed | 原生成功音数 | 尝试数 | 生成代码动作步 | 总原生动作步 | END | 退出码 | 原始证据 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | --- |
| 2026-10-26 05:38:35 | `manual_direct_A_gpt61sol_300s` | 3300 | 4/6 | 6 | 1333 | 1333 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/manual_A_seed3300_smoke_rep1_20261026T053835Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/manual_A_seed3300_smoke_rep1_20261026T053835Z_1721d0f44bf544459da95679d7d656c2/shadow_piano_native_audit_summary.json) |
| 2026-10-26 05:47:41 | `manual_direct_B_gpt61sol_300s` | 3300 | 4/6 | 6 | 1421 | 1421 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/manual_B_seed3300_smoke_rep1_20261026T054741Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/manual_B_seed3300_smoke_rep1_20261026T054741Z_57ac5f5bdc794bf0949b2e790b013de3/shadow_piano_native_audit_summary.json) |
| 2026-10-26 05:56:06 | `manual_direct_C_gpt61sol_300s` | 3300 | 1/6 | 1 | 327 | 327 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/manual_C_seed3300_smoke_rep1_20261026T055606Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/manual_C_seed3300_smoke_rep1_20261026T055606Z_6f72337259e2483e8d70d6dd2d4fbdd5/shadow_piano_native_audit_summary.json) |

六次生成/修复请求均正常返回，未发生超时，因此没有额外重试。每组均2代码块、1次修复。C只尝试首音，其他5音未尝试；三组均有公开释放计时拒绝。此为三次独立冷启动诊断，不是跨seed记忆复用对照。详见[完整统计与视频](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/manual_abc_direct_20261026T053755Z/results.md)。


### C组双seed：3300→3301（UTC）

| 时间 | 实验版本 | Seed | 原生成功音数 | 尝试数 | 生成代码动作步 | 总原生动作步 | END | 退出码 | 原始证据 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | --- |
| 2026-10-26 07:15:23 | `C_two_seed_existing_finish` | 3300 | 4/6 | 6 | 875 | 875 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/manual_C2_seed3300_20261026T071433Z_smoke_rep1_20261026T071523Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/manual_C2_seed3300_20261026T071433Z_smoke_rep1_20261026T071523Z_0653fab20665483f97a73acaaa7453da/shadow_piano_native_audit_summary.json) |
| 2026-10-26 07:25:17 | `C_two_seed_existing_finish` | 3301 | 5/6 | 6 | 1010 | 1010 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/manual_C2_seed3301_20261026T071433Z_smoke_rep1_20261026T072517Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/manual_C2_seed3301_20261026T071433Z_smoke_rep1_20261026T072517Z_4bf6a59ce96f4f95a6103c2325dc4ec5/shadow_piano_native_audit_summary.json) |

两轮间使用已有finish_episode。3300公开finalize仅5条，未入库；3301仍重新生成。3301公开6条完成后产生有效缓存，但没有追加第三轮验证复用。本次不是记忆提升的成功对照。无API超时，无额外重跑。详见[完整双seed报告](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/manual_c_two_seed_20261026T071433Z/results.md)。

### 部分经验归档与动作函数库：3300→3301（UTC）

| 时间 | 实验版本 | Seed | 原生成功音数 | 尝试数 | 生成代码动作步 | 总原生动作步 | END | 退出码 | 原始证据 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | --- |
| 2026-10-26 09:20:20 | `action_functions_preflight_503` | 3300（请求） | — | — | — | — | — | 1 | [终端](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/action_functions_3300_3301_20261026T092020Z/seed3300.terminal.log)；API预检503，未启动Kit，无原生审计 |
| 2026-10-26 09:22:14 | `action_functions_partial_memory` | 3300 | 1/6 | 1 | 335 | 335 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/action_functions_seed3300_retry1_20261026T092020Z_smoke_rep1_20261026T092214Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/action_functions_seed3300_retry1_20261026T092020Z_smoke_rep1_20261026T092214Z_3d740cb9ef664de3acfa9001dd1f326b/shadow_piano_native_audit_summary.json) |
| 2026-10-26 09:31:01 | `action_functions_partial_memory` | 3301 | 3/6 | 6 | 936 | 936 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/action_functions_seed3301_20261026T092020Z_smoke_rep1_20261026T093101Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/action_functions_seed3301_20261026T092020Z_smoke_rep1_20261026T093101Z_8f50181115a747f7a68f9dd3d623c099/shadow_piano_native_audit_summary.json) |

本轮从空代码/参数记忆开始，只复制公开键位缓存。3300首次生成105.40秒、修复106.07秒，两段代码均过早提交释放完成，公开结束0次；不完整经验与8个实验性函数保存，有效整段策略为空。3301首次生成97.86秒、无修复，公开结束6次并END；虽收到8个旧函数卡片，源码未引用任何旧函数，而是重写策略。因此1/6→3/6和API合计211.46→97.86秒不能归因于有效代码复用。3301后三个音私有诊断为incomplete_note，该判分不参与记忆更新。

两轮代码执行墙钟分别80.79和231.37秒，独立于API等待统计。HEAD与第三方视频均含音频流。GPU结束后只修正未使用旧函数的冗余拼装和索引过滤，未追加GPU试跑；无代理、gpt-6.1-sol、四物理步动作和5D模型保持。完整逐请求、源码和视频见[结果报告](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/action_functions_3300_3301_20261026T092020Z/results.md)，另保存[上一轮十seed请求耗时](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/action_functions_3300_3301_20261026T092020Z/previous10_request_timing.md)。

### A组seed0–4：API403中断，尚未完成（UTC）

计划显示seed0–4对应原生3300–3304。使用已有A独立配置，各seed空源码/参数记忆，关闭文字与动作函数库，仅共享公开键位缓存；gpt-6.1-sol、low、GPU0串行，无代理。控制提示、5D模型、四物理步动作和OpenTacBench保持。

| 时间 | 实验版本 | Seed | 原生成功音数 | 尝试数 | 生成代码动作步 | 总原生动作步 | END | 退出码 | 原始证据 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | --- |
| 2026-10-26 10:55:12 | `A_five_fresh_API403_incomplete` | 3300（显示0） | 1/6 | 1 | 185 | 185 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/A_five_seed3300_20261026T105417Z_smoke_rep1_20261026T105512Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/A_five_seed3300_20261026T105417Z_smoke_rep1_20261026T105512Z_409478c1b8834fbf8131ce8e7b6d660f/shadow_piano_native_audit_summary.json) |

首次生成HTTP200、68.65秒；公开释放检查失败后请求修复，服务商HTTP403拒绝，称客户端异常、要求标准Codex客户端。失败修复等待约1.22秒来自run.log:170，未返回usage；总token未知，不将缺失调用记0。执行源码1代码块，代码执行46.19秒。该行属于API基础设施中断，不能作为完整A组成功率或效率样本；后五音未执行，原生3301–3304尚未启动，无额外重试。HEAD和第三方视频均保存；[五组当前状态和全部产物](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/A_fresh_3300_3304_20261026T105417Z/results.md)。

### 直接 run_responses：A 组五 seed 及 API 中断尝试（UTC）

显示 seed 0–4 对应原生 3300–3304。使用已有 ./run_responses 逐条串行运行，未新增脚本。每个 seed/尝试使用独立 A fresh 源码与参数记忆；文字技能和动作函数库关闭，整理 API 调用为 0，仅共享公开键盘端点标定。GPU0、gpt-6.1-sol、low、无代理，四物理步与冻结 5D 模型保持。下列时间为终端结束 UTC，区别于历史行的目录启动时间。

| 终端结束 UTC | 实验版本 | Seed/attempt | 原生成功音数 | 尝试数 | 生成代码动作步 | 总原生动作步 | END | 退出码 | 原始证据 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | --- |
| 2026-10-26 14:02:45 | `A_direct_403_infrastructure` | 3300/1 | 0/6 | 0 | 0 | 0 | 否 | 1 | [终端](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/manual_A_0_4_20261026T135957Z/seed3300.attempt1.terminal.log) · [日志](/mnt/data/ljz/opentacbench/capx-runs/direct_A_seed3300_attempt1_20261026T135957Z_smoke_rep1_20261026T140153Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/direct_A_seed3300_attempt1_20261026T135957Z_smoke_rep1_20261026T140153Z_c2d70806227f45f69abff226cfe588a7/shadow_piano_native_audit_summary.json) · API 中断，不纳入完整策略样本 |
| 2026-10-26 14:06:08 | `A_direct_403_infrastructure` | 3300/2 | — | — | — | — | — | 1 | [终端](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/manual_A_0_4_20261026T135957Z/seed3300.attempt2.terminal.log) · API 中断，不纳入完整策略样本 |
| 2026-10-26 14:15:29 | `A_direct_fresh_5seed` | 3300/3 | 1/6 | 1 | 280 | 280 | 否 | 1 | [终端](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/manual_A_0_4_20261026T135957Z/seed3300.attempt3.terminal.log) · [日志](/mnt/data/ljz/opentacbench/capx-runs/direct_A_seed3300_attempt3_20261026T135957Z_smoke_rep1_20261026T141124Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/direct_A_seed3300_attempt3_20261026T135957Z_smoke_rep1_20261026T141124Z_d814ad4848ed447dbc2c8bb7fc5c19bf/shadow_piano_native_audit_summary.json) |
| 2026-10-26 14:25:51 | `A_direct_fresh_5seed` | 3301/1 | 3/6 | 6 | 1113 | 1113 | 是 | 1 | [终端](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/manual_A_0_4_20261026T135957Z/seed3301.attempt1.terminal.log) · [日志](/mnt/data/ljz/opentacbench/capx-runs/direct_A_seed3301_attempt1_20261026T135957Z_smoke_rep1_20261026T141903Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/direct_A_seed3301_attempt1_20261026T135957Z_smoke_rep1_20261026T141903Z_19329d644fc94842ac3e85d28b9604e9/shadow_piano_native_audit_summary.json) |
| 2026-10-26 14:28:19 | `A_direct_403_infrastructure` | 3302/1 | 0/6 | 0 | 0 | 0 | 否 | 1 | [终端](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/manual_A_0_4_20261026T135957Z/seed3302.attempt1.terminal.log) · [日志](/mnt/data/ljz/opentacbench/capx-runs/direct_A_seed3302_attempt1_20261026T135957Z_smoke_rep1_20261026T142728Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/direct_A_seed3302_attempt1_20261026T135957Z_smoke_rep1_20261026T142728Z_26c7d8fd874d4baeacad1f953c335ccb/shadow_piano_native_audit_summary.json) · API 中断，不纳入完整策略样本 |
| 2026-10-26 14:37:11 | `A_direct_fresh_5seed` | 3302/2 | 4/6 | 6 | 1209 | 1209 | 是 | 1 | [终端](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/manual_A_0_4_20261026T135957Z/seed3302.attempt2.terminal.log) · [日志](/mnt/data/ljz/opentacbench/capx-runs/direct_A_seed3302_attempt2_20261026T135957Z_smoke_rep1_20261026T143004Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/direct_A_seed3302_attempt2_20261026T135957Z_smoke_rep1_20261026T143004Z_2f5fd00ddcf4494c8371f555ed0d12ff/shadow_piano_native_audit_summary.json) |
| 2026-10-26 14:38:34 | `A_direct_403_infrastructure` | 3303/1 | — | — | — | — | — | 1 | [终端](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/manual_A_0_4_20261026T135957Z/seed3303.attempt1.terminal.log) · API 中断，不纳入完整策略样本 |
| 2026-10-26 14:48:19 | `A_direct_fresh_5seed` | 3303/2 | 2/6 | 6 | 1245 | 1245 | 是 | 1 | [终端](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/manual_A_0_4_20261026T135957Z/seed3303.attempt2.terminal.log) · [日志](/mnt/data/ljz/opentacbench/capx-runs/direct_A_seed3303_attempt2_20261026T135957Z_smoke_rep1_20261026T144047Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/direct_A_seed3303_attempt2_20261026T135957Z_smoke_rep1_20261026T144047Z_68314fce3f1742089b87674b5469d4c8/shadow_piano_native_audit_summary.json) |
| 2026-10-26 14:56:49 | `A_direct_fresh_5seed` | 3304/1 | 4/6 | 6 | 1119 | 1119 | 是 | 1 | [终端](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/manual_A_0_4_20261026T135957Z/seed3304.attempt1.terminal.log) · [日志](/mnt/data/ljz/opentacbench/capx-runs/direct_A_seed3304_attempt1_20261026T135957Z_smoke_rep1_20261026T145029Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/direct_A_seed3304_attempt1_20261026T135957Z_smoke_rep1_20261026T145029Z_e4ca85ff70d14e54b3d5e71737f2dd88/shadow_piano_native_audit_summary.json) |

五组正常执行结束结果依次为 3300：1/6、280 动作、2 代码块；3301：3/6、1113 动作、2 代码块；3302：4/6、1209 动作、2 代码块；3303：2/6、1245 动作、2 代码块；3304：4/6、1119 动作、1 代码块。清场动作均为 0；3300 只尝试首音，后五音未尝试，未 END；其余四组尝试六音并 END。普通任务失败未补跑。

3300 首段因目标弯曲 0.50324267 rad 超过 0.5 rad 边界，修复后又提前提交释放。3301–3303 首段均触发释放时间检查，再次生成后完成六次尝试；3303 生成代码按 8 次合格采样/60 Hz 计为 0.1333 秒，但采样之间只有 7 个间隔，实际 0.1166667 秒，小于 0.12 秒。3304 无代码异常。

四次 403 中断均保留，等待至少 60 秒后同 seed 重启；两次 preflight 未启动 Kit，另两次 reset 后零代码、零动作。所有正常策略请求均 HTTP200，没有任务结果筛选重跑。五组服务商 token 顶层和分项计数均不一致，accounting.status=needs_review，不作为可信计费/效率值；每轮初次/修复 API 墙钟及原始 usage 分开保存。

五组 HEAD/第三方共 10 份视频均含音频流。完整五行结果、全部尝试、逐请求时间和视频见[本轮报告](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/manual_A_0_4_20261026T135957Z/results.md)。OpenTacBench tracked diff 为空；未改控制提示、评分和力估计器。

### 直接 run_responses：B组五seed初始化受API403阻断（北京时间）

计划显示0–4（原生3300–3304），B独立生成并冻结首份源码，后续复用。使用已有./run_responses；GPU0、gpt-6.1-sol、low、无代理、冻结5D和四物理步。四次请求均403，未取得源码、未产生冻结版本，后四seed未启动。以下均为基础设施中断，不纳入有效B控制比较。

| 终端结束北京时间 | 实验版本 | Seed/attempt | 原生成功音数 | 尝试数 | 生成代码动作步 | 总原生动作步 | END | 退出码 | 原始证据 |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| 2026-10-26 23:16:28 +08:00 | `B_direct_API403_infrastructure` | 3300/1 | — | — | — | — | — | 1 | （终端产物已删除） |
| 2026-10-26 23:18:48 +08:00 | `B_direct_API403_infrastructure` | 3300/2 | 0/6 | 0 | 0 | 0 | 否 | 1 | （终端产物已删除） · （日志产物已删除） · （审计产物已删除） |
| 2026-10-26 23:21:08 +08:00 | `B_direct_API403_infrastructure` | 3300/3 | 0/6 | 0 | 0 | 0 | 否 | 1 | （终端产物已删除） · （日志产物已删除） · （审计产物已删除） |
| 2026-10-26 23:24:19 +08:00 | `B_direct_API403_infrastructure` | 3300/4 | 0/6 | 0 | 0 | 0 | 否 | 1 | （终端产物已删除） · （日志产物已删除） · （审计产物已删除） |

第一次预检403；后三次预检200、正式请求403。最后一次延长到至少120秒后重试，仍被拒绝。没有源码，因此不能建立B冻结版本；未启动3301–3304，也不使用A代码替代。所有缺失usage记未知。未新增脚本、未修改任务/提示/模型。详见（四次请求证据与五组待测状态产物已删除）。

清理说明（北京时间 2026-10-27 00:04:58）：用户要求删除本节四次失败运行。3个run目录、3个私有审计目录及1个实验目录已删除；历史故障数值保留，原始链接已移除。[清理回执](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/manual_B_0_4_restart_20261026T160458Z/cleanup_receipt.json)。

### 删除旧失败产物后重启B：预检403（北京时间）

| 终端结束时间 | 实验 | Seed | 原生成功音数 | 动作数 | 退出码 | 证据 |
|---|---|---:|---|---|---:|---|
| 2026-10-27 00:06:02 +08:00 | B_restart_preflight403 | 3300 | — | — | 1 | [终端](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/manual_B_0_4_restart_20261026T160458Z/seed3300.attempt1.terminal.log) |

按用户要求删除上一轮四次API403运行的7个目录，保留历史故障登记。现有./run_responses已重新执行一次，但预检仍403，未启动Kit，无源码/冻结版本，后四seed未运行。[重启状态与清理回执](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/manual_B_0_4_restart_20261026T160458Z/results.md)。未新增脚本，未改OpenTacBench或控制提示。

### 用户要求再次重试B：预检仍403（北京时间）

| 终端结束时间 | 实验 | Seed/attempt | 原生成功音数 | 动作数 | 退出码 | 证据 |
|---|---|---:|---|---|---:|---|
| 2026-10-27 00:59:40 +08:00 | B_retry_preflight403 | 3300/2 | — | — | 1 | [终端](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/manual_B_0_4_restart_20261026T160458Z/seed3300.attempt2.terminal.log) |

按原 ./run_responses、GPU0、gpt-6.1-sol、无代理重试一次。预检403，Kit未启动，无源码或冻结版本，3301–3304仍未运行。不作为有效控制失败。

### 纯client Responses/SSE：B/C各五seed（北京时间）

通过现有./run_responses串行，先B3300–3304、后C3300–3304。仅client协议及失败分类/记账改变，无Codex CLI；无代理、gpt-6.1-sol low、GPU0、四物理步和冻结5D。下面时间为终端结束北京时间。完整保留10次有效策略与4次基础设施中断，不汇总成功音总数。

| 终端结束时间 | 实验版本 | Seed/attempt | 原生成功音数 | 已尝试 | 生成/清场动作 | 总原生动作 | END | 退出码 | 原始证据 |
|---|---|---|---:|---:|---:|---:|---|---:|---|
| 2026-10-27 02:43:37 +08:00 | B_client_sse | 3300/1 | 4/6 | 6 | 1294/0 | 1294 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/client_sse_B_seed3300_attempt1_20261026T183404Z_smoke_rep1_20261026T183625Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/client_sse_B_seed3300_attempt1_20261026T183404Z_smoke_rep1_20261026T183625Z_c342487ac47f43ab9f0a7f0cc6ec604a/shadow_piano_native_audit_summary.json) |
| 2026-10-27 02:45:20 +08:00 | B_client_sse | 3301/1 | 1/6 | 1 | 201/0 | 201 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/client_sse_B_seed3301_attempt1_20261026T183404Z_smoke_rep1_20261026T184338Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/client_sse_B_seed3301_attempt1_20261026T183404Z_smoke_rep1_20261026T184338Z_be76f422f6b04a8a9047f3247aa7e8ac/shadow_piano_native_audit_summary.json)；基础设施中断，排除有效策略比较 |
| 2026-10-27 02:53:00 +08:00 | B_client_sse | 3301/2 | 5/6 | 6 | 1335/0 | 1335 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/client_sse_B_seed3301_attempt2_20261026T183404Z_smoke_rep1_20261026T184621Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/client_sse_B_seed3301_attempt2_20261026T183404Z_smoke_rep1_20261026T184621Z_6b7915777e684ef69d7d4dc543ff55de/shadow_piano_native_audit_summary.json) |
| 2026-10-27 03:00:13 +08:00 | B_client_sse | 3302/1 | 5/6 | 6 | 1420/0 | 1420 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/client_sse_B_seed3302_attempt1_20261026T183404Z_smoke_rep1_20261026T185301Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/client_sse_B_seed3302_attempt1_20261026T183404Z_smoke_rep1_20261026T185301Z_c7e67e4da0dc4a5f81348431b980c15f/shadow_piano_native_audit_summary.json) |
| 2026-10-27 03:06:26 +08:00 | B_client_sse | 3303/1 | 4/6 | 6 | 1213/0 | 1213 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/client_sse_B_seed3303_attempt1_20261026T183404Z_smoke_rep1_20261026T190014Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/client_sse_B_seed3303_attempt1_20261026T183404Z_smoke_rep1_20261026T190014Z_16f6c00d94f949d6b3a136c1915e4570/shadow_piano_native_audit_summary.json) |
| 2026-10-27 03:13:32 +08:00 | B_client_sse | 3304/1 | 5/6 | 6 | 1451/0 | 1451 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/client_sse_B_seed3304_attempt1_20261026T183404Z_smoke_rep1_20261026T190627Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/client_sse_B_seed3304_attempt1_20261026T183404Z_smoke_rep1_20261026T190627Z_7bf697b801d7438ab66cf3df27318173/shadow_piano_native_audit_summary.json) |
| 2026-10-27 03:14:27 +08:00 | C_client_sse | 3300/1 | 0/6 | 0 | 0/0 | 0 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/client_sse_C_seed3300_attempt1_20261026T183404Z_smoke_rep1_20261026T191333Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/client_sse_C_seed3300_attempt1_20261026T183404Z_smoke_rep1_20261026T191333Z_8d97cedf3a724cd5835d9ef35be91c48/shadow_piano_native_audit_summary.json)；基础设施中断，排除有效策略比较 |
| 2026-10-27 03:16:31 +08:00 | C_client_sse | 3301/1 | 0/6 | 0 | 171/0 | 171 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/client_sse_C_seed3301_attempt1_20261026T183404Z_smoke_rep1_20261026T191428Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/client_sse_C_seed3301_attempt1_20261026T183404Z_smoke_rep1_20261026T191428Z_c499aac355c148b29b9b12a90496e599/shadow_piano_native_audit_summary.json)；基础设施中断，排除有效策略比较 |
| 2026-10-27 03:17:21 +08:00 | C_client_sse | 3302/1 | 0/6 | 0 | 0/0 | 0 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/client_sse_C_seed3302_attempt1_20261026T183404Z_smoke_rep1_20261026T191632Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/client_sse_C_seed3302_attempt1_20261026T183404Z_smoke_rep1_20261026T191632Z_53c01aee4df7468683ef971274211d58/shadow_piano_native_audit_summary.json)；基础设施中断，排除有效策略比较 |
| 2026-10-27 03:27:10 +08:00 | C_client_sse | 3300/2 | 1/6 | 1 | 278/0 | 278 | 否 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/client_sse_C_seed3300_attempt2_20261026T183404Z_smoke_rep1_20261026T192420Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/client_sse_C_seed3300_attempt2_20261026T183404Z_smoke_rep1_20261026T192420Z_518abeac802e4fcd82261afc3bc0ed8c/shadow_piano_native_audit_summary.json) |
| 2026-10-27 03:33:08 +08:00 | C_client_sse | 3301/2 | 4/6 | 6 | 1105/0 | 1105 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/client_sse_C_seed3301_attempt2_20261026T183404Z_smoke_rep1_20261026T192711Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/client_sse_C_seed3301_attempt2_20261026T183404Z_smoke_rep1_20261026T192711Z_df6ff54f841349dea3dfad8b9c5186d0/shadow_piano_native_audit_summary.json) |
| 2026-10-27 03:37:59 +08:00 | C_client_sse | 3302/2 | 3/6 | 6 | 982/0 | 982 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/client_sse_C_seed3302_attempt2_20261026T183404Z_smoke_rep1_20261026T193309Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/client_sse_C_seed3302_attempt2_20261026T183404Z_smoke_rep1_20261026T193309Z_fc7569cc75df4571a41ea42ff1ebdfb5/shadow_piano_native_audit_summary.json) |
| 2026-10-27 03:43:05 +08:00 | C_client_sse | 3303/1 | 3/6 | 6 | 1051/0 | 1051 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/client_sse_C_seed3303_attempt1_20261026T183404Z_smoke_rep1_20261026T193800Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/client_sse_C_seed3303_attempt1_20261026T183404Z_smoke_rep1_20261026T193800Z_b1d78912a5034c23ad8666fedd7374d7/shadow_piano_native_audit_summary.json) |
| 2026-10-27 03:48:42 +08:00 | C_client_sse | 3304/1 | 4/6 | 6 | 1158/0 | 1158 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/client_sse_C_seed3304_attempt1_20261026T183404Z_smoke_rep1_20261026T194306Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/client_sse_C_seed3304_attempt1_20261026T183404Z_smoke_rep1_20261026T194306Z_ee44bf3e7bee48b986c0f54dbc37165e/shadow_piano_native_audit_summary.json) |

B五组成功音数依次4/6、5/6、5/6、4/6、5/6；均六音尝试并END，原始代码被冻结，后四组初始生成0，但每组临时修复1。C依次1/6、4/6、3/6、3/6、4/6；3300只尝试一音且未END，其余六音并END。C3301形成有效修复版，3302–3304均1代码块、零模型请求及零token，代码和参数版本保持；对应成功数低于B，不能称控制策略提升。安全结束的alignment_lost/uncertain没有触发后续优化。

C首次上游SSE错误最初漏分类，导致3301/3302提前启动；原始记录全部标为基础设施中断，原候选保留隔离，修正端点错误分类后从3300恢复。数据不按原生得分筛选重跑；普通C3300代码失败不补跑。B/C有效样本请求构造与控制相同，只有端点错误分类/记账修复；25项回归通过。服务商token计数needs_review，保留原始值；无调用的0为已知0。

HEAD及第三方共20份有效运行视频均包含音频流，真值及原生分数仅用于私有报告；记忆更新在读取私有审计前执行。详见[十行结果、逐请求耗时、版本及全部产物](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/client_sse_BC_3300_3304_20261026T183404Z/results.md)。

### A与B/C相同seed的补充对照（不新增实验）

A来自manual_A_0_4_20261026T135957Z，B/C来自client_sse_BC_3300_3304_20261026T183404Z。这里只补充历史对照，不增加运行数或审计数。三组音序、轻重、持音、控制提示、四物理步、5D模型及任务版本一致；A旧client、B/C新SSE协议是已知混杂。

| seed | A成功/6 | B成功/6 | C成功/6 | A/B/C动作 | A/B/C API秒 |
|---|---:|---:|---:|---|---|
| 3300 | 1/6 | 4/6 | 1/6 | 280 / 1294 / 278 | 123.11 / 61.27 / 48.98 |
| 3301 | 3/6 | 5/6 | 4/6 | 1113 / 1335 / 1105 | 91.61 / 28.82 / 40.20 |
| 3302 | 4/6 | 5/6 | 3/6 | 1209 / 1420 / 982 | 93.30 / 29.99 / 0.00 |
| 3303 | 2/6 | 4/6 | 3/6 | 1245 / 1213 / 1051 | 100.77 / 30.37 / 0.00 |
| 3304 | 4/6 | 5/6 | 4/6 | 1119 / 1451 / 1158 | 46.21 / 22.07 / 0.00 |

B在五个seed均高于A；C相对A两组提高、两组持平、一组下降，不能仅因C低于B就否认相对无记忆A的局部提升。A/C3300只尝试一音，不用于完整六音动作效率比较。C3302–3304零请求属真实跳过API；非零耗时含协议/排队差异，不证明推理能力提升，token需核对。没有追加实验、筛选原生得分或修改任务。

[完整逐seed对照及限制](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/client_sse_BC_3300_3304_20261026T183404Z/results.md)。

### C触觉技能冷启动／复用配对（2026-10-27 UTC）

使用已有run_responses串行，两条有效执行固定同一份LLM上层代码及seed3300。冷启动空触觉记忆，复用读取冷启动公开保持得到的一条白键轻音技能。数据盘保留首份源码、5D接触特征、函数哈希、候选参数及真实力离线审计。原生标签不输入检索或模型。

| 版本 | seed | 原生成功/6 | 尝试 | 生成/清场动作 | 代码块 | END | 退出码 | 证据 |
|---|---:|---:|---:|---|---:|---|---:|---|
| 接线中断，不纳入控制比较 | 3300 | — | — | — | — | 否 | 137 | [日志](/mnt/data/ljz/opentacbench/capx-runs/C_tactile_skill_cold_seed3300_20261027T014444Z_smoke_rep1_20261027T014444Z/run.log)；pilot开关漏透传，停止本次进程后保留首份模型源码 |
| 空记忆冷启动 | 3300 | 6/6 | 6 | 1143/0 | 1 | 是 | 0 | [日志](/mnt/data/ljz/opentacbench/capx-runs/C_tactile_skill_cold_seed3300_20261027T014842Z_smoke_rep1_20261027T014844Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/C_tactile_skill_cold_seed3300_20261027T014842Z_smoke_rep1_20261027T014844Z_1c4f9306e0ae4e44861e0866b4c42e09/shadow_piano_native_audit_summary.json) |
| 只读触觉技能复用 | 3300 | 6/6 | 6 | 1143/0 | 1 | 是 | 0 | [日志](/mnt/data/ljz/opentacbench/capx-runs/C_tactile_skill_reuse_seed3300_20261027T014842Z_smoke_rep1_20261027T015511Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/C_tactile_skill_reuse_seed3300_20261027T014842Z_smoke_rep1_20261027T015511Z_f4ca93d599514e9ba28e22cc20fc1724/shadow_piano_native_audit_summary.json) |

四次命中并应用：F4、A4、A4、E4，来源均为冷启动F4轻音，Z修正−0.0890888 mm；四次均从实际action记录确认。两条6/6且1143步相同，未显示成功数或总动作收益。匹配后相同物理窗口的真实力MAE三处略降、一处略升，变化小于0.001 N，不能据单对样本称稳定控制提升。重音未获得合格记忆，继续原控制器。

有效两条API请求为0；首份源码生成请求发生在单列的接线诊断中，不能把生成成本当0。56项相关CPU检查及补充透传检查通过；四份视频均含音频。未追加试验、未按成绩换源码、未修改OpenTacBench。

[完整报告、技能快照、全部视频及私有力图](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/C_tactile_skill_pair_20261027T014842Z/results.md)。

### C触觉参数技能：四个新seed的冻结记忆复测（2026-10-27 北京时间）

保持运行代码、5D估计器、检索阈值及控制条件不变，已有run_responses在GPU0串行跑3301–3304，每seed空库/记忆各一次。两边均只读，冻结控制源码5b577958b50d3f238bac286002cf48f666b06ee6d2677cad2693f0f9c139b7a0，记忆仅为3300的F4白键轻音经验。没有重新生成代码、无整组重试或按得分筛选。

| 终端结束北京时间 | 条件 | seed | 原生成功/6 | 尝试 | 生成/清场动作 | 代码块 | 命中/实际采用 | END | 退出码 | 证据 |
|---|---|---:|---:|---:|---|---:|---|---|---:|---|
| 2026-10-27 12:41:27 +08:00 | empty | 3301 | 3/6 | 6 | 1164/0 | 1 | 0/0 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/C_tactile_transfer_empty_seed3301_20261027T043600Z_smoke_rep1_20261027T043600Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/C_tactile_transfer_empty_seed3301_20261027T043600Z_smoke_rep1_20261027T043600Z_1c79ea1f87824e64b903b82c6fc8556d/shadow_piano_native_audit_summary.json) |
| 2026-10-27 12:46:58 +08:00 | memory | 3301 | 3/6 | 6 | 1164/0 | 1 | 1/1 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/C_tactile_transfer_memory_seed3301_20261027T043600Z_smoke_rep1_20261027T044127Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/C_tactile_transfer_memory_seed3301_20261027T043600Z_smoke_rep1_20261027T044127Z_27297c58f9784b7c84bd80cde449561d/shadow_piano_native_audit_summary.json) |
| 2026-10-27 12:52:49 +08:00 | empty | 3302 | 4/6 | 6 | 1281/0 | 1 | 0/0 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/C_tactile_transfer_empty_seed3302_20261027T043600Z_smoke_rep1_20261027T044658Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/C_tactile_transfer_empty_seed3302_20261027T043600Z_smoke_rep1_20261027T044658Z_710433e2db4a46a6bb0bbd31ce7e50f6/shadow_piano_native_audit_summary.json) |
| 2026-10-27 12:58:50 +08:00 | memory | 3302 | 4/6 | 6 | 1281/0 | 1 | 1/1 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/C_tactile_transfer_memory_seed3302_20261027T043600Z_smoke_rep1_20261027T045249Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/C_tactile_transfer_memory_seed3302_20261027T043600Z_smoke_rep1_20261027T045249Z_b6a403ee2da140ec8d9c8b4dd02ebdce/shadow_piano_native_audit_summary.json) |
| 2026-10-27 13:04:37 +08:00 | empty | 3303 | 5/6 | 6 | 1275/0 | 1 | 0/0 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/C_tactile_transfer_empty_seed3303_20261027T043600Z_smoke_rep1_20261027T045850Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/C_tactile_transfer_empty_seed3303_20261027T043600Z_smoke_rep1_20261027T045850Z_5b99703d69494367809253b32b591266/shadow_piano_native_audit_summary.json) |
| 2026-10-27 13:10:38 +08:00 | memory | 3303 | 5/6 | 6 | 1275/0 | 1 | 0/0 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/C_tactile_transfer_memory_seed3303_20261027T043600Z_smoke_rep1_20261027T050437Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/C_tactile_transfer_memory_seed3303_20261027T043600Z_smoke_rep1_20261027T050437Z_e6d66d36a1f24304bafe32c01eec9de7/shadow_piano_native_audit_summary.json) |
| 2026-10-27 13:16:30 +08:00 | empty | 3304 | 4/6 | 6 | 1238/0 | 1 | 0/0 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/C_tactile_transfer_empty_seed3304_20261027T043600Z_smoke_rep1_20261027T051038Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/C_tactile_transfer_empty_seed3304_20261027T043600Z_smoke_rep1_20261027T051038Z_9f99490dbc04478e91e4eaac10f98f72/shadow_piano_native_audit_summary.json) |
| 2026-10-27 13:22:13 +08:00 | memory | 3304 | 4/6 | 6 | 1238/0 | 1 | 0/0 | 是 | 1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/C_tactile_transfer_memory_seed3304_20261027T043600Z_smoke_rep1_20261027T051630Z/run.log) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/C_tactile_transfer_memory_seed3304_20261027T043600Z_smoke_rep1_20261027T051630Z_e85101ad3d424acf8528482adc2384cc/shadow_piano_native_audit_summary.json) |

逐seed配对：3301为3/6、1164步；3302为4/6、1281步；3303为5/6、1275步；3304为4/6、1238步，两条件均相同。每次六音尝试、一个代码块并END，均无代码异常；退出码1来自未全对。

3301在273步查询、274步首次动作差异；3302在961步查询、962步首次动作差异；两者匹配前目标逐帧一致，分别实际采用一次。3303/3304没有白键轻音，均无命中且动作目标全程逐帧一致。输入库哈希前后不变，八条API请求均0，16份HEAD/第三方视频均含音轨。

离线实测力：匹配后同物理窗口MAE，3301/F4轻0.022578→0.021407 N（约−5.18%），3302/C4轻0.021651→0.020680 N（约−4.48%）。绝对变化约0.001 N，仅两个有匹配新seed、每seed一对，不作统计显著性或整体策略提升结论。全部原生失败均为重音，当前库没有重音技能。未修改OpenTacBench、估计器或运行代码；未追加实验。

[完整逐音报告、动作证据、16份视频及对照图](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/C_tactile_skill_transfer_3301_3304_20261027T043600Z/results.md)。


### 2026-10-27 入口修复后的单 seed 门控验证

本条为独立验证，不加入前述历史批次，也未携带试采记忆进入新对照。GPU3、joint 空记忆、seed3300，仍由原 `./run_responses` 启动；API 无代理、无重试。终端结束时间为北京时间 2026-10-27T20:07:37+08:00。

| seed | 音序及轻重 | 原生成功音数 | 原生尝试／公开提交 | 生成／清场动作 | 代码块／重新生成 | END | 退出码 |
|---|---|---:|---|---:|---|---|---:|
| 3300 | F4轻、A4轻、C#4重、A4轻、A4重、E4轻 | 4/6 | 6／5 | 3635／220 | 1／0 | 否 | 1 |

初始生成 8082 token、API 耗时 125.441 秒；整理 7314 token、API 耗时 8.202 秒，返回 `no_change`，公开执行检查未通过。API 耗时包含服务端排队和传输，不能称为纯模型推理时间。原生两次失败均为重音（C#4、A4），原因记录为 `incomplete_note`，本条不将其进一步归因于估计误差。

源码通过模块校验，生成代码正常返回。两个已核实的问题：生成策略把 `steps=4` 当成四个物理步，但底层每个动作步已推进四个物理步，实际每次调用16步、反馈约15Hz；代码却仅累计4/240秒。最后物理步15420返回公开 `native_done=True`，生成函数抛出并捕获 `NativeDone` 后直接返回，跳过第六次公开提交与计数；宿主因完成计数仍为5而未提交END。

结论：入口修复得到真实运行验证，但完整单seed门控未通过；未启动四组并行，不进行自动补跑。

[结构化结果](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/single_seed_entry_gate_3300_20261027T112731Z/gate_result.json) · [日志](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/single_seed_entry_gate_3300_20261027T112731Z/joint/runs/seed_3300/gate_joint_seed3300_20261027T112731Z_smoke_rep1_20261027T112735Z/run.log) · [代码](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/single_seed_entry_gate_3300_20261027T112731Z/joint/runs/seed_3300/gate_joint_seed3300_20261027T112731Z_smoke_rep1_20261027T112735Z/trial_01_sandboxrc_0_reward_0.000_taskcompleted_0/code.py) · [原生审计](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/single_seed_entry_gate_3300_20261027T112731Z/joint/private_audits/gate_joint_seed3300_20261027T112731Z_smoke_rep1_20261027T112735Z_3ec2e4895d75420381b7700d92d9a874/shadow_piano_native_audit_summary.json) · [HEAD视频](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/single_seed_entry_gate_3300_20261027T112731Z/joint/runs/seed_3300/gate_joint_seed3300_20261027T112731Z_smoke_rep1_20261027T112735Z/trial_01_sandboxrc_0_reward_0.000_taskcompleted_0/videos/video_combined.mp4) · [第三方视频](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/single_seed_entry_gate_3300_20261027T112731Z/joint/runs/seed_3300/gate_joint_seed3300_20261027T112731Z_smoke_rep1_20261027T112735Z/trial_01_sandboxrc_0_reward_0.000_taskcompleted_0/videos/video_combined_third.mp4)


### 2026-10-27 文字＋触觉证据＋代码更新：单 seed 3300（21:07 完成）

使用现有 `./run_responses`、GPU 3、joint 模式、全新独立记忆，模型 gpt-6.1-sol/low，代理关闭、无重试。运行 API 文档明确动作与物理步单位后的版本；未修改任务或力估计器。

| seed | 音序／力度 | 原生成功音 | 原生尝试／公开结束记录 | 生成动作／清场动作 | 代码块／重新生成 | END／运行退出码 | 更新结果 |
|---|---|---|---|---|---|---|---|
| 3300 | F4轻、A4轻、C#4重、A4轻、A4重、E4轻 | 6/6 | 6/5 | 1186/220 | 1/0 | 否／1 | API 成功，no_change；未生成修改候选 |

生成调用 88.13 秒、7462 token（输入4691／输出2771）；更新调用 5.03 秒、9714 token（输入9669／输出45）。这是接口请求耗时，不是纯模型推理耗时；独立 preflight 不计入这两项用量。后台 shell 最终退出0仅代表队列结束，实际 run 退出码为1。

本次代码使用 steps=1、DT=4/240，按压期间测力相隔4物理步。原生最后一音在物理步5621释放计分，native_done=True；生成的 release 随即跳出并误报 release_timeout，返回False，导致最后的 finish_piano_note_attempt 与 END 未执行。原生6/6与CaP-X整条未完成须分别报告。

公开记忆保存1187帧、24片段（接近／按下／保持／释放各6段）。原始公开日志保留停止原因，但证据 manifest 的 code_execution 只保存 has_exception；更新器又按异常优先、函数名排序选中 hold 和 press，未传 release 函数、释放片段或 release_timeout 原因。输入压缩JSON为24368字节，未混入私有判分。API返回空 functions/rules/rule_evidence，校验合法但无修改，文字库仍为空；没有待执行候选或有效活动版本。不能称为更新成功，也不能据此归因于模型性能。

[完整审计](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/single_joint_update_3300_20261027T125710Z/result_audit.json) · [运行日志](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/single_joint_update_3300_20261027T125710Z/joint/runs/seed_3300/joint_update_seed3300_20261027T125710Z_smoke_rep1_20261027T125720Z/run.log) · [生成代码](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/single_joint_update_3300_20261027T125710Z/joint/runs/seed_3300/joint_update_seed3300_20261027T125710Z_smoke_rep1_20261027T125720Z/trial_01_sandboxrc_0_reward_0.000_taskcompleted_0/code.py) · [更新输入](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/single_joint_update_3300_20261027T125710Z/joint/runs/seed_3300/joint_update_seed3300_20261027T125710Z_smoke_rep1_20261027T125720Z/tactile_evidence_update/public_input.json) · [更新结果](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/single_joint_update_3300_20261027T125710Z/joint/runs/seed_3300/joint_update_seed3300_20261027T125710Z_smoke_rep1_20261027T125720Z/piano_evidence_update_result.json) · [私有原生审计](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/single_joint_update_3300_20261027T125710Z/joint/private_audits/joint_update_seed3300_20261027T125710Z_smoke_rep1_20261027T125720Z_8e63a50359f84cb89ca682b953081ca5/shadow_piano_native_audit_summary.json) · [HEAD视频](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/single_joint_update_3300_20261027T125710Z/joint/runs/seed_3300/joint_update_seed3300_20261027T125710Z_smoke_rep1_20261027T125720Z/trial_01_sandboxrc_0_reward_0.000_taskcompleted_0/videos/video_combined.mp4) · [第三方视频](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/single_joint_update_3300_20261027T125710Z/joint/runs/seed_3300/joint_update_seed3300_20261027T125710Z_smoke_rep1_20261027T125720Z/trial_01_sandboxrc_0_reward_0.000_taskcompleted_0/videos/video_combined_third.mp4)。

本条进程已结束；未追加 seed、未启动并行组、未人工修改生成代码或替换更新结果。

后续收尾修复验证（离线回放，不计为新增实验）：保持本条原生成源码和全部公开观测，1186次动作目标与468次测力读取逐项匹配；终止仍在物理步5621，主机补记末次尝试后提交一次END，CaP-X返回Task Completed=True。原生结果未变，原代码completed=5和release_timeout日志保留，补记单独标为host_native_termination且release_verified=false；不据此晋升策略或改写上表。见[离线回放结果](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/single_joint_update_3300_20261027T125710Z/terminal_finalization_replay.json)。


### 2026-10-27 原生终止修复后的真实单 seed 验证（23:54 完成）

此前22:46的首次启动在预检收到HTTP400“Input must be a list”，仿真未启动，不记作0/6。修复t-cap/run预检为消息列表并复用现有Responses报文/SSE解析；用户明确要求修复后重启。原失败日志保留于[预检失败审计](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/single_joint_terminal_fix_3300_20261027T144629Z/result_audit.json)。

| seed | 音序／力度 | 原生成功音 | 原生尝试／公开结束 | 生成动作／清场 | 代码块／重新生成 | 生成token／更新token | END／退出码 | 更新 |
|---|---|---|---|---|---|---|---|---|
| 3300 | F4轻、A4轻、C#4重、A4轻、A4重、E4轻 | 4/6 | 6/6（策略5、主机1） | 1225/220 | 1/0 | 8080/0 | 是，1次／1 | 输入构建失败，未调用更新API |

GPU3，joint模式，全新独立记忆；gpt-6.1-sol/low，无代理、无自动重试。真实预检200/OK，生成请求98.60秒，输入4934／输出3146 token；策略源码381行。测力540次，连续按压读数间隔4物理步，公开质量ok215/no_contact325，无OOD。以上不证明估计力逐物理步准确。

C#4重、A4重原生判incomplete_note，其他四音成功；生成代码五条PIANO_NOTE都报告press_budget，公开保持信心与原生判分不同。此次不是上次6/6代码的原样重放，不能将成绩变化归因于主机收尾修复。最后一音在物理步5777终止，代码正确传播native_done并正常返回，无release_timeout；主机仅补记原生已消费的末次尝试并提交一次END，原生失败仍为失败，run退出1。后台队列退出0仅表示队列结束。

更新器返回rejected_or_unavailable：No tactile clip is associated with the executed helpers。按名称排序选了_release_pose（369次调用）和hold（540次调用），两者都无含触觉样本的关联片段；实际press和release各调用6次，分别有关联片段18和12个。选函数时先占满两个名额再检查触觉关联，导致发送前失败；不是API超时或模型拒绝修改。记忆仍保存1226帧和24片段，未产生候选。该筛选缺陷及此前释放证据遗漏尚待修复。

[完整结果](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/single_joint_preflight_fix_3300_20261027T154339Z/result_audit.json) · [run.log](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/single_joint_preflight_fix_3300_20261027T154339Z/joint/runs/seed_3300/joint_preflight_fix_seed3300_20261027T154339Z_smoke_rep1_20261027T154350Z/run.log) · [更新错误](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/single_joint_preflight_fix_3300_20261027T154339Z/joint/runs/seed_3300/joint_preflight_fix_seed3300_20261027T154339Z_smoke_rep1_20261027T154350Z/piano_evidence_update_result.json) · [私有原生审计](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/single_joint_preflight_fix_3300_20261027T154339Z/joint/private_audits/joint_preflight_fix_seed3300_20261027T154339Z_smoke_rep1_20261027T154350Z_3b820e5b108c47ca827f9b41f0d4e6ef/shadow_piano_native_audit_summary.json) · [HEAD视频](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/single_joint_preflight_fix_3300_20261027T154339Z/joint/runs/seed_3300/joint_preflight_fix_seed3300_20261027T154339Z_smoke_rep1_20261027T154350Z/trial_01_sandboxrc_0_reward_0.000_taskcompleted_0/videos/video_combined.mp4) · [第三方视频](/mnt/data/ljz/opentacbench/shadow_piano_memory_comparison/single_joint_preflight_fix_3300_20261027T154339Z/joint/runs/seed_3300/joint_preflight_fix_seed3300_20261027T154339Z_smoke_rep1_20261027T154350Z/trial_01_sandboxrc_0_reward_0.000_taskcompleted_0/videos/video_combined_third.mp4)。实验已结束，无剩余本次进程，未追加seed或批量实验。
