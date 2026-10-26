# 随机六音与逐次公开技能整理

2026-10-26：额外的批量编排和套件审计脚本已按用户要求删除。本文保留历史实验方法及结果，当前运行入口统一为 t-cap 的 `./run_responses`；多个 seed 直接用 shell 循环串行调用。

本实验使用同一份 `env_configs/opentac/play_piano_shadow_digit_random6_online_skills.yaml`，让原生任务按实际 seed 随机生成 C4–B4 的六音及 light/heavy，持音 0.12 秒。固定 240 Hz 物理、四物理步一次动作、`shadow_digit_key_top_force_5d_matched60hz_8episode.v1` 估计器。单音失败后继续，最终原生逐音成功数供离线报告使用；原生成功、真实力和私有审计不进入生成代码或技能整理。

定位方式为**公开端点与几何辅助定位**。冷启动时 LLM 观察 HEAD 图片，确认公开 A0/C8 端点后向 `locate_visual_keys` 提交两个归一化白键像素。适配器由公开端点、0.0235 m 白键间距及黑键半格关系计算当前音序的目标，不查询私有逐键位置。缓存保存相机、镜像、端点和控制坐标系指纹；兼容时下一 seed 的初次请求就使用文本和公开键位，不上传图片，并省去仅为拍定位图执行的清场动作。指纹不匹配时恢复图片输入。

每个 seed 开始前固定技能库，结束后复用现有 Responses 客户端整理一次；最多 4096 输出 token，120 秒，无网络重试。整理器仅打开显式白名单的公开请求、动作、手部状态、DIGIT 估计、生成代码与公开异常记录。每条变更要求引用本次公开 event_id；策略推断标记为待验证。最多 16 条、8 KB，更新失败保留旧版。原始静态库和旧 smoke YAML 保留。

在 t-cap 目录直接运行 `./run_responses`，调用者显式设置配置、任务 seed 和 GPU，清除全部大小写代理变量。旧套件脚本命令及 `--limit`、`--resume` 参数已停用。历史实验期间未修改固定配置或估计器；技能库与端点缓存是唯二的跨 seed 运行时变化。

数据盘目录保存 `manifest.json`、冻结配置、`results.md/json`、每组终端和技能整理日志、`skills/`、`reflections/seed*/`。每个整理目录保留输入库、公开证据、完整请求、响应、提案、来源和输出库。原生私有审计路径仅用于结果表。

结果表逐 seed 列出成功音数/6、尝试数、生成动作/清场动作、生成 token、整理 token、重新生成次数、图片输入次数、技能库输入/输出校验值及 END/异常，不汇总成功音总数。动作步为四个物理步；公开保持判断不能证明四个中间物理步均达标。不同 seed 的难度、键位缓存和技能更新同时变化，十组顺序实验不足以单独证明技能更新造成性能改善。

实现验证：运行前 39 项相关 CPU 测试通过；实验后加入整体超时与拒绝提案留档检查，共 40 项通过。覆盖端点计算新音、镜像变更失效、显式 seed、reset 后 prompt、不上传缓存图片、私有字段白名单、更新拒绝及超时后保留原库。

## 本次十组结果

已在 GPU 0 串行完成 3300–3309，未替换 seed、未补跑实验，全部尝试六音并提交 END；完成后停止 GPU 实验。下表 token 为提供方报告的输入加输出 token，生成与整理分开计；动作步各推进四个物理步。退出码 1 是原生未全对的明确失败退出，不等同于 Python 崩溃。

数据盘套件：[`random6_3300_3309_20261025_v1`](/mnt/data/ljz/opentacbench/shadow_piano_online_skills/random6_3300_3309_20261025_v1)。逐次完整数值与全长库校验值在 [results.json](/mnt/data/ljz/opentacbench/shadow_piano_online_skills/random6_3300_3309_20261025_v1/results.json)；版本、源代码快照与冻结配置分别为 `manifest.json`、`source_snapshot/` 和 `config_frozen.yaml`。运行起点 cap-x 为 `a7ab66288ea1427299242f057dc1e976a23f84d6`，t-cap 为 `e308c7d47bb595d0bb397f42988a1a08f9dd35ff`，新增实现当时尚未提交，实际文件校验值以快照为准。OpenTacBench 为 `179bb8b5fe987f30ac17c407c5265b40c8f223a7`，受版本控制文件没有改动。

| seed | 音序及轻重 | 原生成功/6 | 尝试 | 生成/清场动作 | 代码块 | 生成请求 | 输入 token | 输出 token | 生成请求总 token | 整理 token | 重新生成 | 图片输入 | 技能库输入→输出 | END/异常 | 产物 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3300 | F4(轻) A4(轻) C#4(重) A4(轻) A4(重) E4(轻) | 6/6 | 6 | 1140/220 | 1 | 1 | 17602 | 3848 | 21450 | 25469 | 0 | 1 | 2b735eff→adb3a763 | 是 / rc=0 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3300_random6_3300_3309_20261025_v1_smoke_rep1_20261024T190946Z/run.log) · [视频](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3300_random6_3300_3309_20261025_v1_smoke_rep1_20261024T190946Z/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/videos/video_combined.mp4) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3300_random6_3300_3309_20261025_v1_smoke_rep1_20261024T190946Z_658a08a508be470b95c15062f0f01052/shadow_piano_native_audit_summary.json) |
| 3301 | D4(重) F4(轻) C#4(重) F#4(重) G#4(轻) F4(重) | 4/6 | 6 | 1160/0 | 1 | 1 | 17147 | 3234 | 20381 | 27303 | 0 | 0 | adb3a763→3a57ec19 | 是 / rc=1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3301_random6_3300_3309_20261025_v1_smoke_rep1_20261024T192438Z/run.log) · [视频](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3301_random6_3300_3309_20261025_v1_smoke_rep1_20261024T192438Z/trial_01_sandboxrc_0_reward_0.000_taskcompleted_0/videos/video_combined.mp4) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3301_random6_3300_3309_20261025_v1_smoke_rep1_20261024T192438Z_b4e0be99064f4697a5cb0bd1d1d90e20/shadow_piano_native_audit_summary.json) |
| 3302 | F4(重) D#4(重) A#4(重) D#4(轻) C4(轻) F#4(重) | 6/6 | 6 | 958/0 | 1 | 1 | 6335 | 2426 | 8761 | 27771 | 0 | 0 | 3a57ec19→3a57ec19 | 是 / rc=0; 技能保留旧版 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3302_random6_3300_3309_20261025_v1_smoke_rep1_20261024T193033Z/run.log) · [视频](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3302_random6_3300_3309_20261025_v1_smoke_rep1_20261024T193033Z/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/videos/video_combined.mp4) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3302_random6_3300_3309_20261025_v1_smoke_rep1_20261024T193033Z_924a2b172ed448e1a60dcd6404881204/shadow_piano_native_audit_summary.json) |
| 3303 | F#4(重) G#4(轻) D#4(重) G4(重) A#4(轻) D4(重) | 4/6 | 6 | 1127/0 | 1 | 1 | 6499 | 2672 | 9171 | 30517 | 0 | 0 | 3a57ec19→d86b301b | 是 / rc=1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3303_random6_3300_3309_20261025_v1_smoke_rep1_20261024T193540Z/run.log) · [视频](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3303_random6_3300_3309_20261025_v1_smoke_rep1_20261024T193540Z/trial_01_sandboxrc_0_reward_0.000_taskcompleted_0/videos/video_combined.mp4) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3303_random6_3300_3309_20261025_v1_smoke_rep1_20261024T193540Z_e64c7a47f4004234a7989c3b49a0feec/shadow_piano_native_audit_summary.json) |
| 3304 | G4(重) A#4(轻) D#4(轻) A#4(重) F4(重) C4(重) | 4/6 | 6 | 1234/0 | 1 | 1 | 17104 | 3424 | 20528 | 12994 | 0 | 0 | d86b301b→d86b301b | 是 / rc=1; 技能保留旧版 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3304_random6_3300_3309_20261025_v1_smoke_rep1_20261024T194123Z/run.log) · [视频](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3304_random6_3300_3309_20261025_v1_smoke_rep1_20261024T194123Z/trial_01_sandboxrc_0_reward_0.000_taskcompleted_0/videos/video_combined.mp4) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3304_random6_3300_3309_20261025_v1_smoke_rep1_20261024T194123Z_0b89d303622448e19263d29883ca60ec/shadow_piano_native_audit_summary.json) |
| 3305 | D#4(重) G#4(轻) A4(重) B4(重) F4(轻) C#4(轻) | 4/6 | 6 | 1061/0 | 1 | 1 | 17373 | 3506 | 20879 | 29451 | 0 | 0 | d86b301b→d86b301b | 是 / rc=1; 技能保留旧版 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3305_random6_3300_3309_20261025_v1_smoke_rep1_20261024T194723Z/run.log) · [视频](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3305_random6_3300_3309_20261025_v1_smoke_rep1_20261024T194723Z/trial_01_sandboxrc_0_reward_0.000_taskcompleted_0/videos/video_combined.mp4) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3305_random6_3300_3309_20261025_v1_smoke_rep1_20261024T194723Z_a922757b11c54667bd6d4b8b650be7d4/shadow_piano_native_audit_summary.json) |
| 3306 | F4(轻) C4(重) E4(轻) G#4(轻) F#4(轻) D#4(重) | 6/6 | 6 | 838/0 | 3 | 3 | 43560 | 5139 | 48699 | 30269 | 2 | 0 | d86b301b→fc4b9ce2 | 是 / rc=0 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3306_random6_3300_3309_20261025_v1_smoke_rep1_20261024T195243Z/run.log) · [视频](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3306_random6_3300_3309_20261025_v1_smoke_rep1_20261024T195243Z/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/videos/video_combined.mp4) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3306_random6_3300_3309_20261025_v1_smoke_rep1_20261024T195243Z_22410ad3fc354d9d9b90964115911328/shadow_piano_native_audit_summary.json) |
| 3307 | B4(重) F4(重) C#4(重) D4(轻) F#4(轻) D4(轻) | 4/6 | 6 | 1335/0 | 1 | 1 | 17152 | 3264 | 20416 | 27069 | 0 | 0 | fc4b9ce2→b3909123 | 是 / rc=1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3307_random6_3300_3309_20261025_v1_smoke_rep1_20261024T195819Z/run.log) · [视频](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3307_random6_3300_3309_20261025_v1_smoke_rep1_20261024T195819Z/trial_01_sandboxrc_0_reward_0.000_taskcompleted_0/videos/video_combined.mp4) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3307_random6_3300_3309_20261025_v1_smoke_rep1_20261024T195819Z_bc520199faa844b0bfd91e9ff3cbd824/shadow_piano_native_audit_summary.json) |
| 3308 | C4(重) D4(轻) A4(重) A#4(轻) F4(轻) C4(重) | 5/6 | 6 | 1188/0 | 1 | 1 | 17198 | 3445 | 20643 | 28000 | 0 | 0 | b3909123→757a0bed | 是 / rc=1 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3308_random6_3300_3309_20261025_v1_smoke_rep1_20261024T200450Z/run.log) · [视频](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3308_random6_3300_3309_20261025_v1_smoke_rep1_20261024T200450Z/trial_01_sandboxrc_0_reward_0.000_taskcompleted_0/videos/video_combined.mp4) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3308_random6_3300_3309_20261025_v1_smoke_rep1_20261024T200450Z_14a3a0a83a794cc28da09096132616f3/shadow_piano_native_audit_summary.json) |
| 3309 | F#4(轻) A4(轻) A4(轻) F4(重) E4(轻) E4(轻) | 6/6 | 6 | 993/0 | 1 | 1 | 16941 | 3573 | 20514 | 25258 | 0 | 0 | 757a0bed→757a0bed | 是 / rc=0 | [日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3309_random6_3300_3309_20261025_v1_smoke_rep1_20261024T201048Z/run.log) · [视频](/mnt/data/ljz/opentacbench/capx-runs/shadow_random6_online_seed3309_random6_3300_3309_20261025_v1_smoke_rep1_20261024T201048Z/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/videos/video_combined.mp4) · [审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_random6_online_seed3309_random6_3300_3309_20261025_v1_smoke_rep1_20261024T201048Z_117fae6989084a1bb0c39518d03561d8/shadow_piano_native_audit_summary.json) |

## 结论与限制

- 四组达到 6/6：3300、3302、3306、3309；其余逐组值见表，不汇总成功音总数。3306 在 XY 速度尚未稳定时触发公开动作保护，两次文本重新生成后完成；一次用于动作保护恢复，另一次补齐最后尝试与 END，已有进度保留。没有基础设施失败、没有替换或重试整个 seed。
- 3300 使用一张 HEAD 图和 220 步视觉清场。3301–3309 初始及重新生成均为零图片、零清场动作，沿用同一标定 `88c9449830c4838b32ab65ce9bc1c6cc48116d95dd989c9fa0e0fe133ad2f5b4`。十组覆盖 C4–B4 的十二种音高，其中包括首组未遇到的新音。该结果支持当前固定布局下的公开端点几何缓存，不证明相机或键盘移动后仍可直接复用；失效条件由指纹检查和 CPU 测试覆盖。
- 每组都调用一次公开技能整理：六次有效变更、一次接受但无变更（3309）、三次拒绝（3302、3304、3305）。拒绝原因均为条目被改写却未附对应公开 event_id，上一有效库完整保留。每次输入、响应、校验结果和输出库都有单独记录；未把私有真实力、原生成功或失败音列表输入整理请求。
- 成功音数、生成动作数和生成 token 没有随更新单调改善。整理自身额外消耗 12994–30517 token，不能宣称总 token 已节省。键位缓存确实省去后续图片和清场；不同 seed 的难度和自动整理同时变化，需要固定种子、冻结库的对照实验才能分离技能库收益。本轮按计划止于十组，不追加挑选实验。
- 所有失败音均为 heavy，原生原因为 `incomplete_note`；没有 `wrong_key` 或 `wrong_note_force` 判罚。逐物理步的目标力连续达标最长只有 4–28 步，低于 0.12 秒所需的 29 步。因而当前瓶颈仍是重音力保持，不能仅凭四步一次的公开采样判定已满足原生持续保持。
- 私有审计中，查询时刻、age=0、quality=ok 且真实接触力至少 0.05 N 的样本，逐组估计 MAE 约 0.006–0.011 N。这个口径排除了 OOD/无接触，且不覆盖查询间的物理步，不能推广为全状态力估计精度或稳定控制证明。
- 这十组没有 E4 heavy，不能据此宣称此前 E4 重音问题已解决。十条视频均含 H.264 画面与 AAC 接触事件合成音轨。

## 私有物理步诊断

以下仅供离线报告，不输入运行时策略或技能库。trace 的力字段记录该步最大接触力对应琴键；本次失败音区间没有其他琴键成为主接触，因此可以重建目标力是否连续落在容差内。角度门槛仍由原生规则判定；这里的力连续步数已不足以满足成功条件。

| seed | 第几个音 | 音符/力度 | 最长连续力达标物理步 | 时间/秒 | 原生要求 |
|---|---|---|---:|---:|---|
| 3301 | 1 | D4/重 | 5 | 0.020833 | ≥29 步、0.12 秒 |
| 3301 | 6 | F4/重 | 8 | 0.033333 | ≥29 步、0.12 秒 |
| 3303 | 1 | F#4/重 | 21 | 0.087500 | ≥29 步、0.12 秒 |
| 3303 | 6 | D4/重 | 11 | 0.045833 | ≥29 步、0.12 秒 |
| 3304 | 4 | A#4/重 | 28 | 0.116667 | ≥29 步、0.12 秒 |
| 3304 | 5 | F4/重 | 14 | 0.058333 | ≥29 步、0.12 秒 |
| 3305 | 3 | A4/重 | 18 | 0.075000 | ≥29 步、0.12 秒 |
| 3305 | 4 | B4/重 | 25 | 0.104167 | ≥29 步、0.12 秒 |
| 3307 | 1 | B4/重 | 4 | 0.016667 | ≥29 步、0.12 秒 |
| 3307 | 2 | F4/重 | 23 | 0.095833 | ≥29 步、0.12 秒 |
| 3308 | 1 | C4/重 | 13 | 0.054167 | ≥29 步、0.12 秒 |

私有报告：[failure_physics_audit.private.v1.json](/mnt/data/ljz/opentacbench/shadow_piano_online_skills/random6_3300_3309_20261025_v1/failure_physics_audit.private.v1.json)。历史套件审计脚本已删除；已有审计结果和实验源码快照保留。

实验后仅补强了整理调用的整体 120 秒 deadline 和拒绝提案的 JSON 留档；本次十组实际调用均未超时，拒绝提案原始文本均已留档。未据审计调节控制 prompt、原生物理、评分或估计器。


## 代码块与 token 统计复核

2026-10-25 复核最终 `run.log`、`all_responses.json` 和完整的累计 `llm_usage.json`：除 3306 为三个代码块、三次代码请求、两次重新生成外，其余九组均为一个代码块、一次代码请求。此前按目录名称排序误选了 3306 的中途异常快照，曾记作 22536 token、一次重新生成；现已更正为 48699 token、两次重新生成。原汇总已在数据盘保留 `.before_codeblock_audit.v1` 备份。原生成功音数和动作数不受此更正影响。

表中的“生成请求总 token”是提供方报告的输入加输出，包含生成请求的输入 prompt，不是代码输出 token。无图初始请求仍有约 2.6 万字符：固定 YAML 控制要求约 1.54 万字符，另有 API 说明、约 4 KB 技能库及当前请求、键位。以 3301 为例，无图片、一个代码块，报告输入 17147、输出 3234，总计 20381 token。3306 两次恢复请求又重发了历史代码和公开执行反馈，文字分别达到 44160、45728 字符。

同样约 2.6 万字符的无图请求，服务方有时报告约 6300 输入 token，有时约 17000；当前计数按提供方原样保存，不把差异全部归因于图像或技能库，也不据此推断精确计费。输出 token 包含报告的 reasoning 部分，不再重复相加。详细数值见数据盘 `codeblock_token_audit.v1.json`。

历史技能整理在每组结束后调用 `reflect_once()`，再由 Responses API 提出完整下一版库；本地程序验证来源及格式后原子保存，失败则保留旧库并进入下一 seed。这十组之间没有人工编辑技能内容。该额外编排现已删除；直接 `run_responses` 不包含这层文字库跨 seed 整理。此次历史统计修复通过七项相关 CPU 测试，未重跑仿真或调用 API。

## 从空技能库开始的独立运行

上一批采用历史静态库的八条技能作为起点，因此不属于零记忆起点。新增 `--initial-skills empty`：仅创建 `schema: shadow_piano_public_skills.v1` 和 `skills: []`，不读取历史技能文件；新实验目录也不带入已有键位缓存。保持同一随机任务配置、固定控制 prompt 和已冻结的力模型。这里的“从零”指跨 episode 的外部技能条目与键位缓存，固定控制规则及离线传感器标定仍然保留。

仍使用 3300–3309，每组后自动调用 API 整理，程序校验并供下一组使用。结果表额外记录每组输入、输出的技能条目数，配合哈希、提案及来源判断是否发生更新。空库加载、首条有来源的技能创建等八项相关 CPU 测试通过。使用独立后台进程运行，无须 Codex 逐次监督；完成后结果留在指定目录，等待用户要求后再分析。

本轮空技能库起点的十组已完成，见 [结果与诊断](/mnt/a/ljz/cap/t-cap/cap-x/docs/opentac-shadow-piano-empty-memory-results.md)。
