# Shadow Hand 钢琴失败案例库

此文件用于**离线诊断和下一轮策略设计**。每次 smoke 后追加一条运行记录：保留运行参数、公开停止信号、原生审计结论、证据路径与原因的确定程度。它没有接入 `capx.skills.SkillLibrary` 或运行时提示词；私有琴键力和原生判分只在离线审计中查看，绝不可传给生成代码。可复用且只依赖公开观测的经验，人工整理到 `env_configs/opentac/shadow_piano_public_skills.v1.yaml`；Shadow visual-force 配置每次启动时加载该文件，并将其加入初始提示词。后续恢复提示词保留初始文本，所以也会携带同一份运行时案例库。

## 运行时案例库维护

每次新 smoke 后，先在本文件记录完整证据，再把可泛化的公开经验写成运行时 YAML 的 `id`、`trigger`、`practice`。不要写私有逐步真实力、原生判分、审计路径或固定的逐音专家坐标。配置通过 `shadow_piano_public_skill_library` 指向该 YAML；下一次运行会重新读取、检查格式和长度，并在日志打印文件路径及 SHA256。运行时提示词包含同一校验值，便于核对本次实际使用的版本。对 OOD 或失败的解释要区分已观测事实与未证实的因果猜测。

## 记录约定

每条记录应包含：`run.log` 路径、运行时间、t-cap 与 cap-x commit、配置与其运行时快照、模型、GPU、seed、音序与力度、DIGIT 配置校验值和估计器版本；失败音符的视觉映射 X/Y、实际到位状态、探针 Z、弯曲目标、动作频率、力反馈增益及限幅、公开估计力与质量、退出码、是否提交 END；另列私有审计路径和原生计分。没有证据的字段写“未记录”，原因用“已证实”或“待验证”标识。**不能用当前 YAML 推断旧运行的实际提示词**；旧运行以各自 `initial_prompt.txt` 和生成的 `code.py` 为准。

以下 18 条是截至 2026-10-22 在 `/mnt/data/ljz/opentacbench/capx-runs/shadow_*/run.log` 中找到的运行。除特别注明，审计中的任务参数为 seed 3150、8 音 `E4 F4 C#4 D#4 D#4 E4 C#4 F4`，力度 `light heavy light heavy light heavy heavy light`，每音持音 0.12 s，严格力度判分开启；动作步为 8 个 240 Hz 物理步。日志中的 GPU 为 6、模型为 `gpt-6-sol`；首条 clear-view v2 使用 GPU 2。历史配置内容随时间变化，不能当成同一次固定参数实验。

| 运行 UTC（`run.log`） | 退出码；原生完成音数 / 8；END | 失败位置、可核对参数与原因 |
| --- | --- | --- |
| [10-20 09:49](/mnt/data/ljz/opentacbench/capx-runs/shadow_visual_force_seed3150_smoke_rep1_20261020T094936Z/run.log) | 0；0；否 | 日志在 `in-progress save code.py` 附近中断；无试验完成记录。原因未确定，退出码 0 不代表任务成功。 |
| [10-20 12:21](/mnt/data/ljz/opentacbench/capx-runs/shadow_visual_force_seed3150_smoke_rep1_20261020T122124Z/run.log) | 0；未记录；未记录 | 只见 Kit 启动与 Lula 提示，没有 trial 总结或可读私有审计；**证据不足**，不能判为成功或定位按键失败。 |
| [10-20 13:41](/mnt/data/ljz/opentacbench/capx-runs/shadow_visual_force_seed3150_smoke_rep1_20261020T134124Z/run.log) | 1；0；否 | `code_path.write_text` 在 ASCII 编码下遇到 U+2019，`UnicodeEncodeError`；**已证实是产物保存错误**，未进行有效按键。 |
| [10-20 14:21](/mnt/data/ljz/opentacbench/capx-runs/shadow_visual_force_seed3150_utf8_smoke_rep1_20261020T142103Z/run.log) | 0；0；是 | 保存问题之后，生成代码含弯引号 U+2019，Python `SyntaxError`；END 也不等于任务成功。 |
| [10-20 16:45](/mnt/data/ljz/opentacbench/capx-runs/shadow_visual_force_mirrored_seed3150_smoke_rep1_20261020T164504Z/run.log) | 0；0；否 | 生成模型认为无法可靠确定八度端点而停止；首块输出的自然语言又被当成代码执行，`SyntaxError`。没有键位映射或按压。 |
| [10-20 18:03](/mnt/data/ljz/opentacbench/capx-runs/shadow_visual_force_mirrored_seed3150_smoke_rep1_20261020T180300Z/run.log) | 1；0；否 | 两块代码只调用 `locate_visual_keys` 并打印映射与手状态，没有执行音符链。视觉映射目标如 E4 `y=-0.06298 m`、F4 `y=-0.03966 m`；映射成功不等于任务完成。 |
| [10-21 06:42](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v2_smoke_rep1_20261021T064209Z/run.log) | 1；0；否 | 第 1 音未进入力带；最终公开估计 `0.1271 N`，sample 306，`quality=ok`。生成代码报 bounded press failed；**已证实未建立可用接触**。GPU 2。 |
| [10-21 08:37](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v2_smoke_rep1_20261021T083750Z/run.log) | 1；0；否 | 原生 `wrong_key`；生成代码称 E4 到位时手腕 Y 曾为 `+0.14638 m`，随后原生终止。具体是哪段动作造成错键需按私有逐步轨迹复核，不能只凭模型打印的“到位”。 |
| [10-21 15:20](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261021T152011Z/run.log) | 1；1；否 | 原生 `wrong_key`；已计 E4，后续生成动作进入 C#4 时原生仍等待前序音，随后尝试在终止后继续动作并触发 `RuntimeError`。 |
| [10-21 19:13](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261021T191328Z/run.log) | 1；4；否 | D#4 阶段 sample 493 返回 `quality=ood`，按安全规则停止。私有同一步目标键力为 `2.086 N`，已经超出此 DIGIT 图像配置的 2 N 响应量程；**已证实超量程与 OOD 同步**，不能把 OOD 当有效牛顿估计。 |
| [10-21 19:28](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261021T192851Z/run.log) | 1；1；否 | 原生 `wrong_key` 后生成代码仍调用动作，出现两次“episode has ended” `RuntimeError`；必须在每个动作后检查 `native_done`。 |
| [10-21 19:38](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261021T193834Z/run.log) | 1；0；否 | Responses API 返回 HTTP 403，`queries=0`、`code blocks=0`；**接口拒绝**，不是视觉或触觉控制失败。 |
| [10-22 00:09](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261022T000938Z/run.log) | 1；0；否 | 同样是 Responses API HTTP 403，未生成动作代码。 |
| [10-22 00:12](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261022T001245Z/run.log) | 1；2；否 | E4、F4 原生均计分；公开中位数连续计数不足，打印 `public_hold_uncertain_after_in_band_release` 后停在下一音索引 2。**已证实公开信心门槛可造成已成功音后的提前停机**。 |
| [10-22 00:18](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261022T001809Z/run.log) | 1；1；否 | F4 重音阶段最终公开估计 `0.4235 N`，低于 `0.43 N` 下限；代码报告无连续入带窗口，未继续。 |
| [10-22 03:22](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261022T032237Z/run.log) | 1；未记录；未记录 | Kit/Isaac 后端抛出 `Failed to get DOF velocities from backend`；缺完整原生审计，**已证实后端调用错误**，不可推断按键策略。 |
| [10-22 14:44](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261022T144424Z/run.log) | 1；1；否 | E4 light 原生计分；F4 heavy 键角已到位，真实力却在动作采样间跌到 `0.41856 N`，合格最长 3 个物理步，未计 F4；生成代码随后弹 C#4，触发 `wrong_key`。详见[逐物理步审计](/mnt/a/ljz/cap/audits/shadow_piano_hold_20261023/README.md)。 |
| [10-22 18:52](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261022T185237Z/run.log) | 1；7；否 | 第 7 音 C#4 heavy 出现 sample 766 的 `quality=ood`；安全停止后仍完成该音释放，第 8 音 F4 light 未尝试。具体参数和证据见下。 |

## 案例：2026-10-22 18:52，七音计分后停止

- **运行参数**：GPU 6；模型 `gpt-6-sol`、Responses API；seed 3150、单 trial、无重试；配置路径 `env_configs/opentac/play_piano_shadow_digit_visual_force_smoke.yaml`。准确提示词在该运行的 `trial_01_sandboxrc_0_reward_0.000_taskcompleted_0/initial_prompt.txt`，不可用后来改动的 YAML 代替。DIGIT 配置 SHA256 `2c4e762965a06d1b38fbd6a54d638087ba2fa057dace69bdb77c741cb82f4208`，渲染力范围 `[0,2] N`；估计器 `shadow_digit_key_top_force_4d.v1`。
- **生成控制参数**：第 7 音 C#4 heavy，公开目标 `1.00 ± 0.14 N`，离线中位数计数门槛 `1.040 N`；映射手腕 `X=0.07000 m`、`Y=-0.081045 m`，探针高度 `z_probe=0.013410 m`，最终食指目标 `0.50 rad`。每动作步 8 个物理步。反馈增益为 `0.012 m/(N·s)`、速率限幅 `±0.003 m/s`，由提示词限定单步 `≤0.00010 m`、相对探针累计 `±0.0010 m`。
- **公开执行证据**：第二代码块从 `piano_state['completed']=6` 恢复；sample 757–765 连续打印 `PIANO_FORCE_TRIM`，公开估计约 `0.719→0.924 N`，命令 Z 从 `0.013410` 降到 `0.012890 m`（约 `0.52 mm`）；sample 766 返回 `quality=ood`、`estimated_key_force_N=None`。生成代码输出 `PIANO_STOP C#4 invalid_force_estimate`，释放并停止，未提交 END。[生成代码](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261022T185237Z/trial_01_sandboxrc_0_reward_0.000_taskcompleted_0/code.py)。
- **私有审计，仅供离线核对**：sample 766 对应物理步 6128，目标键真实力瞬时约 `1.3704 N`，超过该音原生上限 `1.14 N`，但仍低于 TACTO 的 2 N 图像量程；同一步公开质量因 4D 深度特征超出**训练特征范围**而标 `ood`，不是估计器输出了 1.37 N。[逐步估计与真值审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v3_smoke_rep1_20261022T185237Z_0abe5c32992a4410a7cdf699d95f2179/shadow_force_estimates_private_audit.jsonl)。这次峰值只有短暂一步，原生未置失败；释放后第 7 音确实计分。原生最后为 **7/8、`native_failure=""`、`end_submitted=false`**，[原生摘要](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_clear_view_seed3150_v3_smoke_rep1_20261022T185237Z_0abe5c32992a4410a7cdf699d95f2179/shadow_piano_native_audit_summary.json)，故整体 smoke 失败、退出码 1。[视频](/mnt/data/ljz/opentacbench/capx-runs/shadow_clear_view_seed3150_v3_smoke_rep1_20261022T185237Z/trial_01_sandboxrc_0_reward_0.000_taskcompleted_0/videos/video_combined.mp4)。
- **前一块的停机因素**：第 6 音 E4 heavy 虽已获原生计分，公开末次估计仅 `0.4414 N`，低于离线中位数门槛 `0.473 N`，使 `hold_confidence=False`，因此提前触发再生成。第二块按公开进度从第 7 音恢复，这次没有错弹已完成的 E4。
- **结论与下一轮检验**：已证实“公开估计 OOD → 按安全规则停止 → 未尝试第 8 音”；1.3704 N 峰值与下调 Z 的阶段同步，但**不能仅凭一条轨迹证明微调是峰值的唯一成因**。候选检验是等待食指接近 0.50 rad 后的接触响应稳定，再启动 Z 反馈，并比较调整前后逐物理步峰值和连贯保持。不要把 OOD 伪装成可用估计，也不要把原生计分写入运行时记忆或提示词。

## 可复用诊断规则

1. `quality=ood` 是特征越界，不是一个可控制的牛顿值；先安全释放，离线核对同一步真实力、深度特征和动作，区分超量程与姿态分布偏移。
2. “已映射键位”“已到位”“公开力入带”“原生已计分”是四个不同事件。用同一运行的生成代码、公开日志和私有原生审计逐步对齐，避免在未计分时切到下一键。
3. 每个动作步只采样一次 DIGIT，原生每个物理步评分；公开连续五个样本不能证明中间 240 Hz 力也连续达标。
4. 记录控制增益、限幅、Z 与弯曲目标的时间序列；若力仍在上升时同步下调 Z，只能提出“可能放大超力”的假设，需用对照试验验证。
5. HTTP 403、Kit 后端错误、代码保存或语法错误单独归类，不计作触觉策略的失败样本。退出码 0 也必须核对 trial 总结、原生计分和 END。
