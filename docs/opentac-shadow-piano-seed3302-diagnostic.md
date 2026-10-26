# Seed 3302：单次只读代码复用诊断

本次仅运行一次3302，GPU 0。使用暂停队列当前的最新策略4a4f11755b185c57a6ba2b8be4ae67a743be7c94b372598d248f446e7893c74e的独立副本，不从私有成绩挑选旧成功版本。键位缓存同样复制公开标定。没有重新生成、自动修复、任务源码修改或原缓存写入。

## 结果

原生 **6/6**，尝试6次，END=True，退出码0，生成动作1267步，清场0步。API请求0，token=0。

| 序号 | 音符 | 力度 | 原生按下物理步 | 原生释放物理步 | 原生通过 |
|---|---|---|---:|---:|---|
| 1 | F4 | 重 | 525 | 685 | 是 |
| 2 | D#4 | 重 | 1493 | 1579 | 是 |
| 3 | A#4 | 重 | 2281 | 2480 | 是 |
| 4 | D#4 | 轻 | 3133 | 3272 | 是 |
| 5 | C4 | 轻 | 3945 | 4092 | 是 |
| 6 | F#4 | 重 | 4956 | 5067 | 是 |

## 为什么不能说3302一直失败

| 运行 | 原生结果 | 主要事实 |
|---|---|---|
| 历史技能起点十组 | 6/6 | 曾完整通过 |
| 空技能库起点十组 | 0/6 | 五音已按下，但释放只执行0.05秒就提交；未满足0.10秒释放条件 |
| 可执行代码记忆十组 | 6/6 | 实际通过，但公开不确定/OOD被用作整段修复触发，额外消耗71106 token |
| 本次当前缓存单独复测 | 6/6 | 源码直接复用，0次API、0 token，六音均有原生press/release事件 |

此前0/6属于特定生成程序的过早释放错误，当前公开释放护栏与策略已避免该问题。本次成功验证现有按键策略，不证明所有seed都会成功，也不把分类修复当成物理成功率提升。

## 本次修复

新增只读代码记忆诊断开关CAPX_SHADOW_CODE_MEMORY_READ_ONLY=1：缓存或标定不兼容时直接拒绝，不回退到模型；执行和结束后均禁止自动修复。本次仍保留真实错误、原生判分和双视角录像。

将安全结束的PIANO_NOTE中hold_uncertain、保持信心False、OOD等内容保留为uncertain_observations，不再单独据此自动改写整段代码。代码异常、释放超时、恢复检查点及未完成尝试链仍归为可处理公开失败。判断过程不读取原生得分。本次六音全部通过，但仍有四条公开不确定摘要，说明必须区分“公开信心”和“原生结果”。

12项相关CPU回归测试通过：只读模式在缓存缺失、执行修复和结束后修复分支均不请求API；安全不确定不触发改码；真实代码错误仍被记录。原十组队列仍保持暂停，当前策略副本与原缓存校验值一致。

## 产物

- [本次结果JSON](/mnt/data/ljz/opentacbench/shadow_piano_diagnostics/seed3302_current_cache_20261025T105313Z/result.json)
- [运行日志](/mnt/data/ljz/opentacbench/capx-runs/shadow_3302_readonly_diag_20261025T105313Z_smoke_rep1_20261025T105313Z/run.log)
- [HEAD视频](/mnt/data/ljz/opentacbench/capx-runs/shadow_3302_readonly_diag_20261025T105313Z_smoke_rep1_20261025T105313Z/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/videos/video_combined.mp4)
- [第三方视频](/mnt/data/ljz/opentacbench/capx-runs/shadow_3302_readonly_diag_20261025T105313Z_smoke_rep1_20261025T105313Z/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/videos/video_combined_third.mp4)
- [原生私有审计](/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1/shadow_3302_readonly_diag_20261025T105313Z_smoke_rep1_20261025T105313Z_047a314839b747b3a67f9042a8e18974/shadow_piano_native_audit_summary.json)

两份视频均为106帧、5fps、21.2秒，含AAC音轨。私有力审计保留独立文件，不写入运行时记忆。通用LLM技能记忆的研究想法已保存于 [设计记录](/mnt/a/ljz/cap/t-cap/cap-x/docs/tactile-executable-skill-memory-design.md)。
