# Shadow DIGIT：60 Hz 重新标定采集

更新：2026-10-24。状态：**seed 3200–3203 八条试采已完成并审计**，已做探索性整 seed
交叉验证。随后按用户要求，仅用这八条冻结了独立候选 sidecar，完成两条在线复验后停止新增实验；旧模型保留。

## 在线结论：两条固定 seed 3150 smoke

候选权重在两条之间不变，控制 YAML、运行时公共库与初始 prompt 不变。
`gpt-6-sol` Responses，low reasoning，max_tokens=8192，单 trial、无 trial 重试、单进程串行，
API 代理关闭。任务为八音、严格逐音力度／0.12 s 保持／释放判分，失败后继续尝试。
OpenTacBench 为 `179bb8b`，没有修改其源码；t-cap `e7ee48f`、cap-x `9d63265` 加当前本地改动。
生成的具体代码不同，因此这是端到端复验，不能当成只改变模型权重的严格因果对照。

| 运行 UTC / GPU | 原生成功 | END | 代码块 / 重新生成 | 代码动作 / 总动作 | 接触查询 MAE / P95 | OOD | 退出码 |
|---|---:|:---:|---:|---:|---:|---:|---:|
| 10-24 16:22:18 / 7 | 7/8 | 是 | 1 / 0 | 1615 / 1835 | 0.00585 / 0.01572 N | 0 | 1 |
| 10-24 17:05:12 / 1 | 6/8 | 是 | 3 / 2 | 1381 / 1601 | 0.00822 / 0.01469 N | 0 | 1 |

两次均完成八次尝试。第一条有 427 次 ok 接触查询、186 次 no_contact；第二条为 406／194。
没有实测接触 ≥0.05 N 却不可用的查询，也没有 ≥0.1 N 的无接触误报。
上述误差只覆盖被代码查询的接触帧，不是全部 240 Hz 物理步；两次都未达到整段八音成功，
因此 wrapper 退出 1 是正确报告。累计 13/16 仅为同一固定音序两轮的逐音计分，不是跨 seed 成功率。

### E4 重音：仍失败，但主要证据是步内保持失配

- 首条唯一失败音为第六音 E4 heavy，`incomplete_note`，峰值 0.48725 N；
  其可用估计 MAE=0.00854 N、P95=0.01747 N，无 OOD。
- 连续真实力在 `[0.43,0.57] N` 内最长仅 **7 tick=0.02917 s**；第二条 E4 heavy 为
  **9 tick=0.03750 s**。均远低于 0.12 s 所需的至少 29 tick；此处是仅力带统计，原生还检查键程等条件。
- 首条末段示例：一个动作的四个真实物理步约为
  `[0.42369,0.48281,0.42415,0.48329] N`，公开末帧估计约 0.49140 N。
  估计接近同一步真值，却看不到中间低于 0.43 N 的低相位，公开九样本保持因此可以误判。
- 此轮 F4 heavy 的 Z 从探针净下降约 0.514 mm，连续力合格 96 tick 并成功；
  E4 heavy 只净下降约 0.01384 mm，公开保持已为 true 就释放。不能把不同相位的末帧计数当作
  全部物理步持续合格，也不能仅因某个公开估计接近 0.5 N 就认定控制已稳定。
- E4 heavy **并非从未成功**：已有原生计分证据在
  `shadow_clear_view_seed3150_v3_smoke_rep1_20261022T185237Z`（press step 5361）与
  `shadow_clear_view_seed3150_v3_smoke_rep1_20261023T030622Z`（5521）。历史配置不同，不混算成功率。

### 第二条额外发现

第二条失败为 F4 heavy 和 E4 heavy，最长力带分别 16／9 tick。
首段生成代码漏掉独立的 0.50 s 降腕阶段，直接同时降低目标 Z 并弯指；
其 `native_done()` 还从不提供该字段的 `get_hand_state()` 读取终止标记，忽略了动作返回值。
最后释放时原生已终止，代码继续动作，触发两次 RuntimeError；第三段完成末次尝试记账并提交 END。
这属于生成策略与终止处理问题，不是估计器包未加载。
异常初触仍有大误差：第二条最大误差 0.24152 N，E4 action 1166 实测 0.37210 N、估计 0.13059 N。
所以平均精度改善不代表耦合冲击接触已可靠，矩形 OOD 边界也未能识别此异常。

### 本轮停止与建议

现有八条足以训练出可用于继续控制调试的候选；无需立即扩大采集量。
目前优先检查保持阶段的步内波动、反馈启动／释放条件和终止返回值处理，保持权重冻结。
不要修改原生评分来使结果通过，也不把这两条称作完整任务解决或所有姿态下的力标定成功。
本轮没有修改控制 prompt、重新采集或针对 E4 单独拟合模型。

原始运行目录：

- `/mnt/data/ljz/opentacbench/capx-runs/shadow_5d_matched8_4phys_seed3150_gpu7_smoke_rep1_20261024T162218Z`
- `/mnt/data/ljz/opentacbench/capx-runs/shadow_5d_matched8_repeat_4phys_seed3150_gpu1_smoke_rep1_20261024T170512Z`

两者 `trial_01_sandboxrc_0_reward_0.000_taskcompleted_0/videos/video_combined.mp4` 分别为
27.0／23.2 秒，均含 H264 视频和 AAC 原生接触合成音频。
逐步审计与失败音放大图分别位于数据盘 `shadow_digit_force_v3/` 下的
`candidate_8episode_online_audit_20261024_run1_v2/` 和 `candidate_8episode_online_audit_20261024_run2_v2/`。
报告的原生 target-index 区间可能包含上一音释放尾部；其他键短暂力计数不等于原生 wrong_key，
须结合逐音判分和动作记录解释，不能直接按该计数认定键位错误。

复跑命令（GPU 必须按当前资源明确设置；候选不是默认模型）：

```bash
cd /mnt/a/ljz/cap/t-cap
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY \
    -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
  CAPX_DIR=/mnt/a/ljz/cap/t-cap/cap-x \
  CONFIG_PATH=env_configs/opentac/play_piano_shadow_digit_visual_force_5d_60hz_smoke.yaml \
  CAPX_SHADOW_FORCE_SIDECAR=/mnt/a/ljz/cap/t-cap/cap-x/env_configs/opentac/shadow_digit_key_top_force_5d_matched60hz_8episode.v1.json \
  UNIVTAC_ROOT=/mnt/a/ljz/cap/OpenTacBench \
  GPU=0 MODE=smoke TRIALS=1 REPEATS=1 MAX_TRIAL_RETRIES=0 MAX_TOKENS=8192 \
  TRIAL_TIMEOUT_SECONDS=7200 PROCESS_TIMEOUT_SECONDS=7800 CAPX_DISABLE_PROXY=1 \
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  MODEL=gpt-6-sol RUN_LABEL=shadow_5d_matched8_4phys_seed3150_gpu0 ./run_responses
```

## 八条数据冻结候选（2026-10-24 UTC）

- 版本：`shadow_digit_key_top_force_5d_matched60hz_8episode.v1`，文件位于
  `env_configs/opentac/shadow_digit_key_top_force_5d_matched60hz_8episode.v1.json`。
- SHA256：`c9ebb6a79e73990c844c52c81f80fc42a35a515f1a6327f8b7d5933dc643f2e8`。
- 用 seed 3200–3203 的 expert/probe 共八条拟合，按每 episode 的 0.1 N 分箱，每箱最多取 30 帧；
  白键 822 帧、黑键 1335 帧。固定五维、Median/IQR、线性及逐维平方项 Ridge alpha=0.01；
  OOD 仍用训练范围向外 5%，没有按旧运行的失败点手工拓宽，没有裁剪力值。
- 完整四组 seed 用于拟合，**没有独立最终留出集**。下文四折交叉验证是开发评估，不能当作
  此冻结全量模型的独立测试成绩。新 CaP-X 运行将作为在线审计。
- 运行时 JSON 只有特征契约、传感器校验、归一化、权重、边界及聚合训练来源，没有逐帧真实力。
  API 的 `calibration_version` 优先返回明确权重版本；旧文件回退到原 schema，返回类型不变。
- 原五维 YAML 默认仍载入旧模型。通过环境变量 `CAPX_SHADOW_FORCE_SIDECAR` 显式选择候选；
  首条对照未改变控制 prompt、动作频率、原生任务或判分。

冻结命令（从 cap-x 目录执行，输出文件存在时拒绝覆盖）：

```bash
/mnt/data/ljz/opentacbench/conda-envs/UniVTAC/bin/python \
  scripts/opentac/freeze_shadow_digit_recalibration.py \
  --audit /mnt/data/ljz/opentacbench/shadow_digit_force_v3/audit_3200_3203_20261024_v1 \
  --template env_configs/opentac/shadow_digit_key_top_force_5d_area_rate.v1.json \
  --version shadow_digit_key_top_force_5d_matched60hz_8episode.v1 \
  --output env_configs/opentac/shadow_digit_key_top_force_5d_matched60hz_8episode.v1.json \
  --provenance /mnt/data/ljz/opentacbench/shadow_digit_force_v3/matched60hz_8episode_freeze.private.v1.json
```

### 冻结后的历史失败回放（不是新闭环成绩）

对 `shadow_5d_quality_nocontact_fix_4phys_seed3150_gpu7_smoke_rep1_20261024T094930Z`
已保存的公开五维特征做离线回放。327 次查询中，候选返回 312 次 no_contact、15 次 ok、0 次 OOD；
15 个可用接触查询 MAE=0.03633 N，RMSE=0.07426 N，P95=0.10810 N。
原来 13 个 OOD 均落入候选边界，但其中 E4 第 380 动作步真实 0.41312 N、预测 0.13775 N，
低估 0.27537 N；其余 12 点绝对误差 0.01518–0.03642 N。
这说明“矩形边界内”不保证异常耦合姿态下可辨识，不能因 OOD 变少就宣称标定成功。
此回放没有改变旧动作，旧运行原生 0/8 的结果不变。

私有报告与对齐图：
`/mnt/data/ljz/opentacbench/shadow_digit_force_v3/candidate_8episode_replay_20261024_v1/`。
可用 `scripts/opentac/evaluate_shadow_digit_frozen_runtime.py --run-dir ... --model ... --output ...`
重现；省略 `--model` 时只审计实际在线估计，不做反事实替换。

## Seed 3200–3203 累计审计与离线筛查

新增 3201–3203 六条均 `status=complete`、`error=null`，没有采集 traceback，进程均已退出。
所有采集器 SHA、探针 SHA、传感器 SHA 与 seed 3200 一致。以下原生成功数是独立采集策略的
任务成绩，不是 CaP-X 的 LLM 控制成绩。

| seed | 策略 | 原生成功／尝试 | 动作数 | 公开帧 | 私有标签行（含 step 0） | 实测全程峰值 N |
|---|---|---:|---:|---:|---:|---:|
| 3200 | expert | 8/8 | 1901 | 1902 | 7602 | 1.11295 |
| 3200 | probe | 6/8 | 2054 | 2055 | 8214 | 1.11193 |
| 3201 | expert | 8/8 | 2005 | 2006 | 8018 | 1.12132 |
| 3201 | probe | 5/8 | 2129 | 2130 | 8517 | 1.08646 |
| 3202 | expert | 8/8 | 1999 | 2000 | 7994 | 1.10430 |
| 3202 | probe | 6/8 | 2082 | 2083 | 8326 | 1.10981 |
| 3203 | expert | 8/8 | 1850 | 1851 | 7399 | 1.09693 |
| 3203 | probe | 5/8 | 1986 | 1987 | 7943 | 1.09231 |

累计 16,006 个动作、16,014 个公开帧、64,013 行私有标签；原始 RGB／depth 约 5.53 GB。
每条私有标签从 step 0 连续覆盖到末帧，公开帧精确对齐，常规间隔为 4 个物理步，
末次原生终止保留实际执行的短间隔。实测力无 NaN/负值，没有 0.05 N 以上错键接触或达到 2 N 的物理步。
最低有效 depth 比例均为 0.655651。C4–B4 共十二个键均完成采样，每个键同时有轻、重音；
四组黑白×轻重各 16 次尝试。每个键／力度组合目前仅 2 或 4 次（含两种策略）。

专家累计 **32/32**；探针累计 **22/32**，其中轻音 **16/16**、重音 **6/16**。
10 次探针失败均为 `incomplete_note`，没有错键或采集异常，主因是合格力连续时间不足，
不是键程不够。逐物理步审计如下：

| seed | 失败音 | 最长连续合格力 tick | 秒数 | 要求 |
|---|---|---:|---:|---:|
| 3200 | D#4 heavy | 21 | 0.08750 | ≥29 tick／0.12 s |
| 3200 | A4 heavy | 5 | 0.02083 | 同上 |
| 3201 | C4 heavy | 9 | 0.03750 | 同上 |
| 3201 | G#4 heavy | 10 | 0.04167 | 同上 |
| 3201 | F#4 heavy | 15 | 0.06250 | 同上 |
| 3202 | E4 heavy | 9 | 0.03750 | 同上 |
| 3202 | A#4 heavy | 7 | 0.02917 | 同上 |
| 3203 | F#4 heavy | 22 | 0.09167 | 同上 |
| 3203 | G4 heavy | 8 | 0.03333 | 同上 |
| 3203 | D#4 heavy | 21 | 0.08750 | 同上 |

探针是固定动作采集策略，不读取估计力或真实力做反馈。因此这些失败不能归因为
“估计器导致动作失败”；它们提供的是目标附近不稳定接触的离线标签，必须保留。

### 冻结旧模型回放

接触定义为同一步实测力 ≥0.05 N；初触为每次接触开始后的约 0.12 s，仅用于离线分层。
运行时模型和 OOD 边界保持原样：

| 汇总 | expert | probe |
|---|---:|---:|
| 全部 OOD 帧 | 24 | 41 |
| 初触 OOD／初触帧 | 16/240 | 34/233 |
| 可用接触估计／接触帧 | 2698/2726 | 1734/1777 |
| 可用接触帧 MAE | 0.00327 N | 0.00614 N |
| 可用初触帧 MAE | 0.00997 N | 0.02227 N |

上述误差只针对接受帧，不代表旧模型在 OOD 帧也准确。
加上六条后，上一条 CaP-X 的 13 个 OOD 点在“同色、相近真实力”的逐维范围中覆盖数由
4/13 增至 8/13。四个白键约 0.172 N 的初触已出现更接近的新样本；逐维范围仍有轻微越界，
不能把范围检查当作力可辨识性证明。一个约 0.413 N 的 E4 旧运行异常接触仍缺少接近样本，
对应第一段代码降腕和弯指耦合的过程，不能用正常先降腕再弯指的数据宣称已覆盖。

### 探索性整 seed 留出：4D 与 5D

仅用这八条新数据，固定 Median/IQR＋线性／逐维平方项 Ridge（alpha=0.01），黑白键分开。
四折：每折用三个 seed 的六条轨迹训练，完整留出剩下一个 seed 的专家和探针两条。
同 seed 的两种策略没有跨训练／测试集；无随机拆帧。两种维度使用相同的历史有效帧。
模型范围仅从训练帧计算（原 5% padding），未从留出标签放宽范围，未裁剪预测。

| 留出探针指标 | 4D | 5D（加面积变化率） |
|---|---:|---:|
| 接触帧数 | 1777 | 1777 |
| 接触 MAE（全部原始预测） | 0.01690 N | 0.00679 N |
| 接触 P95 绝对误差 | 0.04311 N | 0.02232 N |
| 初触 MAE（233 帧） | 0.04216 N | 0.01734 N |
| 初触 RMSE | 0.07047 N | 0.02448 N |
| 初触 P95 绝对误差 | 0.16443 N | 0.05630 N |
| 初触偏差 | +0.03629 N | +0.00678 N |
| 初触 OOD | 0/233 | 0/233 |
| 接触不可用帧 | 0/1777 | 1/1777 |

5D 的一个不可用接触帧被预测为 <0.05 N，返回 no_contact；没有被当作可用估计。
5D 白键重音初触 MAE 0.02304 N、P95 0.06698 N，仍接近白键 0.07 N 的任务容差，
不宜因此认为闭环控制已经可靠。四折非接触帧 11,488 个，按估计 ≥0.1 N 计误报为零；
**目标容差上界以上的超力帧为零，超力漏检能力无法评估**。

这组筛查支持保留面积变化率、继续扩充匹配 60 Hz 的采集数据。只有四个 seed，
采集动作高度相似，不能宣称模型已完成独立验证；3200–3279 原规划训练／3280–3299
独立留出约定继续保留，尚未使用后者数据。

下一批应同时补降腕／弯指耦合接触和目标容差上方的有界探针，而不是仅增加相同
正常策略的 seed。过量程 >2 N 的帧仍要单独标记，不能用饱和外观拟合成可靠力值。
此处未修改采集动作或任务，未开始下一批、未导出候选运行时权重或运行 CaP-X。

### 本轮产物与复现

```text
/mnt/data/ljz/opentacbench/shadow_digit_force_v3/audit_3200_3203_20261024_v1/
  report.private.json
  pilot_grouped_cv.private.json
  <episode>.private.npz
  <seed>_<expert|probe>_alignment.png
  coverage_old_new_runtime.png
```

原始 HDF5 在 `matched_60hz_v1/episodes/` 中保留。
离线筛查命令（输出必须是新文件）：

```bash
cd /mnt/a/ljz/cap/t-cap/cap-x
/mnt/data/ljz/opentacbench/conda-envs/UniVTAC/bin/python \
  scripts/opentac/evaluate_shadow_digit_recalibration_pilot.py \
  --audit /mnt/data/ljz/opentacbench/shadow_digit_force_v3/audit_3200_3203_20261024_v1 \
  --output /mnt/data/ljz/opentacbench/shadow_digit_force_v3/pilot_grouped_cv_recheck.private.json
```

以下 seed 3200 单对审计保留作历史对照。

## Seed 3200 实采结果

两条均使用 GPU 4，采集器／探针脚本 SHA 与当前本地文件一致。
音序为 `D#4重、A4重、G4轻、C#4轻、C#4重、G4重、D#4轻、A4轻`。
这是独立数据采集，没有 LLM 代码生成；下表不能算成 CaP-X smoke 成功率。

| 控制策略 | 原生成功／尝试 | 动作数 | 公开帧 | 私有标签行（含 step 0） | 实测全程峰值 | 采集耗时 |
|---|---:|---:|---:|---:|---:|---:|
| expert | 8/8 | 1901 | 1902 | 7602 | 1.11295 N | 346.95 s |
| probe / varied | 6/8 | 2054 | 2055 | 8214 | 1.11193 N | 383.77 s |

- `status=complete`、`error=null`，两条的采集进程均已退出。未见 Python traceback。
  Kit 的 `DriverShaderCacheManager::init()` 提示没有阻止采集，不能据此判成采集失败。
- 公开步长均为 4 tick；各自最后一个动作在原生终止时只执行 1 tick，因此保留了一个
  1 tick 的末帧。这不是缺步，私有物理步连续且逐帧精确对齐。
- 深度有效比例最低均为 0.655651；参考配置 SHA 与旧冻结模型相同。
  原始 RGB／depth 合计约 1.37 GB。两条没有任何物理步达到 2 N，没有错键接触（阈值 0.05 N）。
- 探针失败为第 1 音 D#4 重音和第 2 音 A4 重音，均为 `incomplete_note`。
  连续合格力最长分别 21 tick（0.0875 s）、5 tick（0.020833 s），低于要求的
  29 tick（至少 0.12 s）；这两个区间的键程条件也已核对。不是采集脚本异常。
  固定探针不做力反馈，其失败样本需要保留。

### 旧模型回放：不是新模型留出评估

逐帧计算公开特征后，才打开私有标签作比较；没有把标签用于控制或更新模型。

| 统计 | expert | probe |
|---|---:|---:|
| 全部 OOD 帧 | 6 | 11 |
| 初触 OOD／初触帧 | 4/60 | 9/58 |
| 接触帧中的可用估计／接触帧 | 633/641 | 433/444 |
| 可用接触帧 MAE | 0.00325 N | 0.00630 N |
| 可用初触帧 MAE | 0.01001 N（56 帧） | 0.02187 N（49 帧） |

所有 OOD 都涉及面积变化率，探针有一帧四个静态特征也越界。误差仅针对旧模型接受的
帧，不能忽略 OOD 后称整体精度达标。末帧与开头的缺历史返回 `insufficient_history`。

新探针的白键最大面积变化率约 9.325/s、黑键约 9.676/s，明显扩展旧专家初触范围。
上一条 CaP-X 的 13 个 OOD 点均落入新数据同色键的逐维总体范围，但这仅是范围检查；
限定在相近真实力的局部范围时仅 4/13 完全落入。黑键已有很接近的特征样本，
白键部分大形变初触仍覆盖不足，不能直接断言可可靠反推真实力。
当前两条只实际按了 G4/A4/C#4/D#4；上一条 CaP-X 的白键是 E4/F4，接触过程也有差异。

因此建议下一小批先补 seed 3201–3203（6 条），使累计 4 个配对 seed 覆盖 C4–B4
全部 12 个键，并继续比较白键初触；确认覆盖后再决定扩到 200 条。不得先放宽 OOD
边界掩盖差异。未修改控制 prompt、运行时估计器、传感器配置或 OpenTacBench。

审计产物（离线／私有）：

```text
/mnt/data/ljz/opentacbench/shadow_digit_force_v3/audit_3200_20261024_v1/
  report.private.json
  3200_expert_a46a17630b.private.npz
  3200_probe_a46a17630b_varied.private.npz
  expert_alignment.png
  probe_alignment.png
  coverage_old_new_runtime.png
```

审计入口为 `scripts/opentac/audit_shadow_digit_recalibration.py`，传入 `--episode`（可重复）、
`--model`、`--expert-cache`、`--runtime-report` 和一个新的 `--output` 目录即可复现。

## 为什么重新采

旧 50 条是 30 Hz 成功专家轨迹。最近 60 Hz CaP-X 的 13 个 OOD 点全部越过
面积变化率范围，其中 5 个还同时越过四个静态深度特征范围；对应真实力仅
0.1313–0.4529 N，并未达到 2 N 饱和。因此应增加当前控制条件下的接触覆盖，
单纯增加旧策略 seed 数不能保证解决问题。详见同目录的五维运行诊断。

## 新批次设计

- 新数据目录：`/mnt/data/ljz/opentacbench/shadow_digit_force_v3/matched_60hz_v1/`。
  旧数据、旧模型和正在使用的 smoke 配置均保留。
- 物理 240 Hz，`action_repeat=4`，公开 RGB/depth 60 Hz，私有实测琴键力 240 Hz。
  面积变化率仍取 8 个物理步的因果窗口，即相隔两个新公开帧，不能把窗口误改成 4 步。
- TACTO 0.0.3、320×240、原 `[0,2] N` 配置；继续保存原始 RGB、depth、参考深度、
  valid mask、动作、公开手部状态、指定 MIDI 和 `requested_note_index`。
  私有文件记录完整 88 键实测力及原生判分，逐物理步与公开数据对齐。
- 每 seed 采两条、每条 8 音：原生专家一条，独立公开状态脚本探针一条。
  每条白轻、白重、黑轻、黑重各两次，seed 之间轮换 C4–B4 的 12 个键。
- 专家保持其原生策略。探针只读取指定音符和实际手部位置，不用真实力、原生判分
  或旧估计器决定按压。后者不在旧模型 OOD 时提前释放，避免新数据再次缺失初触过程。
  探针失败也保留，不能训练时再筛成“只有成功专家”。
- 探针采用当前 prompt 的按压高度先验、0.50 rad 平滑弯曲、0.30 s 释放，随后抬手。
  `runtime` 为 0.45 s 弯曲；`higher/lower` 在其高度上加/减 0.3 mm；
  `slower` 为 0.75 s 弯曲。四种配置在采集前冻结，不由标签调整。
  `varied` 每个 seed 在四个键上分配这四种配置，同键轻重使用相同配置。
  这些是待验证的采集动作，不是已验证的安全控制器或真实力标签。
- 使用已有原生 `piano_continue_on_note_failure` 和逐音结束接口保留失败样本。
  不更改任务源码、物理、力阈值或判分。原生硬终止仍会停止采集。
- 不录 MP4/WAV，不请求 LLM；RGB 原始观测仍保存在 HDF5。

## 配对试采命令（3200 已完成）

在确认指定 GPU 空闲后执行。默认只采 seed 3200 的专家和探针各一条，串行，
每条超时 1200 秒；失败即停，不覆盖已存在日志或不完整数据。

```bash
cd /mnt/a/ljz/cap/t-cap
GPU=0 START_SEED=3200 END_SEED=3200 \
  bash ./cap-x/scripts/opentac/run_shadow_digit_recalibration.sh
```

日志位于新目录的 `logs/seed_3200_expert.log` 和 `logs/seed_3200_probe.log`。
每条目录包含 `public.h5`、`private_labels.h5`、`result.json`、`audit.json`、
`alignment.png`、`renderer.log`。`result.json` 保存动作配置、采集脚本 SHA、传感器 SHA
和 OpenTacBench/cap-x commit。私有标签不能作为运行时记忆或传给生成代码。

先审计以下项目，再扩量：

1. 公开采样间隔确为 4 tick，私有标签连续；所有公开帧能精确对应实测力。
2. 四组实际都有接触，完整保存初触、保持和释放；检查实测接触了哪个键。
3. 深度有效比例、参考深度和传感器 SHA 一致；超过 2 N 的帧单列为饱和区域。
4. 将新探针的四个静态特征及面积变化率与这次 CaP-X OOD 区域对照。
   如果仍不覆盖，先调整下一批采集动作，不能直接放宽旧模型 OOD 边界。

## 扩量候选：200 条新轨迹

本轮八条审计后，需先补耦合初触和超力采样策略；现有 `varied` 策略还不覆盖这两类。
以下保留原扩量模板，单独执行它不能补齐本次指出的所有缺失区域。

建议范围为 seed 3200–3299，每 seed 两条，共 **200 条、1600 次计划按键**。
这不是已经采完的数据，也不能按数量保证精度。先完成上述配对试采审计。

```bash
cd /mnt/a/ljz/cap/t-cap
GPU=0 START_SEED=3200 END_SEED=3299 \
  bash ./cap-x/scripts/opentac/run_shadow_digit_recalibration.sh
```

预先约定训练 seed 3200–3279、留出 seed 3280–3299。同 seed 的专家和探针必须
进入同一集合；不能随机拆帧。旧 50 条作为独立旧域对照，是否混入训练需另列消融。

下一步离线训练须单独实现新批次读取：旧 `evaluate_shadow_digit_area_rate.py`
限制成功专家、`action_repeat=8`、3100–3149，**不能直接用于本批次**。
新版本先沿用 4D/5D 比较，不增加关节弯曲度作为力特征；单列初触、重音、释放、
每个琴键和控制配置的误差、OOD/不可用比例、超力漏检与无接触误报。
留出集不能用于决定特征边界或调整模型。随后另跑新 CaP-X smoke 检查控制域泛化，
通过后才替换运行时 sidecar。逐帧真实力不能写进公开 sidecar。

TACTO 外观仍由模拟接触力驱动；本标定目标是琴键顶部实测接触力，
不是独立测得的指尖法向力，也不能推广成真实 DIGIT 硬件的已验证牛顿估计。
