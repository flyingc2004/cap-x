# DIGIT 短时触觉检索编码器 v1

这是独立的离线实验，输出 32 维轨迹 key。没有接入 CaP-X 控制、修改技能库或 OpenTacBench，也不表示已验证技能复用收益。当前数据加载器仅支持本机 DIGIT public.h5；Franka 适配交接见本文末尾。

## 固定实验

- 3100–3129 训练，3130–3139 开发，3140–3149 测试；3200–3203 expert/probe 全部作为外部分布检查。
- 仅读取公开深度、参考深度、掩码、时间及可选的四维目标/四维实际状态。元数据仅提取 seed、传感器校验和时间单位用于划分/兼容性。
- 有符号形变毫米图与有效比例缩小到 24×32。过去 0.5 秒取 16 点；只选择各目标时间之前最近的观测，最大允许年龄 1/30 秒；每帧有效比例至少 60%。
- CNN 16/32/32 → 每帧 64 维 → GRU 64 → 32 维。纯触觉和动作条件版本分别运行训练随机种子 0/1/2，串行执行。
- VICReg 25/25/1，同窗噪声标准差 0.01（训练归一化尺度）和 2% 像素遮蔽；不预测未来，不重建深度，不使用真假成功或专家阶段标签。
- AdamW 1e-3，weight decay 1e-4，batch 128，最多 50 epoch，开发损失 10 epoch 无改善则停。归一化仅拟合训练集，全无接触窗口最多占训练采样的 25%。
- 开发集只用于 checkpoint 选择；所有权重冻结后才评估留出数据、打开私有真实力。模型没有使用力估计器输出或真实力训练。

VICReg 是借鉴的训练机制，不是本项目提出的新算法：[论文](https://arxiv.org/abs/2105.04906)。

## 本机运行

在 cap-x checkout 内执行；输出目录必须不存在。以下命令默认不访问网络，不自动选择 GPU。

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /mnt/data/ljz/opentacbench/conda-envs/UniVTAC/bin/python \
  scripts/opentac/train_digit_trajectory_encoder.py suite \
  --run /mnt/data/ljz/opentacbench/tactile_trajectory_encoder_v1/填写新的运行名 \
  --device cpu --threads 4
```

suite 顺序执行预处理、六次训练、公开检索评估、独立私有力审计，写入 run.log 和 status.json。失败立即停止并记录非零退出状态。若手动指定 cuda:N，只使用该设备；CPU 为默认。

也可分阶段运行 prepare、train、evaluate，最后单独调用 scripts/opentac/audit_digit_trajectory_neighbors.py --run RUN。源代码会在 prepare 时快照并记录哈希；训练/评估前检查一致性。恢复失败任务时先检查已存在的产物，不覆盖已有训练目录。

每个 models/tactile[_action]/seed_N 目录保存配置、epoch 日志、best.pt 和训练摘要。freeze.json 固定权重和数据契约。evaluation 保存训练集索引、每个 key 的轨迹起止 step、两组留出近邻和深度图；private_audit 保存真实力对照，REPORT.md 汇总结果。

encode 入口接受 --checkpoint、--episode、--end-step、--output，从已有公开缓存导出 key。search 入口接受 --index、--query 和 --k；不允许不同权重、传感器配置或窗口契约的 key 混查。原始历史接口为 FrozenTrajectoryEncoder.encode_window，不依赖 Kit。

## 指标解释

- 当前四维、同窗四维展开和两类学习编码器使用相同训练 gallery、相同留出 query。
- 同窗增广的精确身份 Recall@1/5 只检查稳定性，不是跨 seed 技能准确率。原生高分辨率四维基线与低分辨率图像扰动不是同一路径，因此不虚构其可比的增广成绩。
- 时间打乱固定最后一帧，避免把最后观测变化误当成历史信息贡献。
- 真实力曲线 MAE/RMSE 是冻结后的物理相似性审计。无接触、接触和超 2 N 窗口分开报告；不把低频观测推断为全部物理步稳定。
- 50 条固定任务数据与 8 条探针不足以证明跨任务、跨传感器或真实机器人泛化。未优于简单基线也保留结果，不自动扩大实验。

## 给另一台服务器 agent 的 Franka 交接提示词

请在现有 t-cap/cap-x checkout 中读取 docs/digit-trajectory-encoder-v1.md，以及 capx/memory/trajectory 下实现，为本机 Franka 双侧 GelSight 数据新增适配并训练独立检索编码器。先检查 git 工作区并保留所有已有改动；不要新建 checkout、修改 OpenTacBench、启动 CaP-X 控制或恢复不相关下载。

1. 先审计本机实际数据：完整 seed 数、时间单位、左右数值 depth/marker、无接触参考或已有深度零点契约、公开机器人状态及是否记录动作指令。不要把 metadata、phase、评分、真实力或专家动作阶段作为编码输入。DIGIT 的 0.5 秒/16 点时间契约可复用，DIGIT 权重和归一化不可复用。
2. 当前 DIGIT 加载器不能直接读取 Franka HDF5，必须新增只读适配器和独立 schema。参考已提交的 public_force_calibration.py 确认 GelSight 深度单位及 far_plane 约定，不能未经核实套用毫米值或伪造 reference。左右各保留形变和有效掩码，分别经共享小 CNN 编码；按固定左/右顺序融合到每帧 64 维，再沿用 GRU 64 和 32 维 key。不要简单平均两侧，也不要伪造单侧缺失观测。
3. 若确认为钢琴 mini-depth-marker 的 2200–2229 批次，固定 2200–2217 训练、2218–2223 开发、2224–2229 测试；其他数据先生成明确的完整 seed 分割清单。同 seed 的多条轨迹必须同组，检查原始副本，不允许逐帧切分。
4. 沿用两视图 VICReg、轻微噪声与遮蔽、AdamW、50 epoch/10 epoch early stop、随机种子 0/1/2和纯触觉主模型。动作版仅使用实际记录且运行时公开的指令与状态；如果文件只有实际状态，没有动作目标，就先完成纯触觉训练，将可选对照准确称为“状态条件”，不要把实际位置假装成动作指令。
5. 固定开发集选择权重，冻结后评估简单特征基线、同窗匹配、跨 seed 近邻与时序敏感性；若有实测力，独立脚本在冻结后审计，绝不写入运行时 index 或用于选模型。只在同一编码器版本内检索，不声称两个独立模型已形成共同空间。
6. 大文件写本机数据盘的新目录，设备由调用参数明确指定，不自动选择 GPU。完成必要的时间对齐/私有数据隔离/保存加载测试后，后台串行训练，报告进程、日志、权重、结果路径和仍缺少的数据。不要升级 Isaac、覆盖环境或修改已有任务物理、判分。
