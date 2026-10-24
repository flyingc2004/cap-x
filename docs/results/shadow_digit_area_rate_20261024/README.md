# Shadow DIGIT 五维力标定结果

此目录随 Git 分发，供另一台服务器核对离线结果。运行 smoke 所需的模型与传感器配置
也已纳入仓库，启动不需要复制 50 条原始 HDF5 轨迹。

| 文件 | 内容 |
|---|---|
| [comparison_report.json](comparison_report.json) | 4D/5D 全部指标、按键与阶段分组、训练／留出 seed、质量统计 |
| [models_offline_only.json](models_offline_only.json) | 此次离线对照的 4D 和 5D 参数，供复核 |
| [heldout_alignment.png](heldout_alignment.png) | 留出轨迹：预测力、私有真实力、面积与变化率对齐图 |
| [initial_contact_scatter.png](initial_contact_scatter.png) | 初触预测与真实力散点图 |
| [manifest.json](manifest.json) | 本包 6 个核心文件的大小和 SHA256 |
| [冻结运行时模型](../../../env_configs/opentac/shadow_digit_key_top_force_5d_area_rate.v1.json) | 仅模型系数、特征顺序、范围和历史窗口，无逐帧真值 |
| [TACTO 配置](../../../env_configs/opentac/config_digit_piano_0_2.v1.yml) | 原始配置仅将 range_force 改为 [0,2]，与标定时字节一致 |

在 cap-x 根目录验证：

```bash
python scripts/opentac/check_shadow_digit_bundle.py
```

完整方法、指标及局限见[离线对照报告](../../opentac-shadow-digit-area-rate-ablation-20261024.md)。
另一台服务器的路径、环境和 smoke 命令见
[t-cap 钢琴迁移文档](https://github.com/flyingc2004/t-cap/blob/feat/opentac-piano-tension-integration/docs/shadow-piano-handoff.md)。
本包不包含逐帧私有力标签、原始深度轨迹或凭据。图中的真实力仅用于人工离线审计，
运行时 API 和模型输入不读取这些图或评估文件。
