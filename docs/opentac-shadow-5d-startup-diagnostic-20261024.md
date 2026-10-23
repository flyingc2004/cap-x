# Shadow 5D smoke 启动失败诊断：2026-10-24

## 结论与停止条件

本次未进入钢琴试验。Isaac Kit 在创建 GPU/CUDA 上下文时失败，随后启动器未输出
trial 结果。已取得足够证据定位到当前执行环境的 CUDA 设备可用性，停止追加仿真运行。
没有修改 OpenTacBench、控制提示词、力模型参数、OOD 门槛或驱动，没有停止其他进程。

`nvidia-smi` 能列卡与 CUDA 可计算设备可用是两项检查。当前服务器执行环境中，
两种 Python 直接调用真实驱动库都得到 `CUDA_ERROR_NO_DEVICE (100)`；不能仅据此
进一步断定驱动损坏或指定硬件修复操作。

## 失败运行

- 北京时间：2026-10-24 02:37:39；原始目录采用 UTC 标记 `20261023T183739Z`。
- 运行：`shadow_5d_area_rate_4phys_seed3150_gpu0_smoke_rep1_20261023T183739Z`。
- 配置：Shadow Hand + DIGIT，5D 面积变化率模型，seed 3150，每动作 4 个物理步。
- [run.log](/mnt/data/ljz/opentacbench/capx-runs/shadow_5d_area_rate_4phys_seed3150_gpu0_smoke_rep1_20261023T183739Z/run.log)。
- [Isaac Kit 日志](/mnt/data/ljz/opentacbench/conda-envs/UniVTAC/lib/python3.10/site-packages/omni/logs/Kit/Isaac-Sim/4.5/kit_20261023_183744.log)。

| 证据 | 位置／结果 |
|---|---|
| GPU 设备创建失败 | run.log 第 29、51 行：`No device could be created` |
| CUDA 设备创建失败 | run.log 第 69 行：`no CUDA-capable device is detected` |
| 结果完整性检查 | run.log 第 82 行：启动器返回 0，但没有完成 trial |
| 最终包装脚本退出码 | run.log 第 83 行：`exit_code=1` |
| 产物 | 目录只有 run.log，无 renderer.log、生成代码、视频或原生计分摘要 |

未出现 5D 模型初始化日志，也未出现 `PIANO_NOTE` 或 END。此次不能记为“0/8 个音”，
也不能用于评价五维估计器、OOD 恢复策略或按键控制。动作和计分账本保留缺失值。

## 本轮实际检查

先发现沙箱的 `/dev` 未暴露 NVIDIA 设备，沙箱内 `nvidia-smi` 退出 9。
随后在服务器执行环境重新检查，排除了只根据沙箱错误作判断。

| 检查 | 结果 |
|---|---|
| 服务器 `nvidia-smi` | 退出 0；580.95.05，8 张 RTX 4090，无计算进程；部分状态字段 ERR/N/A |
| NVIDIA 设备节点 | 服务器环境可见 nvidia0–7、nvidiactl、nvidia-uvm |
| 节点普通文件权限 | nvidia0、nvidia1、nvidiactl、nvidia-uvm 均 0666，当前用户可读写 |
| UniVTAC Python | torch 2.5.1+cu124；`cuda_available=False`；检查程序退出 2 |
| Conda 直接驱动检查，限制 GPU 0 | `cuInit(0)=100 CUDA_ERROR_NO_DEVICE` |
| 系统 Python，去掉 GPU 限制、LD_LIBRARY_PATH 和 LD_PRELOAD | `cuInit(0)=100 CUDA_ERROR_NO_DEVICE` |
| 两次实际加载的库 | `/usr/lib/x86_64-linux-gnu/libcuda.so.580.95.05` |

### 后续健康状态检查：8 张卡均要求重置

继续只读运行 `nvidia-smi -q -i 0`，并用 `nvidia-smi -q -x` 汇总全部 GPU，得到：

| GPU | PCI 地址 | 状态 | GPU Recovery Action |
|---|---|---|---|
| 0 | 00000000:13:00.0 | GPU requires reset | Reset |
| 1 | 00000000:2E:00.0 | GPU requires reset | Reset |
| 2 | 00000000:49:00.0 | GPU requires reset | Reset |
| 3 | 00000000:64:00.0 | GPU requires reset | Reset |
| 4 | 00000000:98:00.0 | GPU requires reset | Reset |
| 5 | 00000000:B1:00.0 | GPU requires reset | Reset |
| 6 | 00000000:CA:00.0 | GPU requires reset | Reset |
| 7 | 00000000:E3:00.0 | GPU requires reset | Reset |

GPU 0 显存使用仅 1 MiB，其他卡同样仅几 MiB，没有显存溢出证据。
设备节点的主次编号与 `/proc/devices`、内核 GPU 登记一致，驱动库与内核模块
版本同为 580.95.05。已进一步明确当前阻断是 GPU 需要恢复的状态。

触发这一状态的底层事件尚未查明：`dmesg` 权限不足，普通用户 `journalctl -k`
看不到系统记录，非交互 `sudo -n journalctl` 需要密码。尚无法确认具体 Xid
或将原因归到某一次仿真、驱动缺陷、供电或 PCIe 硬件事件。

因此，修改 GPU 编号、Conda 配置或钢琴 prompt 暂无证据能够解决此启动错误。
需管理员读取对应内核日志并按设备恢复要求处理；本轮没有执行 GPU reset、
驱动卸载／重载或整机重启。

## 已有模型结论与未完成工作

- 已有专家留出结果支持面积变化率：初触 MAE 从 0.02419 N 降至 0.00475 N，
  白键重音初触从 0.04005 N 降至 0.00693 N。
- 运行时已通过 33 项 CPU 检查，以及 946 帧留出深度回放；五维特征与离线缓存误差为 0。
- 本轮没有得到新的 GPU 任务结果，不能宣称已验证 CaP-X 初触姿态下的五维力估计。
- 恢复实验的最低条件是 CUDA 驱动初始化成功，并能在指定空闲卡上完成最小 CUDA 计算；
  随后再运行一条 seed 3150 smoke，按私有审计比较估计力、真实力和原生逐音结果。

此报告是基础设施诊断，不加入给 agent 使用的运行时 skill library。
