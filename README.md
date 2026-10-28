


# CaP-X

### A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation

[Project Page](https://capgym.github.io/) &ensp;|&ensp; [Paper](https://arxiv.org/abs/2603.22435) 

**Max Fu<sup>&#42;,1,2</sup>, Justin Yu<sup>&#42;,2</sup>, Karim El-Refai<sup>&#42;,2</sup>, Ethan Kou<sup>&#42;,2</sup>, Haoru Xue<sup>&#42;,1,2</sup>,
Huang Huang<sup>3</sup>, Wenli Xiao<sup>4</sup>, Guanzhi Wang<sup>1</sup>, Fei-Fei Li<sup>3</sup>, Guanya Shi<sup>4</sup>, Jiajun Wu<sup>3</sup>,
Shankar Sastry<sup>2</sup>, Yuke Zhu<sup>1</sup>, Ken Goldberg<sup>&dagger;,2</sup>, Jim Fan<sup>&dagger;,1</sup>**

<sup>1</sup>NVIDIA &ensp; <sup>2</sup>UC Berkeley &ensp; <sup>3</sup>Stanford University &ensp; <sup>4</sup>Carnegie Mellon University

<sup>&#42;</sup>Equal contribution &ensp; <sup>&dagger;</sup>Equal advising



---
**CaP-X** is an open-access framework for systematically studying Code-as-Policy agents in robot manipulation. It consists of four components:


| Component      | What it does                                                                                                                                                                                   |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CaP-Gym**    | Interactive Gymnasium environments where agents control robots by generating Python code that composes perception and control primitives. 39 tasks across Robosuite, LIBERO-PRO, and BEHAVIOR. |
| **CaP-Bench**  | Systematic benchmark evaluating coding agents across abstraction levels, interaction modes, and visual grounding modalities. 8 tiers (S1-S4 single-turn, M1-M4 multi-turn).                    |
| **CaP-Agent0** | Training-free agentic framework with multi-turn visual differencing, auto-synthesized skill libraries, and parallel ensembled reasoning.                                                       |
| **CaP-RL**     | Reinforcement learning on the coding agent via GRPO, using environment rewards to post-train language models. Transfers from sim to real with minimal gap.                                     |


---

## 本分支：触觉 key 与代码补丁闭环（2026-10-28）

**实现代码补丁闭环框架，正在实验触觉key是否有用。** 当前验证任务为 OpenTacBench
Franka＋Shadow Hand＋单指 DIGIT 钢琴。下面是本分支的实现与实验状态，不表示已在其他任务上验证。

### 何时查询记忆

首次生成完整策略模块，或显式加载受控实验的历史源码；执行期间持续记录公开触觉、动作、
实际手部状态与函数调用。**只有发生已有公开检查认可的错误、准备修复代码时才查询历史**，
正常执行与 episode 结束均不额外请求 LLM 整理经验。

```text
当前公开错误 → 定位函数调用与触觉窗口 → 兼容历史检索
            → 当前代码/错误/证据 + 历史修正 → LLM补丁
            → 校验并执行 → 原检查重检 → 保存修复经历与代码版本
```

未命中时，LLM仍用当前错误、源码和当前公开证据修复。代码从 run_piano 入口重新进入，
根据宿主的公开提交记录恢复；已提交音不重弹，当前音先读取实际手部与触觉状态，不能依靠
生成代码自己的计数，也不能把请求API期间的墙钟时间算作物理保持时间。

### 触觉 key 的表示设计

即时与时序表示并行存在，共同关联同一接触片段；**时序 key 不替换冻结的5D力估计器**。
这里的 key 指检索向量，不是琴键坐标。

| 表示 | 维数 | 输入与构造 | 用途 |
|---|---:|---|---|
| instantaneous_5d | 5 | 有效像素上绝对深度形变均值、形变超过0.01/0.05/0.2 mm的像素比例，以及超过0.05 mm比例的时间变化率 | 保留即时测力与触觉检索；力模型和特征顺序冻结 |
| window4 | 64 | 最近0.5秒的16帧，每帧使用均值＋三个像素比例；按冻结feature_mean/feature_std标准化，时间展开为16×4，再L2归一化 | 不经学习的时序检索基线 |
| tactile | 32 | 16帧深度相对参考面的有符号形变与有效覆盖图，降采样至24×32；冻结CNN＋GRU编码，再L2归一化 | 触觉轨迹检索 |
| tactile_action | 32 | 同上，额外输入每帧4D动作目标和4D实际状态；XYZ相对窗口首帧实际位置，使用冻结归一化 | 结合接触和动作过程的检索 |

实际状态字段顺序为 wrist_x_offset_m、wrist_y_m、wrist_z_m、index_flexion_rad。
四物理步一次动作时观测为60 Hz；时序窗口必须按物理时间选择16个30 Hz采样点，
不能直接取最后16条。允许从同episode的前一动作阶段补足真实历史，但不跨reset，
不插值、不补帧、不把缺失状态填成0。历史不足、缺失有效掩码、传感器或参考深度不兼容时
返回不可用，不伪造向量；时序通路不可用不影响原5D控制通路。

编码器通过 contract 和 encode(window) 接口接入，保留向量、版本、质量及原始样本来源。
现有适配器封装 FrozenTrajectoryEncoder.encode_window；后续编码器可以输出不同维数，
只需使用独立的编码器/预处理版本，不能与旧空间直接比较距离。

### 从 key 找到可执行经验

- 仅检索此前CaP-X公开执行中具有修复关联的经历；专家离线索引不作为可执行技能库。
- 先匹配错误类型、传感器/力模型、动作周期、坐标系、黑白键、轻重、动作阶段，
  时序表示还必须匹配编码器、归一化与窗口版本。维度相同不代表空间兼容。
- 当前优先查询 tactile_action、tactile、window4，再尝试有效的5D；各空间独立排序，
  不混合不同空间的距离。时序使用归一化向量的均方根距离，5D使用冻结IQR尺度。
  去重后最多提供两条历史修复；当前没有经标定的相似度拒绝阈值，距离近不等于经验必然有效。
- 沿 clip → invocation/function → 历史补丁 → 公开执行检查取证据，给LLM的是可解释
  的错误、轨迹摘要和实际修正代码，不把64/32维数字直接当作语言经验。
- 保存连续RGB/depth、参考深度、掩码的文件引用与校验值，保留失败、OOD和不完整过程。
  真实力、原生判分、reward和success仅进入独立私有审计，不参与编码、检索或补丁晋升。

### 补丁与验证边界

当前释放错误实验向LLM提供完整源码作为只读上下文，只允许修改嵌套release函数与
当前未完成尝试的恢复分支；区域外源码、力目标、物理/判分与动作硬边界保持不变。
其他错误沿用既有修复协议。每episode最多一次修复请求，300秒、8192输出token，API清除代理。

修复前错误永久保留；修复后从明确执行边界检查新函数是否实际调用、原检查是否通过及
是否出现新错误。**局部释放修复有效**可以作为检索经验，但不自动获得**整段策略可直接复用**
资格。原生终止后的合法提交仅用于生命周期收尾，不冒充释放验证或官方成功。

本轮已验证3300经历被3306检索，历史释放函数进入实际补丁；3306原生6/6，
其修正版在3301零API执行得到5/6，原释放错误未复发。无历史3306组因补丁哈希漏抄一位而
未执行，且也提出了合理计时修复；因此尚不能据此宣称触觉key提高了修复成功率。
相同首音产生了距离为0的容易匹配，仍需更多接触过程及语言检索对照。相关CPU测试140项通过。

入口配置：[play_piano_shadow_digit_evidence_memory.yaml](env_configs/opentac/play_piano_shadow_digit_evidence_memory.yaml)，
在父仓库使用现有 ./run_responses。设置CAPX_SHADOW_TASK_SEED、CAPX_SHADOW_EVIDENCE_ROOT、
CAPX_SHADOW_TRAJECTORY_RUN及显式GPU；模型目录需包含cache/manifest.json、cache/normalization.json
和models/{tactile,tactile_action}/seed_0/best.pt。权重、连续触觉帧、视频及私有审计保留在数据盘，
不随Git上传；另一台服务器需另行配置这些路径。

实现：[编码器接口](capx/memory/trajectory/model.py)、[采样与预处理](capx/memory/trajectory/data.py)、
[公开证据索引](capx/integrations/opentac_shadow_online_skills.py)、
[错误检索与代码修复](capx/integrations/opentac_shadow_code_memory.py)。
详情见[代码记忆设计](docs/opentac-shadow-piano-code-memory.md)和
[实验账本](docs/opentac-shadow-piano-experiment-ledger.md)。账本中的服务器绝对路径用于定位原始产物，
在其他机器上需要映射或单独复制。

## Installation

CaP-X uses [uv](https://docs.astral.sh/uv/) for dependency management. Requires **Python 3.10** and a **CUDA-capable GPU**.

```bash
git clone --recurse-submodules https://github.com/capgym/cap-x && cd cap-x

# Or if already cloned without --recurse-submodules:
git submodule update --init --recursive

# Install uv (if not present)
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.10 && uv venv -p 3.10

# Base install
uv sync
```

### Simulator-specific setup

Pick **one** simulator family to install, as Robosuite (1.5.0) and LIBERO (`robosuite==1.4.0`) would be in conflict.

#### Robosuite

```bash
uv sync --extra robosuite
```

#### LIBERO-PRO

LIBERO requires a **separate virtual environment**.

```bash
uv venv .venv-libero --python 3.12
source .venv-libero/bin/activate
uv sync --active --extra libero --extra contactgraspnet
```

See [docs/libero-tasks.md](docs/libero-tasks.md) for running any of 130+ LIBERO tasks.

#### BEHAVIOR (Isaac Sim)

BEHAVIOR tasks run on NVIDIA Isaac Sim via OmniGibson. Requires Python 3.10 and CUDA 12.x.

```bash
cd capx/third_party/b1k
./uv_install.sh --dataset          # installs OmniGibson, Isaac Sim, BDDL, cuRobo, and downloads assets
cd ../../..                        # back to repo root

# Post-install fix — copy cuRobo JIT headers (run with b1k venv active)
source capx/third_party/b1k/.venv/bin/activate
cp capx/third_party/curobo/src/curobo/curobolib/cpp/*.h \
   $(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")/curobo/curobolib/cpp/
```

> The `--dataset` flag downloads robot assets, BEHAVIOR-1K scene/object assets, and 2025 challenge task instances. You will be prompted to accept the NVIDIA Isaac Sim EULA and BEHAVIOR dataset license. To auto-accept, add `--accept-dataset-tos`.

For headless servers:
```bash
sudo apt-get update && sudo apt-get install -y libegl1 libgl1
# Remove duplicate Vulkan ICD if present (causes segfault on multi-GPU systems)
sudo rm -f /usr/share/vulkan/icd.d/nvidia_icd.json
```

See [docs/behavior-tasks.md](docs/behavior-tasks.md) for task details and expected baselines.

### Optional extras

```bash
uv sync --extra verl             # RL training with VeRL/GRPO
uv sync --extra contactgraspnet  # Contact-GraspNet grasp planning
uv sync --extra curobo           # cuRobo GPU-accelerated IK & motion planning (requires CUDA)
```

## Quick Start

### 1. Perception servers (auto-launched)

Perception servers (SAM3, ContactGraspNet, PyRoKi) are **auto-launched** by the YAML config when you run an evaluation. No manual setup required for most configs.

> **SAM3 authentication:** SAM3 weights require HuggingFace access. Request access at the [SAM3 repo](https://github.com/facebookresearch/sam3), then authenticate locally with `huggingface-cli login`. Weights are cached after first download.

To pre-launch servers (e.g. for sharing across multiple eval runs):

```bash
# Start SAM3 + GraspNet + PyRoKi with automatic GPU allocation
uv run --no-sync --active capx/serving/launch_servers.py --profile default
```

Use `--dry-run` to preview the allocation. Other profiles:

```bash
--profile full      # All perception servers (SAM3, GraspNet, PyRoKi, OWL-ViT, SAM2)
--profile minimal   # PyRoKi only (for oracle/privileged evals)
```

### 2. Set up an LLM proxy

The evaluation harness queries an LLM through a local proxy that exposes an OpenAI-compatible API.

```bash
# OpenRouter (get a key at openrouter.ai/keys)
echo "sk-or-v1-your-key-here" > .openrouterkey
uv run --no-sync --active capx/serving/openrouter_server.py --key-file .openrouterkey --port 8110
```

> **Note:** `.openrouterkey` are git-ignored. The default server URL in configs is `http://127.0.0.1:8110/chat/completions`. 

See [docs/configuration.md](docs/configuration.md) for all provider options (OpenRouter, vLLM, custom).

### 3. Run evaluation

```bash
# Robosuite: single-turn benchmark (100 trials, 12 parallel workers)
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
    --model "google/gemini-3.1-pro-preview"

# Robosuite: multi-turn with visual differencing
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm.yaml \
    --model "google/gemini-3.1-pro-preview"

# LIBERO-PRO: spatial task (requires .venv-libero)
source .venv-libero/bin/activate
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/libero/franka_libero_spatial_0.yaml \
    --model "google/gemini-3.1-pro-preview"

# BEHAVIOR: R1Pro radio pickup (20 trials) — requires b1k venv
source capx/third_party/b1k/.venv/bin/activate
OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1 \
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/r1pro/r1pro_pick_up_radio.yaml \
    --model "google/gemini-3.1-pro-preview"

# Interactive Web UI
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
    --web-ui True
# Open http://localhost:8200

# Regression tests
./scripts/regression_test.sh quick    # 10-trial smoke test (~30s)
./scripts/regression_test.sh test1    # Full single-turn (~3 min)
```

> **Tip (BEHAVIOR):** Isaac Sim uses `OMNIGIBSON_GPU_ID` (not `CUDA_VISIBLE_DEVICES`) to select the GPU. For best performance on multi-GPU systems, run perception servers on a separate GPU (e.g. `OMNIGIBSON_GPU_ID=0` for the eval, and pre-launch SAM3/GraspNet with `CUDA_VISIBLE_DEVICES=1`). Set `OMNI_KIT_ACCEPT_EULA=YES` and `OMNIGIBSON_HEADLESS=1` for headless evaluation.

---

## Documentation

| Guide | Contents |
| ----- | -------- |
| [Adding Environments](docs/adding-environments.md) | Creating simulators, task environments, YAML configs |
| [Adding APIs](docs/adding-apis.md) | Implementing and registering new robot control APIs |
| [Configuration](docs/configuration.md) | YAML format, CLI flags, LLM provider setup |
| [LIBERO-PRO Tasks](docs/libero-tasks.md) | Setup, running any of 130+ LIBERO tasks, suite reference |
| [BEHAVIOR Tasks](docs/behavior-tasks.md) | Setup, R1Pro tasks, expected baselines, environment variables |
| [Development](docs/development.md) | Testing, linting, LIBERO/GraspNet setup, checkpoints, known issues |
| [Real-World Franka Panda Bringup](docs/real-franka.md) | Bringup with robots_realtime, real-robot QuickStart |
| [RL Training](docs/rl-training.md) | CaP-RL with GRPO/VeRL, sim-to-real transfer |
| [Skill Library Compilation](scripts/skill_library_compilation/README.md) | Analyze eval outputs, compile reusable skill libraries |

---

## Citation

```bibtex
@article{fu2025capx,
  title     = {{CaP-X}: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation},
  author    = {Fu, Max and Yu, Justin and El-Refai, Karim and Kou, Ethan and Xue, Haoru and Huang, Huang and Xiao, Wenli and Wang, Guanzhi and Li, Fei-Fei and Shi, Guanya and Wu, Jiajun and Sastry, Shankar and Zhu, Yuke and Goldberg, Ken and Fan, Jim},
  journal   = {arXiv preprint arXiv:2603.22435},
  year      = {2025},
  url       = {https://arxiv.org/abs/2603.22435}
}
```

## License

This project is released under the [MIT License](LICENSE).
