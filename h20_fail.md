# H20 workspace — RoboTwin/SAPIEN Vulkan 旧失败复盘 + RLinf 成功修正

> 目的:记录在 coder-workspace(H20)容器上旧 `conda RoboTwin` 环境里遇到的
> RoboTwin/SAPIEN Vulkan 失败,并保留一套轻量复核流程。
>
> **2026-06-20 修正:**同一个 H20 Docker 里,RLinf 的 `.venv` 已经能跑通
> RoboTwin rollout。因此,下面旧结论中“这台 H20/固件层禁用 Vulkan、容器内修不了”
> 已经不成立。现在更合理的判断是:**H20 卡和当前 Docker 可以跑 RoboTwin;此前失败更可能来自
> 被删掉的 `conda RoboTwin` 环境、Vulkan ICD/driver capabilities 环境变量、或 SAPIEN/mplib/CUDA
> 依赖栈差异。**

---

## 0. 当前统一结论

截至 2026-06-20,旧 H20/Vulkan 结论和当前 Hy-VLA 测试要合并成下面这个判断:

- **不是 H20 固件层禁用图形/Vulkan。** 同一个 Docker 中,RLinf `.venv` 已经能跑 RoboTwin rollout;
  设置 NVIDIA ICD 后也能看到 `Render Well`。关键环境变量是
  `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`,
  `VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json`,
  `NVIDIA_DRIVER_CAPABILITIES=all`。
- **Hy-VLA 模型链路已经走通。** 当前 `RoboTwin_hy` 测试能加载
  `/home/jovyan/code/wge/ttt_drift/ckpts/Hy-VLA-RoboTwin`,并进入 policy;
  日志里已经看到 `[Hy-VLA] action summary: shape=(16,) ... finite=True`,说明模型能产出有限 action。
- **原主 blocker 是 RoboTwin 的 EE action cuRobo 规划,现在已有稳定 workaround。** 失败点曾从
  expert precheck/warmup 被逐步隔离到 `TASK_ENV.take_action(action, action_type="ee") -> curobo`,
  典型报错是 `RuntimeError: CUDA driver error: an illegal instruction was encountered`。当前通过
  `RoboTwinHy26` + torch2.6/cu124 + warp1.11 + RoboTwin source cuRobo + 关闭 cuRobo graph/LBFGS fused
  kernel 已经跑通。
- **当前 `RoboTwinHy` 栈与 RLinf 成功栈不同。**
  `RoboTwinHy` 是 `torch 2.4.1+cu121 / warp 1.12.0 / RoboTwin_hy source curobo`;
  RLinf 是 `torch 2.6.0+cu124 / warp 1.11.1 / site-packages nvidia-curobo@a35a708...`。
  因此后续更像是 cuRobo/Torch/Warp 栈兼容性问题,不是 SAPIEN 渲染或 Hy checkpoint 问题。
- **当前剩余问题是速度,不是可运行性。** 单 rollout 约 7-9 分钟,适合 smoke/regression,不适合直接串行
  全量 benchmark;后续应优化 cuRobo workaround、减少 rollout 数或并行化。

调试中遇到的 `No valid instructions found` 已通过 task name fallback patch 规避;`ffmpeg` 缺失是录像依赖,
不是主链路 blocker。

### 0.1 已验证跑通结果与速度

2026-06-20,`RoboTwinHy26` 环境在 H20 上已经完成 `adjust_bottle` 的 Hy-VLA eval:

```text
环境: /home/jovyan/miniconda3/envs/RoboTwinHy26
代码: /home/jovyan/code/wge/ttt_drift @ f03505d
RoboTwin: /home/jovyan/code/wge/RoboTwin_hy
torch: 2.6.0+cu124
warp: 1.11.1
curobo: /home/jovyan/code/wge/RoboTwin_hy/envs/curobo/src/curobo
结果: Success rate: 1/1 => 100.0%
```

稳定跑通使用的关键开关:

```bash
HYVLA_REQUIRE_SOURCE_CUROBO=1
HYVLA_PATCH_ROBOTWIN_TRACEBACK=1
HYVLA_PATCH_CUROBO_NO_GRAPH=1
HYVLA_PATCH_CUROBO_DISABLE_LBFGS_KERNEL=1
```

速度记录:

- 默认 expert precheck 开启:`TEST_NUM=1` 单 rollout 约 **9m13s**。
- `HYVLA_PATCH_SKIP_EXPERT_CHECK=1` 后:单 rollout 约 **7m46s**。
- 跳过 expert 只省约 **1m27s**(约 16%),说明主要耗时不在 expert precheck,而在真实
  Hy-VLA rollout + `action_type="ee"` 的 cuRobo 规划/执行。
- 当前速度适合 smoke/regression,不适合全量 benchmark。慢的主要代价来自稳定性 workaround:
  `cuRobo graph disabled` 和 `LBFGS CUDA kernel disabled`。

外部速度调研结论:

- RoboTwin 2.0 官方论文/文档公开的是评测规模和成功率,没有找到可靠的单 rollout wall-clock baseline。
  官方论文说明 benchmark 使用 Aloha-AgileX,每个 policy 在 50 个任务上评测,每个任务在 Easy/Hard
  两种条件下各 100 rollouts[^robotwin-paper-eval]。
- Hugging Face/LeRobot 的 RoboTwin 2.0 文档只给出安装耗时约 20 分钟,没有给 eval wall-clock
  指标[^hf-robotwin-install]。
- X-VLA 的 RoboTwin-2.0 eval 采用 server/client 方式跑模型和仿真,并说明 episode 数等可配置,
  但也没有公开每个 rollout 的耗时[^xvla-eval]。
- 因此不能把我们这次 7m46s/rollout 直接和公开榜单速度做一一对比。只能判断:如果按官方全量
  50 tasks × 100 rollouts × Easy/Hard 串行跑,当前速度量级会非常慢;后续若要大规模评测,必须做
  速度优化、减少 rollout 数或多 GPU/多进程并行。

[^robotwin-paper-eval]: RoboTwin 2.0 paper, Sec. 4.5: https://arxiv.org/html/2506.18088v1
[^hf-robotwin-install]: Hugging Face LeRobot RoboTwin 2.0 docs: https://huggingface.co/docs/lerobot/main/robotwin
[^xvla-eval]: X-VLA RoboTwin-2.0 evaluation README: https://github.com/2toinf/X-VLA/blob/main/evaluation/robotwin-2.0/README.md

---

## 1. 环境(H20 那台)

- host: `coder-workspace-brown-panther-85`,user `jovyan`
- 代码:`/home/jovyan/code/wge/ttt_drift`
- 旧 RoboTwin 路径:`/home/jovyan/code/wge/RoboTwin`;当前 Hy 测试路径:`/home/jovyan/code/wge/RoboTwin_hy`
- 已知可跑通的环境:`/home/jovyan/code/wge/RLinf/.venv`
- 离线实验(step0/step1)环境:uv,Python 3.11
- 旧失败环境:conda env `RoboTwin`,Python 3.10,torch 2.4.1+cu121(已删除)
- GPU:8× NVIDIA **H20Z**,driver **570.124.06**(NVIDIA 开源内核模块),虚拟化模式 **Pass-Through**

---

## 2. 装通 RoboTwin 的过程(这些都成功了,供参考)

按顺序解决的坑:

1. RoboTwin 安装(官方):
```
conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin
cd /home/jovyan/code/wge/RoboTwin
bash script/_install.sh
python script/update_embodiment_config_path.py
bash script/_download_assets.sh
```
`_install.sh` 装 `script/requirements.txt`(含 sapien 3.0.0b1、mplib)、pytorch3d、curobo。

2. curobo/pytorch3d 编译报 `CUDA_HOME not set` → 装匹配 torch(cu121)的 CUDA toolkit:
```
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
```

3. 编译报 `unsupported GNU version! gcc later than 12 not supported`(系统 gcc 13)→ 装 gcc-12 并指定为 nvcc host 编译器:
```
sudo apt install -y gcc-12 g++-12
export CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-12"
export TORCH_CUDA_ARCH_LIST="9.0"
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
pip install -e envs/curobo --no-build-isolation
```

4. 把 Hy-VLA 推理依赖装进 RoboTwin env(transformers 走 PyPI,因为 git commit clone 报 `gnutls_handshake failed`;flash-attn 用预编译 wheel,版本三要素 torch2.4/cp310/abiFALSE):
```
pip install "transformers==4.57.*" safetensors "huggingface-hub>=0.23" timm==1.0.21
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

5. eval 报 `libGL.so.1: cannot open shared object file` → 装系统库(apt 索引过期,先 update):
```
sudo apt-get update
sudo apt install -y libgl1 libglib2.0-0 libgomp1
```

6. 至此 RoboTwin 能跑到加载策略,但每个任务输出只有 `Render Error`。

---

## 3. 旧失败观测:Vulkan 建不出实例

绕开 RoboTwin 直接测 SAPIEN 渲染,拿到底层错误:

```
RuntimeError: failed to find a rendering device
```
`vulkaninfo` 报:
```
loader_scanned_icd_add: Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0
vkCreateInstance: Found no drivers!
ERROR_INCOMPATIBLE_DRIVER
```

---

## 4. 旧诊断:当时排除过的常见原因

| 怀疑点 | 检查命令 | H20 上的结果 | 是否元凶 |
|---|---|---|---|
| 内核与用户态驱动版本错配 | `cat /proc/driver/nvidia/version` vs lib 版本 | 都是 570.124.06,一致 | 否 |
| 设备节点缺失 | `ls -l /dev/nvidia* /dev/dri/` | `/dev/nvidia0-7`、`nvidiactl`、`nvidia-modeset`、`/dev/dri/renderD128-135` 全在 | 否 |
| 容器没开 graphics 能力 | `echo $NVIDIA_DRIVER_CAPABILITIES` | `all` | 否 |
| 缺 Vulkan 后端库 | `ldd libGLX_nvidia.so.0 \| grep "not found"`;`ls libnvidia-glvkspirv* glcore* rtcore*` | 无 not found,库齐全 | 否 |
| libGLX_nvidia 不是有效 ICD | `nm -D libGLX_nvidia.so.0 \| grep vk_icd` | 导出 `vk_icdGetInstanceProcAddr` 等,是有效 ICD | 否 |
| vGPU/compute-only 配置 | `nvidia-smi -q \| grep -i Virtualization` | **Pass-Through**(非 vGPU) | 否 |
| 我们手写的 ICD json 错 | 用驱动自带 `/etc/vulkan/icd.d/nvidia_icd.json`(api 1.4.303)重试 | **同样报错** | 否 |
| implicit layer 干扰 | `VK_LOADER_LAYERS_DISABLE='*'` 重试 | 仍报错(那是无关的 `VK_LAYER_NV_optimus`) | 否 |
| 软件 Vulkan 兜底 | `VK_ICD_FILENAMES=.../lvp_icd.json`(lavapipe)跑 SAPIEN | 能建实例,但 SAPIEN 需 Vulkan-CUDA 互操作扩展 → `ErrorExtensionNotPresent` | 不可用 |

**旧结论已降级为历史假设。** RLinf `.venv` 在同一个 Docker 中跑通 RoboTwin 后,这些观测只能说明
当时那套 `conda RoboTwin` 运行路径没有正确初始化 NVIDIA Vulkan,不能再推出 H20 固件/平台层不支持
RoboTwin。

---

## 5. 更新后的轻量确认流程

目标不是重新做完整 benchmark,而是确认“这张 H20 + 这个 Docker 能不能跑 RoboTwin 渲染”,然后在
**独立 conda env** 里跑 Hy。RLinf `.venv` 只作为已知可用的只读对照,不要往里面装 Hy 依赖。

### 5.1 新建 Hy/RoboTwin 专用 conda env
```
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0 libgomp1 libvulkan1 mesa-vulkan-drivers vulkan-tools gcc-12 g++-12

conda create -n RoboTwinHy python=3.10 -y
conda activate RoboTwinHy

conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-12"
export TORCH_CUDA_ARCH_LIST="9.0"
export FORCE_CUDA=1
export MAX_JOBS=4
export TMPDIR=/home/jovyan/tmp
mkdir -p "${TMPDIR}"

export ROBOTWIN_DIR=/home/jovyan/code/wge/RoboTwin_hy
export CKPT_PATH=/home/jovyan/code/wge/ttt_drift/ckpts/Hy-VLA-RoboTwin
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
export NVIDIA_DRIVER_CAPABILITIES=all
export XDG_RUNTIME_DIR=/tmp/xdg-jovyan
mkdir -p "${XDG_RUNTIME_DIR}"
```

### 5.2 安装 RoboTwin_hy + Hy adapter
```
cd "${ROBOTWIN_DIR}"
bash script/_install.sh
python script/update_embodiment_config_path.py
# assets 已经下载过可跳过;不确定就跑一遍。
bash script/_download_assets.sh
```

如果 `_download_assets.sh` 中途因为 HuggingFace SSL/网络断开失败,不要重跑完整 `_install.sh`。先只补缺的
zip,再用 `unzip -n` 跳过已存在文件,避免交互式 `replace ... [y/n/A/N]` 卡住:

```
conda activate RoboTwinHy
cd /home/jovyan/code/wge/RoboTwin_hy

# 例:日志里缺的是 embodiments.zip。
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
huggingface-cli download TianxingChen/RoboTwin2.0 embodiments.zip \
  --repo-type dataset \
  --local-dir /home/jovyan/code/wge/RoboTwin_hy/assets \
  --resume-download

cd /home/jovyan/code/wge/RoboTwin_hy/assets
unzip -n embodiments.zip

cd /home/jovyan/code/wge/RoboTwin_hy
python script/update_embodiment_config_path.py
```

PyTorch3D 可能从源码编译。`Building wheel for pytorch3d` 长时间无输出时,先看是否有
`cc1plus`/`nvcc` 占 CPU;有就不是卡死:

```
ps -ef | egrep 'nvcc|cc1plus|c\+\+|ninja|pytorch3d' | grep -v grep
top -u jovyan
```

如果想先绕过源码编译,可试官方 wheel 入口;命中则很快,不命中会报 `No matching distribution`:

```
pip uninstall -y pytorch3d
pip install iopath
pip install --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/download.html
```

把 Hy-VLA 装进同一个 conda env。注意 `--no-deps`:不要让 pyproject 升级/替换 RoboTwin 刚装好的
torch、SAPIEN、mplib、curobo。
```
cd /home/jovyan/code/wge/ttt_drift
pip install -e . --no-deps

pip install "git+https://github.com/huggingface/transformers@9293856c419762ebf98fbe2bd9440f9ce7069f1a" \
    safetensors "huggingface-hub>=0.23" timm==1.0.21 scipy

# 如果 git clone transformers 因网络失败,用 PyPI 版本 + 本仓库 vendor fallback:
# pip install -U "transformers>=4.57,<4.58" safetensors "huggingface-hub>=0.23" timm==1.0.21 scipy

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

安装 Hy 依赖时 pip 可能提示 `hy-vla requires torch>=2.7`、`accelerate/deepspeed/... is not installed`。
这是因为本仓库 `pyproject.toml` 记录的是完整训练栈,而 RoboTwin eval 需要保留 RoboTwin 的
`torch==2.4.1`/SAPIEN/mplib/curobo 栈。这里不要为消除这些 resolver warning 去升级 torch 或补训练依赖。
真正必须通过的是下面 preflight 里的 `flash_attn = True` 和 `hy_vla import OK`。

### 5.3 环境记录 + import preflight
```
which python
python - <<'PY'
import importlib.util, os, sys, torch, transformers
print("python", sys.executable)
print("torch", torch.__version__, torch.version.cuda)
print("transformers", transformers.__version__, transformers.__file__)
for name in ["CONDA_PREFIX", "VIRTUAL_ENV", "CUDA_HOME", "VK_ICD_FILENAMES",
             "VK_DRIVER_FILES", "NVIDIA_DRIVER_CAPABILITIES"]:
    print(name, "=", os.environ.get(name))
for name in ["transformers.modeling_layers", "timm", "flash_attn"]:
    print(name, "=", importlib.util.find_spec(name) is not None)
import sapien
print("sapien", sapien.__file__)
import hy_vla
print("hy_vla import OK")
PY
```

### 5.4 Vulkan/SAPIEN smoke
`vulkaninfo` 能列出 NVIDIA H20Z 就说明 ICD 选择已经对。裸 SAPIEN smoke 只做 90 秒限时检查;
如果它超时,但后面的 RoboTwin eval 打印 `Render Well`,以 RoboTwin 的真实路径为准。

```
vulkaninfo --summary 2>&1 | sed -n '1,80p'

timeout 90s python - <<'PY'
import sapien
sapien.set_log_level("info")
s = sapien.Scene(); s.add_ground(-1); s.set_ambient_light([0.5,0.5,0.5])
c = s.add_camera("c",128,128,1.0,0.01,100); s.update_render(); c.take_picture()
print("RENDER OK", c.get_picture("Color").shape)
PY
```

预期:`vulkaninfo` 能看到 NVIDIA ICD/device。若 Python 输出 `RENDER OK ...`,裸 SAPIEN 也通过;
若只在 RoboTwin eval 中看到 `Render Well`,也足以说明真实 eval 渲染链路通过。

### 5.5 可选:Hy 原始测试的最小 rollout
```
TASKS_OVERRIDE=adjust_bottle \
TEST_NUM=1 \
ROBOTWIN_DIR=/home/jovyan/code/wge/RoboTwin_hy \
CKPT_PATH=/home/jovyan/code/wge/ttt_drift/ckpts/Hy-VLA-RoboTwin \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/eval_robotwin_test.sh
```

如果日志在 `TASK_ENV.setup_demo -> CuroboPlanner -> motion_gen.warmup()` 阶段报
`RuntimeError: CUDA error: an illegal instruction was encountered`,这还没进入 Hy action 推理,
是 cuRobo planner warmup 的 CUDA/TorchScript kernel 问题。先用下面的调试开关跳过 cuRobo warmup
后重试:

```
HYVLA_PATCH_ROBOTWIN_TRACEBACK=1 \
HYVLA_PATCH_CUROBO_NO_GRAPH=1 \
HYVLA_PATCH_CUROBO_SKIP_WARMUP=1 \
TASKS_OVERRIDE=adjust_bottle \
TEST_NUM=1 \
ROBOTWIN_DIR=/home/jovyan/code/wge/RoboTwin_hy \
CKPT_PATH=/home/jovyan/code/wge/ttt_drift/ckpts/Hy-VLA-RoboTwin \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/eval_robotwin_test.sh
```

该 patch 会备份并修改 `/home/jovyan/code/wge/RoboTwin_hy/envs/robot/planner.py`,备份文件名为
`planner.py.hyvla_no_graph.bak`。

如果跳过 warmup 后又在 `TASK_ENV.play_once()` 的专家轨迹规划里报同类 cuRobo/CUDA illegal instruction,
说明 RoboTwin 的专家 seed/instruction 预检查仍在走 cuRobo,还没进入 Hy policy。用下面的开关跳过
专家预检查,直接进入 policy rollout,并用 task name 作为 fallback instruction:

```
HYVLA_PATCH_ROBOTWIN_TRACEBACK=1 \
HYVLA_PATCH_SKIP_EXPERT_CHECK=1 \
HYVLA_PATCH_CUROBO_NO_GRAPH=1 \
HYVLA_PATCH_CUROBO_SKIP_WARMUP=1 \
TASKS_OVERRIDE=adjust_bottle \
TEST_NUM=1 \
ROBOTWIN_DIR=/home/jovyan/code/wge/RoboTwin_hy \
CKPT_PATH=/home/jovyan/code/wge/ttt_drift/ckpts/Hy-VLA-RoboTwin \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/eval_robotwin_test.sh
```

如果日志出现 `Episode 0: No valid instructions found` / `IndexError: list index out of range`,说明旧 patch
只跳过了 expert check,但没有替换 instruction 生成行;重新 `git pull` 后用同一条命令再跑即可,新的
`HYVLA_PATCH_SKIP_EXPERT_CHECK=1` 会把空 instruction list 回退成 task name。

判读:
- 如果 5.4 通过,但 5.5 因 Hy-VLA import/权重/transformers 失败,说明 **RoboTwin/SAPIEN 环境可用**,
  需要修 Hy-VLA 依赖或 checkpoint 路径。
- 如果 5.4 在新 conda env 里失败,先和 RLinf `.venv` 的成功环境变量对照,再回到 Vulkan loader/ICD 排查。

### 5.6 只有 smoke 失败时才需要的 Vulkan 诊断
```
nvidia-smi -L
cat /proc/driver/nvidia/version
echo "caps=$NVIDIA_DRIVER_CAPABILITIES"
ls -l /dev/nvidia* /dev/dri/
ldd /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 | grep -i "not found"
nm -D /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 | grep -i vk_icd
ls -l /usr/lib/x86_64-linux-gnu/libnvidia-glvkspirv.so* /usr/lib/x86_64-linux-gnu/libnvidia-glcore.so* /usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so*
cat /etc/vulkan/icd.d/nvidia_icd.json
nvidia-smi -q | grep -iE "Virtualization|vGPU|MIG|Compute Mode|GSP"
```

### 5.7 还想往根因挖,可继续试的方向
- **完整 loader 跟踪**(看协商在哪断、是否 dlopen 了次级库失败):
```
VK_LOADER_DEBUG=all VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json vulkaninfo 2>&1 | sed -n '1,120p'
```
- **看 GPU 是否报告图形能力 / 有无错误**:`nvidia-smi -q | grep -iE "Graphics|ECC|Fabric|Persistence"`;`dmesg 2>/dev/null | grep -i nvidia | tail -40`(可能无权限)。
- **换一块 GPU 试**:`CUDA_VISIBLE_DEVICES=1 vulkaninfo --summary`(各卡逐一)。
- **更新 Vulkan loader**:系统 `libvulkan1` 是 1.3.275,可试装更新的 loader 再协商。
- **最终判定手段**:RLinf `.venv` 只作为只读成功对照;若新 conda env 不行,优先比较环境变量、
  SAPIEN/mplib/CUDA 版本和 NVIDIA ICD 选择。

### 5.8 给平台方的诉求(仅当新 conda + RLinf 对照都失败时)
旧版诉求不应再直接发送。只有在新 conda env 和 RLinf `.venv` 对照都稳定失败时,再整理新的
`vulkaninfo` 和 SAPIEN smoke 日志给平台方。

---

## 6. 当前判断

另一台卡上 Hy 原始测试已跑通;同一 H20 Docker 中 RLinf RoboTwin rollout 也已跑通。2026-06-20 对照结果:

```
RoboTwinHy:
  python  /home/jovyan/miniconda3/envs/RoboTwinHy/bin/python
  torch   2.4.1+cu121, torch.version.cuda 12.1
  warp    1.12.0
  curobo  /home/jovyan/code/wge/RoboTwin_hy/envs/curobo/src/curobo

RLinf .venv:
  python  /home/jovyan/code/wge/RLinf/.venv/bin/python
  torch   2.6.0+cu124, torch.version.cuda 12.4
  warp    1.11.1
  curobo  /home/jovyan/code/wge/RLinf/.venv/lib/python3.11/site-packages/curobo
```

综合判断:

- **H20 这张卡/这个 Docker 可以跑 RoboTwin。**
- 旧 `h20_fail.md` 的固件级定性已经过期。
- Hy-VLA policy 本身已能 load 并产出 finite `(16,)` action。
- 原 blocker 是 `RoboTwinHy` 的 cuRobo 执行 EE action 时在 H20 上触发 CUDA illegal instruction;
  当前已通过 `RoboTwinHy26` + torch2.6/cu124 + warp1.11 + RoboTwin source cuRobo + no-graph/LBFGS
  workaround 跑通。
- 不要继续 patch torch2.4/cu121/warp1.12/source-curobo 这条旧环境;下一步应新建独立环境,复制 RLinf 已验证的
  torch2.6/cu124/warp1.11 方向,但 cuRobo 仍需使用 RoboTwin_hy 旧 API source 版本。site-packages
  `nvidia-curobo` v0.8/v2 会和 RoboTwin_hy planner 的 `curobo.types.math` 旧导入不兼容。
- 当前剩余问题是速度:单 rollout 约 7-9 分钟,跳过 expert precheck 只节省约 1m27s,说明主要耗时在
  真实 Hy-VLA rollout + EE cuRobo 规划。

已补一个保守入口:`scripts/setup_robotwin_hy26_stack.sh`。它默认 clone
`RoboTwinHy -> RoboTwinHy26`,替换 torch/warp/SAPIEN/mplib,重装 RoboTwin source cuRobo 并做 import
路径自检;后续 eval 可加 `HYVLA_REQUIRE_SOURCE_CUROBO=1`,强制确认 `curobo` 仍来自
`RoboTwin_hy/envs/curobo/src`。
