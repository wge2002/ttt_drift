# 服务器上跑 Hy-VLA —— 指令清单

> 本仓库 = `wge2002/ttt_drift`,根目录 vendor 了 `Hy-Embodied-0.5-VLA` 源码 + `drifting_code/`(He drift 参考)。
> 环境要求(README):Linux / Python 3.12 / CUDA 12.x / PyTorch ≥2.4 / GPU ≥16GB VRAM。
>
> ⚠️ 已知两个上游坑(详见本仓库 README 与脚本注释):
> 1. `scripts/quick_start.py` 里的 ckpt id `tencent/Hy-VLA-RoboTwin` 与真实 HF repo `tencent/Hy-Embodied-0.5-VLA-RoboTwin` 不一致,下面用真实的。
> 2. `scripts/eval_robotwin_test.sh` 在循环里误用 `local`,会被 `set -e` 直接打挂,下面给了修复。

---

## 0. clone

```bash
git clone git@github.com:wge2002/ttt_drift.git
cd ttt_drift
```

## 1. 环境 + 安装

推荐 uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # 一次性
uv sync                                           # 按 pyproject/uv.lock 建虚拟环境
source .venv/bin/activate
```
或 pip:
```bash
conda create -n hyvla python=3.12 -y && conda activate hyvla
pip install -r requirements.txt
```
> 注:依赖一个 transformers fork(pin 在 requirements/pyproject)。若该 fork URL 拉不动,仓库里 `hy_vla/hunyuan_vl_mot/` 是 verbatim 兜底副本。

## 2. 下载权重(不在 git repo 里,需单独拉)

国内建议用 HF 镜像或 ModelScope。
```bash
# HF(镜像加速)
export HF_ENDPOINT=https://hf-mirror.com
pip install -U "huggingface_hub[cli]"
huggingface-cli download tencent/Hy-Embodied-0.5-VLA-RoboTwin \
    --local-dir ./ckpts/Hy-VLA-RoboTwin

# 或 ModelScope
pip install -U modelscope
modelscope download --model Tencent-Hunyuan/Hy-Embodied-0.5-VLA-RoboTwin \
    --local_dir ./ckpts/Hy-VLA-RoboTwin
```
checkpoint 自带 `tokenizer.json` / `vlm_config_dict` / `chat_template.jinja` / `norm_stats.pkl`,开箱即用。

## 3. 冒烟测试(不需要 RoboTwin 模拟器,只跑一次前向)

用真实 ckpt id 跑(绕开 quick_start.py 的 id bug):
```bash
python - <<'PY'
import torch
from hy_vla import HyVLA, HyVLAConfig
ckpt = "./ckpts/Hy-VLA-RoboTwin"          # 用本地已下好的目录
config = HyVLAConfig.from_pretrained(ckpt)
policy = HyVLA.from_pretrained(ckpt, config=config)
policy.enable_video_encoder_if_needed()
policy = policy.to(device="cuda", dtype=torch.bfloat16).eval()
img   = torch.zeros(1, 6, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
state = torch.zeros((1, config.max_state_dim), device="cuda", dtype=torch.bfloat16)
batch = {
    "observation.images.top_head":   img,
    "observation.images.hand_left":  img,
    "observation.images.hand_right": img,
    "observation.state": state,
    "task": ["pick up the bottle"],
}
with torch.no_grad():
    a = policy.forward_evaluate(batch)["pred"][..., :config.action_feature.shape[0]]
print("OK, action shape =", a.shape)
PY
```
能打印出 action shape = 环境 + 权重链路通了。

## 4. step 0 —— mask 有效性诊断(idea 的地基,见 idea md §6)

不需要 RoboTwin 模拟器,一张卡 + 权重即可:
```bash
python scripts/diag_step0_mask.py --ckpt ./ckpts/Hy-VLA-RoboTwin --out step0_mask_diag.jsonl
```
脚本做什么:用**同一份 noise**跑「vision 正常」与「全 mask」两支,记录每步 `‖v_t‖` 与最终 action chunk,核心看**换不同指令时 masked 动作是否随之变化**(=是否存在可 drift 的 language 先验)。stdout 直接给判读,完整数据写进 jsonl。
把 `step0_mask_diag.jsonl` 下载发我,我做相关性 / 流形分析。

## 5. 完整 RoboTwin eval(需要单独装 RoboTwin 2.0 模拟器)

`robotwin_eval/` 只是适配器;真正跑要有 RoboTwin 2.0 仓库 + 其仿真依赖(SAPIEN 等)。安装见官方 [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin):

### 5.0 H20 当前推荐路径:新建独立 conda env + `RoboTwin_hy`

2026-06-20 更新:同一个 H20 Docker 中,RLinf 的 `.venv` 已经能跑通 RoboTwin rollout,所以 H20/Vulkan
不是根因。但 **不要污染 RLinf `.venv`**。当前 Hy 测试应新建一个独立 conda 环境,只借用 RLinf
成功经验里的 NVIDIA ICD 环境变量。

```bash
# 系统库。如果已装过,apt 会直接跳过。
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0 libgomp1 libvulkan1 mesa-vulkan-drivers vulkan-tools gcc-12 g++-12

# 新环境:不要用 RLinf .venv。
conda create -n RoboTwinHy python=3.10 -y
conda activate RoboTwinHy

# 给 curobo/pytorch3d 编译用。若 script/_install.sh 已经装好,这步也不会污染其他环境。
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

安装/修复 RoboTwin 仿真依赖:

```bash
cd "${ROBOTWIN_DIR}"
bash script/_install.sh
python script/update_embodiment_config_path.py
# 如果 assets 已经下载过可跳过;不确定就跑一遍。
bash script/_download_assets.sh
```

如果 `_download_assets.sh` 中途因为 HuggingFace SSL/网络断开失败,不要重跑完整 `_install.sh`。先只补缺的
zip,再用 `unzip -n` 跳过已存在文件,避免交互式 `replace ... [y/n/A/N]` 卡住:

```bash
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

`_install.sh` 里的 PyTorch3D 可能会从源码编译。看到 `Building wheel for pytorch3d` 长时间无输出时,
先在另一个 shell 看是否真在编译:

```bash
ps -ef | egrep 'nvcc|cc1plus|c\+\+|ninja|pytorch3d' | grep -v grep
top -u jovyan
```

如果有 `cc1plus`/`nvcc` 占 CPU,继续等即可。如果想先绕过源码编译,可试官方 wheel 入口;命中则很快,
不命中会报 `No matching distribution`:

```bash
pip uninstall -y pytorch3d
pip install iopath
pip install --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/download.html
```

把 Hy-VLA adapter 装进同一个 conda env。这里 **必须用 `--no-deps`**,避免 pyproject 把
RoboTwin 刚装好的 torch/SAPIEN/mplib/curobo 栈升级乱掉:

```bash
cd /home/jovyan/code/wge/ttt_drift
pip install -e . --no-deps

# Hy 推理最小依赖。优先装 pinned transformers fork;网络失败时再用 PyPI 4.57 + 仓库 vendor fallback。
pip install "git+https://github.com/huggingface/transformers@9293856c419762ebf98fbe2bd9440f9ce7069f1a" \
    safetensors "huggingface-hub>=0.23" timm==1.0.21 scipy

# 如果 git clone transformers 因网络失败,用这条替代:
# pip install -U "transformers>=4.57,<4.58" safetensors "huggingface-hub>=0.23" timm==1.0.21 scipy

# Python 3.10 + torch2.4/cu12 ABI=false 对应旧成功环境里的 flash-attn wheel。
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

检查环境。这里会确认你没有进 RLinf `.venv`,并检查 Hy 关键模块:

```bash
which python
python - <<'PY'
import importlib.util, os, sys, torch, transformers
print("python", sys.executable)
print("torch", torch.__version__, torch.version.cuda)
print("transformers", transformers.__version__, transformers.__file__)
for name in ["CONDA_PREFIX", "VIRTUAL_ENV", "CUDA_HOME", "VK_ICD_FILENAMES", "VK_DRIVER_FILES", "NVIDIA_DRIVER_CAPABILITIES"]:
    print(name, "=", os.environ.get(name))
for name in ["transformers.modeling_layers", "timm", "flash_attn"]:
    print(name, "=", importlib.util.find_spec(name) is not None)
import sapien
print("sapien", sapien.__file__)
import hy_vla
print("hy_vla import OK")
PY

vulkaninfo --summary 2>&1 | sed -n '1,80p'

timeout 90s python - <<'PY'
import sapien
sapien.set_log_level("info")
s = sapien.Scene(); s.add_ground(-1); s.set_ambient_light([0.5, 0.5, 0.5])
c = s.add_camera("c", 128, 128, 1.0, 0.01, 100)
s.update_render(); c.take_picture()
print("RENDER OK", c.get_picture("Color").shape)
PY
```

如果这个裸 SAPIEN smoke 超时,但后面的 RoboTwin eval 打印了 `Render Well`,以 RoboTwin 的结果为准;
那条路径更接近真实任务。不要因此回去改 RLinf `.venv`。

确认 Hy 原始测试路径时,只跑 1 个 task × 1 rollout 即可:

```bash
TASKS_OVERRIDE=adjust_bottle \
TEST_NUM=1 \
ROBOTWIN_DIR=/home/jovyan/code/wge/RoboTwin_hy \
CKPT_PATH=/home/jovyan/code/wge/ttt_drift/ckpts/Hy-VLA-RoboTwin \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/eval_robotwin_test.sh
```

判读:如果 SAPIEN smoke 通过,这张 H20/这个 Docker 就能跑 RoboTwin;后续失败应优先看 Hy-VLA
依赖、checkpoint 路径或 adapter 参数,而不是再定性为 H20 固件/Vulkan 被禁。

### 5.1 从零安装 RoboTwin env(通用路径)

```bash
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools
conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin
git clone https://github.com/RoboTwin-Platform/RoboTwin.git
cd RoboTwin
bash script/_install.sh
python script/update_embodiment_config_path.py
bash script/_download_assets.sh
```

⚠️ eval 在 RoboTwin 这个 env(py3.10)里同进程跑仿真 + Hy-VLA,所以要把 Hy-VLA 装进它:
```bash
conda activate RoboTwin
pip install -e /home/jovyan/code/wge/ttt_drift
python -c "import hy_vla; print('hy_vla import OK')"
```

原版回归(`local` bug 已在仓库内修好,无需 sed;`eval_robotwin_test.sh` 会自动把本仓库 symlink 进 `RoboTwin/policy/hy_vla`):
```bash
ROBOTWIN_DIR=/home/jovyan/code/wge/RoboTwin_hy CKPT_PATH=$(pwd)/ckpts/Hy-VLA-RoboTwin CUDA_VISIBLE_DEVICES=0 TEST_NUM=10 bash scripts/eval_robotwin_test.sh
```
全量(50 任务 ×100 rollout,很慢):`bash scripts/eval_robotwin_full.sh`(同样 env 变量)。

## 5.5 step 1 真测 —— 扫 guidance_w 看 OOD 成功率

核心实验。`guidance_w<1` 把动作往语言先验拉(velocity blend);用环境变量 `HYVLA_GUIDANCE_W` 逐档驱动:
```bash
ROBOTWIN_DIR=/home/jovyan/code/wge/RoboTwin_hy CKPT_PATH=$(pwd)/ckpts/Hy-VLA-RoboTwin CUDA_VISIBLE_DEVICES=0 TEST_NUM=20 TASK_CONFIG=demo_randomized bash scripts/eval_sweep_w.sh
```
- `TASK_CONFIG=demo_randomized` = OOD(强域随机化);再跑一遍 `TASK_CONFIG=demo_clean` 作 ID 对照。
- 扫 `W_GRID="1.0 0.75 0.5 0.25"`(默认),日志落在 `eval_logs/<task_config>/w_<w>/`。
- 收集:`grep -r 'Success rate' eval_logs/`。
- 判据:OOD 下成功率随 `w`↓ 上升、而 ID 下不升/降 = 干净的 OOD 专属收益。

## 6. 把结果打包回传给 Claude

```bash
# 完整 stdout + 退出码都留痕
<你的命令> 2>&1 | tee run_$(date +%Y%m%d_%H%M%S).log
# eval 日志 + 诊断 jsonl 一起打包
tar czf results_$(date +%Y%m%d_%H%M%S).tar.gz eval_logs/ *.jsonl run_*.log 2>/dev/null
```
把这个 tar.gz(或单个 .log / .jsonl)下载下来发给我即可。结构化的 jsonl 我能直接算"先验 score 幅度 ↓ 是否与成功率 ↑ 同向"。
