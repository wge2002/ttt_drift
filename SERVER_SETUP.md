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

`robotwin_eval/` 只是适配器;真正跑要有 RoboTwin 2.0 仓库 + 其仿真依赖(SAPIEN 等),并把本仓库 symlink 进 `RoboTwin/policy/hy_vla`。

先修脚本 bug(循环里的 `local`):
```bash
sed -i 's/^\([[:space:]]*\)local /\1/' scripts/eval_robotwin_test.sh
```
再跑(按需改路径):
```bash
ROBOTWIN_DIR=/path/to/RoboTwin \
CKPT_PATH=$(pwd)/ckpts/Hy-VLA-RoboTwin \
CUDA_VISIBLE_DEVICES=0 TEST_NUM=10 \
bash scripts/eval_robotwin_test.sh        # 6 任务快速回归,日志落在 ./eval_logs/
```
全量(50 任务 ×100 rollout,很慢):`bash scripts/eval_robotwin_full.sh`(同样的 env 变量)。

## 6. 把结果打包回传给 Claude

```bash
# 完整 stdout + 退出码都留痕
<你的命令> 2>&1 | tee run_$(date +%Y%m%d_%H%M%S).log
# eval 日志 + 诊断 jsonl 一起打包
tar czf results_$(date +%Y%m%d_%H%M%S).tar.gz eval_logs/ *.jsonl run_*.log 2>/dev/null
```
把这个 tar.gz(或单个 .log / .jsonl)下载下来发给我即可。结构化的 jsonl 我能直接算"先验 score 幅度 ↓ 是否与成功率 ↑ 同向"。
