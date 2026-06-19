# Test-Time Drift toward the Language Manifold —— 方案构想与评估

> 项目:`ttt_drift`。目标:对抗 VLA / WAM 在部署时的分布偏移(OOD)与 rollout 复合误差,且**不依赖 0/1 reward / 不需要 human-in-the-loop**。
> 关联文件:`Drifting-Model-paper-summary.md`(Kaiming He 的 Drifting 论文摘录)、`VLA-TTT-调研笔记.md`(TTT/TTA 调研)。
> 已 vendor 进仓库:`hy_vla/`(base model = Hy-Embodied-0.5-VLA 源码,含 `robotwin_eval/`、`scripts/`)、`drifting_code/`(lambertae/drifting 参考实现,用于改 He 的 drift toy)。
> 状态:idea 阶段。⚠️ 原稿对"先验免费/(a) 零成本"过度乐观,已按 Hy-VLA 源码核实修正(见 §2.1、§4.5、§6)。

---

## 0. 一句话

把测试时的单个 OOD 样本,沿"**language-only action 先验**"的 score 场做几步 drift,把被坏视觉推离流形的 action latent(乃至输入图像)拉回到合法动作流形上——力天然来自"当前 vision-conditioned action 偏离 language 流形多远",因此**无需任何外部 reward**。

---

## 1. 借用 Kaiming He 的 Drifting:借的是什么,不是什么

He 等的 *Generative Modeling via Drifting*(arXiv:2602.04770)是一个**训练时**的生成范式:核心是一个漂移场 **V**,把生成分布 q 推向数据分布 p,并满足 **q == p 时 V == 0(平衡)**。它本身:

- **不是**推理时方法;
- **没有** OOD / reward / 视觉条件的概念;
- 解决的是"如何训出一个单步生成器"。

**我们借的是它的骨架,不是它的任务**:一个"当且仅当落在目标分布上时归零、否则指向目标分布"的力场。把它搬到 test-time、把目标分布换成 language 先验、并去掉监督信号。论文里必须先把这层关系框死,否则会被质疑"只是套用 He"。

| 维度 | He 的 Drifting | 本方案 |
|---|---|---|
| 时机 | 训练时演化整个分布 | 测试时对单个样本 drift 几步 |
| 目标分布 | 数据分布 p | language-only action 先验 `p(a\|l)` |
| 力 V | mini-batch 估计的漂移场 | 先验的 score `∇log p(a\|l)` |
| 监督 | 真实数据样本 | 无(力来自模型自身条件结构) |
| 平衡条件 | q==p 时 V==0 | a 落在 language 流形上时力为 0 |

---

## 2. 数学本质:它其实就是"OOD 时多信先验"

VLA 是条件分布 `p(a | v, l)`。按贝叶斯拆开:

```
p(a | v, l)  ∝  p(v | a, l) · p(a | l)
                 └─ 似然(靠视觉) ─┘   └─ 先验(只靠语言) ─┘
```

**OOD 的本质 = 视觉似然项 `p(v|a,l)` 不可信**(vision encoder 把 latent 推到训练未覆盖的区域,产生离流形的"垃圾分量")。理性应对就是**更多依赖先验 `p(a|l)`**。而:

- "language 流形" = 先验 `p(a|l)`;
- "把 action 拉向流形的力" = 先验的 **score**:

```
F(a) = ∇_a log p(a | l)
```

这个力的性质天然满足"drift":**在 language 流形上为 0,偏离时指向流形**——与 He 的平衡场 V 同构。这是本 idea 数学上最强、最干净的一环。

### 2.1 这个力"免费"是有前提的(原稿在此处过度乐观)

设想:若 VLA 是 **diffusion / flow-matching 策略**,把 vision 条件 **mask / dropout** 掉跑一遍 denoiser,其输出即 `∇log p(a|l)` 的估计,模型自身就给出这个力,无需显式拟合 language 流形。MG-Select 用的正是"mask 掉条件后的动作分布",只是拿去做**选择**而非 drift。

**但"mask 一下就拿到合法先验"只有一个充分条件:base model 训练时做过 condition / modality dropout**(因而学到了一个 unconditional / language-only 模式)。否则推理时硬 mask 掉 vision,是把模型推到**它从没见过的输入**上,吐出的 velocity 可能是 OOD 垃圾,而**不是** `p(a|l)`。

⚠️ **已对 Hy-VLA 源码核实:它不满足这个前提**(见 §4.5)。所以"这力免费"在本项目选定的 base model 上是一个**待验证的经验问题,不是数学事实**。先验是否有效,是整个 idea 的地基,必须最先验证(见 §6 step 0)。

---

## 3. 真正的新意:把"选"升级成"力"

| 工作 | 用 mask/先验信号做什么 |
|---|---|
| MG-Select / SCALE | 当**标量信号** → 在候选里**挑一个** / 调探索强度(verifier-free best-of-N) |
| **本方案** | 当**可微向量场** → 反传去**移动** latent,乃至**移动输入图像** |

即:把一个 selection signal 升级为 **drift gradient**。这是没人占的格子,paper 的主张应钉在此。

---

## 4. 实现光谱(由轻到重,建议从轻先验证)

**(a) 自适应 vision guidance ——「零成本」是有条件的,在 Hy-VLA 上不成立。**
CFG 本就是 `score = score(l) + w·(score(v,l) − score(l))`。"往 language 流形拉"在数学上**等价于把 vision 的 guidance 权重 w 调小**。**前提是 base model 本来就用 CFG / condition dropout 训练过**——这样 `w` 才存在、`score(l)` 才合法。在这种模型上 (a) 确实近乎零成本,只改一个标量。
但 **Hy-VLA 没有这个旋钮**(§4.5):采样是确定性 Euler 积分,无 `w`、无 unconditional 分支、训练无 dropout。在 Hy-VLA 上,(a) 不是"调旋钮",而是要先**造**出一个合法的 language-only 分支(≈ §4(b)/微调级成本)。
→ 推论:若想让 (a) 名副其实地当"最便宜的首发实验",要么换一个 CFG 训练过的 base model,要么接受在 Hy-VLA 上首发实验改为 §6 的 step 0(mask 有效性检验)。

**(b) 中等 —— latent drift。**
在 action latent 上做几步梯度,沿 `∇log p(a|l)` 把 a\* 投影回流形(本质是 manifold-projection denoising)。

**(c) 最重 / 最有故事 —— image drift。**
把力一路反传到输入图像像素,目标是"修感知"。最贴合最初设想,也风险最大(见 §5.3)。

---

## 4.5 base model 现实校验(已对 Hy-VLA 源码核实)

base model = `Tencent-Hunyuan/Hy-Embodied-0.5-VLA`,已 vendor 进本仓库根目录(`hy_vla/`)。对采样/前向代码核实结论:

| idea 原稿的隐含假设 | Hy-VLA 代码现实(`hy_vla/modeling_hy_vla.py`) |
|---|---|
| CFG 有权重 `w`,降 `w` 即往 language 拉 | `sample_actions`(~L1340–1380)为确定性 Euler:`x_t += dt * v_t`,**无 `guidance_scale`/`cfg`/`w`、无 unconditional 分支** |
| mask vision 跑一遍 = `∇log p(a\|l)` 的免费估计 | 前向/推理**从不 drop/mask/置空 vision token**;训练也**无 condition dropout**(无 null-embedding 路径) |
| vision 与 language 是可分离的两路条件 | 二者 **concat 成一条序列**,走统一 2D causal mask 的**共享注意力头**;无独立 cross-attention。视觉是"硬"条件,不是可加权的引导项 |

**唯一现成的可用杠杆**:代码为"某相机缺失"准备了 `img_mask` + 用 `-1` padding 的机制(~L1159)。可借它把**全部**相机标为缺失,作为 step 0 检验里"mask 掉 vision"的实现入口(注意:全相机缺失对训练分布仍是 OOD,这正是要检验的点)。

**结论**:§2.1 的"免费先验"、§4(a) 的"零成本"在此 base model 上均不成立。整个 idea 在 Hy-VLA 上能否落地,取决于"mask 掉 vision 后的 velocity 是不是一个合法的 language 先验"——这必须最先验证。

---

## 5. 三个决定成败的坑

### 5.1 别拉到"均值",要拉到"流形"
language 先验是**分布**(同一句话对应一堆合法动作),不是一个点。若"距离"定义为到先验均值的距离,drift 会把动作拍成 mode-average,**抹掉视觉带来的特化**,本末倒置。必须用 **score 场**(在流形上为 0、只清掉**垂直于流形**的分量),保留沿流形自由度。

### 5.2 写明核心假设:OOD = 偏离流形的噪声,而非"合法的新动作"
若某未见场景**确实需要**离 language 先验很远的合法动作,往先验拉就是帮倒忙。对策:加门控——**仅当动作落在先验近零密度区**才施力,且力随偏离程度**饱和**。

### 5.3 无 reward 的 TTT 会自我强化幻觉
(c) 方案最大风险:把图像优化到"让策略更自信",可能只是把图改成**策略爱看的样子**(类似 entropy-min 塌缩),感知并没修对。对策:
- **trust-region**:限制图像/latent 改动幅度、锚定原始输入、只走几步;
- **判据**:不能只看"力变小 / 置信变高",要直接看 **OOD 成功率**是否真涨。力降但成功率不动 = 在自我欺骗。

---

## 6. 最小验证(de-risk plan)

- **模型**:`Hy-Embodied-0.5-VLA`(conditional flow matching,370M action expert,10 步 Euler;无 CFG,见 §4.5)。
- **数据**:Hy-VLA 自带 `robotwin_eval/`(RoboTwin 2.0);OOD 维度优先用 Randomized split,后续再上 LIBERO-PRO / CALVIN 的扰动 split。
- **指标**:OOD 成功率 vs ID 成功率;外加诊断指标"**先验 score 幅度 ↓ 是否与成功率 ↑ 同向**",用以证明不是自我欺骗。

**顺序(已按 §4.5 现实校正,不再是"先跑 a"):**

- **step 0(地基,半天,最便宜)—— mask 有效性检验。** 借 `img_mask` 把全部相机标为缺失,跑 `sample_actions`,看输出 action chunk 是"忽略画面但仍朝指令方向的合法先验"还是"乱动/塌成均值"。脚本:`scripts/diag_step0_mask.py`。
  - 有效 → §2.1 成立,继续往下;
  - 无效 → 必须先 LoRA 微调注入 vision-dropout 的 unconditional 分支(掉到中等成本),否则 (a)(b)(c) 全是空中楼阁。
  - **✅ 结果(2026-06-19,dummy 零观测,seed 0,RoboTwin ckpt):PASS。** masked 分支的 action chunk 随指令显著变化:跨指令 mean pairwise ‖Δa‖ = **5.62**,甚至**略高于** normal 分支的 4.99(ratio 1.13);各指令 masked 动作与空指令基线的距离 2.4–6.8,非零且有结构;action std 0.24–0.34、无塌缩。→ **尽管 Hy-VLA 训练时无 condition dropout,mask 掉 vision 后语言仍在合法地条件化动作**,§2.1 地基成立,不需要先做微调。
  - 注意边界:本次用全零 dummy 图像,故 (i) 只证明了先验"存在且随指令变化",**未**证明其"正确"(需 sim/GT);(ii) `vision_displacement ‖a_normal−a_masked‖≈10` 是"垃圾(黑图)视觉把动作推离语言流形 ~10"的体现——正是核心假设要对抗的现象,但真实 OOD 图像下才有意义。
- **step 1 —— (a) 手动 CFG / velocity blend**(step 0 既已证明 masked 分支是合法先验,(a) 现在可行且最轻:每个 Euler step 跑两遍 denoise(full + masked),`v = w·v_full + (1−w)·v_masked`,`w<1` 即往语言先验拉;`w` 可随偏离自适应。**无需训练,仅 2× 前向**)。脚本:`scripts/step1_velocity_blend.py`。
  - **✅ 离线验证 PASS(2026-06-19,dummy 零观测,seed 0):** (i) **w=1 与原版 `sample_actions` 逐 bit 一致**(anchor 0.0000,且 stock 跑两遍 floor 也 0.0000 → 模型前向确定性,blend 机制无误);(ii) **w:1→0 单调、平滑**,动作从 a_full 平滑滑到 a_masked(三指令 ‖a_full−a_lang‖≈10–12,w=0.5 约居中);(iii) abs_mean 随 w↓ 轻微降、std 健康、**无塌缩**;velocity 场全程 5–8 稳定,`v_full` 沿语言先验轨迹的范数随 w↓ 增大(= guidance 项幅度,有界稳定)。
  - **下一步(真正的检验):** 装 RoboTwin 2.0,把该 blend 插进 `robotwin_eval/policy_wrapper.py`,在 OOD split 上扫 `w` 看成功率,并验证「先验偏离幅度 ↓ 与成功率 ↑ 同向」。需真实图像 + sim。
- **step 2 —— (b) latent drift / (c) image drift**(更重,(a) 见效后再上)。
- **baseline**:MG-Select、VLS。
- **依赖**:step 1 的成功率检验需 RoboTwin 2.0 模拟器(见 `SERVER_SETUP.md` §5);离线的 `w`-收敛验证不需要。

可选预实验:基于 He 的 toy 2D notebook 改一个"language 流形 + OOD 点 drift"的玩具,先肉眼看现象(力是否把离群点拉回流形、是否塌成均值)。

---

## 7. 与调研中已有工作的定位

- **MG-Select(2510.05681)**:同源信号(mask 条件分布),但做 selection;本方案做 drift gradient。
- **SCALE(2602.04208)**:self-uncertainty 调制感知+动作,单前向;本方案把不确定性变成可微的力并反传。
- **VLS(2602.03973)**:用 VLM reward 引导去噪路径(需外部 VLM 打分);本方案的力**内生于策略自身**,不需外部打分器。
- **Anti-Exploration TTS(2512.02834)**:约束动作落入训练分布内,动机一致;本方案给出"力场"的具体来源与可微实现。

---

## 8. 一句话主张(给 paper 的 thesis)

> 将 test-time adaptation 表述为:在 vision 似然不可信的 OOD 区,沿 **language 先验的 score 场**对 action latent(乃至输入)做平衡式 drift——一个"落在合法动作流形上即归零"的内生力,把对抗偏移变成一次无监督的流形投影,无需任何 reward。
