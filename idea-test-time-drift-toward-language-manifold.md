# Test-Time Drift toward the Language Manifold —— 方案构想与评估

> 项目:`ttt_drift`。目标:对抗 VLA / WAM 在部署时的分布偏移(OOD)与 rollout 复合误差,且**不依赖 0/1 reward / 不需要 human-in-the-loop**。
> 关联文件:`Drifting-Model-paper-summary.md`(Kaiming He 的 Drifting 论文摘录)、`VLA-TTT-调研笔记.md`(TTT/TTA 调研)。
> 状态:idea 阶段,本文件用于把核心论证、与已有工作的关系、风险与最小验证钉清楚。

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

### 2.1 关键:这个力大概率"免费"

若 VLA 是 **diffusion / flow-matching 策略**(π0、RDT 一类):把 vision 条件 **mask / dropout** 掉跑一遍 denoiser,其输出即 `∇log p(a|l)` 的估计。**无需真去拟合 language 流形或显式算分布距离**——模型自身就给出这个力。这也直接接上调研里的 **MG-Select**:它用的正是"mask 掉条件后的动作分布",只是拿去做了**选择**,而非 drift。

---

## 3. 真正的新意:把"选"升级成"力"

| 工作 | 用 mask/先验信号做什么 |
|---|---|
| MG-Select / SCALE | 当**标量信号** → 在候选里**挑一个** / 调探索强度(verifier-free best-of-N) |
| **本方案** | 当**可微向量场** → 反传去**移动** latent,乃至**移动输入图像** |

即:把一个 selection signal 升级为 **drift gradient**。这是没人占的格子,paper 的主张应钉在此。

---

## 4. 实现光谱(由轻到重,建议从轻先验证)

**(a) 最轻 / 几乎零成本 —— 自适应 vision guidance。**
CFG 本就是 `score = score(l) + w·(score(v,l) − score(l))`。"往 language 流形拉"在数学上**等价于把 vision 的 guidance 权重 w 调小**。于是方案可先退化为"**按偏离程度自适应降低 w**",连图像反传都不需要,最快能跑出 ablation,也最适合先证伪/证实核心假设。

**(b) 中等 —— latent drift。**
在 action latent 上做几步梯度,沿 `∇log p(a|l)` 把 a\* 投影回流形(本质是 manifold-projection denoising)。

**(c) 最重 / 最有故事 —— image drift。**
把力一路反传到输入图像像素,目标是"修感知"。最贴合最初设想,也风险最大(见 §5.3)。

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

- **模型**:一个 flow / diffusion VLA(如 π0 类 flow-matching 策略)。
- **数据**:LIBERO-PRO 或 CALVIN 的 OOD split(物体/场景/指令扰动)。
- **对比**:
  - (a) 自适应降 vision guidance
  - (b) latent drift
  - (c) image drift
  - baseline:MG-Select、VLS
- **指标**:OOD 成功率 vs ID 成功率;外加诊断指标"**先验 score 幅度 ↓ 是否与成功率 ↑ 同向**",用以证明不是自我欺骗。
- **顺序**:先把 (a) 跑通——几乎零实现成本,却能直接证伪/证实核心假设;再依次上 (b)(c)。

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
