# VLA 的 Test-Time Training(TTT)及"类似思想"工作调研

> 范围:Vision-Language-Action(VLA)/ 机器人策略 在**部署 / 推理阶段做自适应**的工作。既包含严格意义上的 test-time training(在测试时真正更新权重),也包含思想相近的 test-time adaptation / test-time RL / 推理时 scaling / verifier 选择 / online memory steering / 推理时 reasoning 等。
>
> 更新日期:2026-06-17。下面每条都带 arXiv 链接,便于后续持续追加。
>
> 标注约定:
> - **【更新权重】** 真正在测试时做梯度更新(最贴近经典 TTT)
> - **【冻结+引导】** 不改权重,在采样/去噪过程中引导冻结策略
> - **【采样+打分】** best-of-N 采样,用 verifier/value 选动作
> - **【自置信选择】** 无外部 verifier,用模型自身不确定性选动作
> - **【推理时推理】** 在动作前显式生成中间表征(子目标图/CoT)

---

## 0. 背景:TTT 这条线的"祖宗"思想

经典 TTT 的核心:**测试样本本身可以构造自监督信号**,在推理时对每个测试输入做几步梯度更新,以对抗分布漂移(distribution drift)。这与机器人里"behavior cloning 在 rollout 中误差累积(compounding error)、状态逐渐漂出训练分布"的痛点天然契合——这也是本文件夹命名 `ttt_drift` 的由来。

相关理论背景(非 VLA,但是这套叙事的根):
- 模仿学习的复合误差 / 协变量漂移:BC 在长 horizon 下最坏误差随步数二次增长,DAgger、GAIL 等用交互/对抗把它压到线性。代表性近期理论:*The Pitfalls of Imitation Learning when Actions are Continuous*([arXiv:2503.09722](https://arxiv.org/pdf/2503.09722))、*Non-Adversarial Imitation Learning Provably Free of Compounding Errors*([arXiv:2603.22713](https://arxiv.org/abs/2603.22713))。
- VLA 把这个问题放大了:动作依赖于"预测的未来"(visual foresight)或随机解码,一旦 OOD,误差会顺着 rollout 滚雪球——这正是下面这些 test-time 方法想救的场景。

---

## 1. 严格意义的 Test-Time Training(测试时更新权重)

这一组最贴近"VLA 的 TTT",共同点是**用 rollout 自身产生的信号在部署时更新模型**。

### TTT-VLA — Test-Time Latent Prompt Optimization 【更新权重(仅 latent prompt)】⭐
- 链接:[arXiv:2606.03127](https://arxiv.org/abs/2606.03127) · [HTML](https://arxiv.org/html/2606.03127v1)
- 机构/时间:Wenbo Zhang、Jianxiong Li、Shuai Yang、Sijin Chen、Jiajun Liu、Lingqiao Liu、Xiao Ma;2026-06-02。
- 核心思想:提出 **Latent Prompt Optimization(LPO)**。训练时用一个额外 proxy task 学一个 latent prompt,作为策略学习的额外条件信号;**测试时只收集当前环境的交互数据,用 proxy task 的自监督信号去优化这个 latent prompt,完全不动策略权重**。把"prompt 这个 steering 接口"本身变成可学习、可在部署时自适应的对象。SimplerEnv 上单/多 embodiment 一致提升;分析显示增益主要来自**纠正少数关键决策**,而非全局改变策略。
- 与 TTT 关系:**这就是字面意义上的 "TTT-VLA"**。它在"改权重 vs. 冻结引导"之间开了第三条路——做测试时梯度优化,但参数面缩到只剩 latent prompt,既有 TTT 的自适应力,又避免动整个策略带来的崩坏风险。

### T³VF — Test-Time Training for Visual Foresight VLA 【更新权重】
- 链接:[arXiv:2605.08215](https://arxiv.org/abs/2605.08215)
- 核心思想:Visual-foresight 型 VLA 的动作质量取决于"预测的未来图像"是否准。关键观察是——**预测的未来帧和它真正发生后的下一帧,天然构成一对监督**。于是在测试时用这对自监督做梯度更新。为避免对每一步都盲目更新带来的崩坏,提出 **adaptive update filtering(自适应更新过滤)** 机制。
- 与 TTT 关系:**最教科书式的 VLA-TTT**——用测试时自然产生的"预测 vs. 实际"配对做 self-supervision,直接对应经典 TTT 的自监督代理任务。

### EVOLVE-VLA — Test-Time Training from Environment Feedback 【更新权重】
- 链接:[arXiv:2512.14666](https://arxiv.org/pdf/2512.14666) · [项目页](https://showlab.github.io/EVOLVE-VLA/) · [HF](https://huggingface.co/papers/2512.14666)
- 核心思想:让 VLA 通过**与环境交互持续自适应**,只需极少甚至零任务示范。用一个学到的 **progress estimator(进度估计器)** 提供 dense 反馈;为对抗噪声信号,引入累积式进度估计 + 渐进式 horizon 扩展。
- 与 TTT 关系:测试时持续 finetune,信号来自环境反馈而非外部标注,属于"自我进化"式 TTT。

### TT-VLA — On-the-Fly VLA Adaptation via Test-Time Reinforcement Learning 【更新权重】
- 链接:[arXiv:2601.06748](https://arxiv.org/abs/2601.06748) · [v2](https://arxiv.org/abs/2601.06748v2) · [综述页](https://www.emergentmind.com/topics/test-time-reinforcement-learning-for-vlas-tt-vla)
- 机构/时间:UMKC、HKUST、Meta AI 等;2026-01。
- 核心思想:推理时做 RL 的 **on-the-fly 策略自适应**。设计 dense reward,利用逐步的 task-progress 信号在测试时 refine 策略,同时**保留 SFT/RL 训得的先验**。在仿真和真机的动态未见场景下提升适应性、稳定性、成功率。
- 与 TTT 关系:把 TTT 的"测试时更新"具体化为 test-time RL,是当前"self-improving, deployment-ready VLA"叙事的代表。

### WorldAgen — Unified State-Action Prediction with Test-Time World Model Training 【更新权重】
- 链接:[AAAI 版 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/38925/42887)
- 核心思想:统一 state-action 预测的世界模型,在部署时用**短探索性 rollout 做轻量 TTT**,把"世界建模"本身变成在线自适应信号,让 agent 快速适应新环境。CALVIN / LIBERO 上有稳定增益。
- 与 TTT 关系:**这是最新一篇典型的"测试时训练世界模型"VLA 工作**,把 TTT 信号建在 world-model 预测上,和 T³VF 的"预测 vs 实际"自监督异曲同工,但落在 world model 而非单帧 foresight 上。

### (邻域)World-model 驱动的在线自适应策略
- **AdaWorldPolicy**(World-Model-Driven Diffusion Policy + Online Adaptive Learning):[arXiv:2602.20057](https://arxiv.org/html/2602.20057v1) — 在线自适应提升成功率、缓解 train/test 分布漂移。
- **Act2Goal**(From World Model to Goal-conditioned Policy):[arXiv:2512.23541](https://arxiv.org/html/2512.23541v1) — 部署时用轻量 on-device finetune 做在线自我改进。
- 说明:这两条是 diffusion policy / world model 方向,非纯 VLA,但"测试时在线更新对抗漂移"思想一致,作为邻域工作记录。

---

## 2. Test-Time Adaptation:冻结策略 + 推理时引导(不改权重)

共同点:**策略权重冻结**,在采样/去噪过程里施加引导来对抗 OOD,代价低、可即插即用。

### VLS — Steering Pretrained Robot Policies via Vision-Language Models 【冻结+引导】
- 链接:[arXiv:2602.03973](https://arxiv.org/abs/2602.03973) · [项目页](https://vision-language-steering.github.io/webpage/) · [代码](https://github.com/Vision-Language-Steering/code)
- 核心思想:training-free 的推理时自适应框架。把自适应看成**推理时控制问题**,用 VLM 为"部分去噪的动作提案"生成 reward,纠正 diffusion/flow-matching 策略的去噪路径,应对物体/场景/指令变化等 OOD。三种 steering 机制:gradient-based refinement、RBF diversity、Feynman–Kac resampling。CALVIN +31%,LIBERO-PRO +13%,真机 Franka 验证。
- 与 TTT 关系:不更新权重,但同样是"测试时针对当前输入做自适应",是 TTT 的轻量替代路线。

### Retrieve-then-Steer — Online Success Memory for Test-Time Adaptation 【冻结+引导】
- 链接:[arXiv:2605.10094](https://arxiv.org/html/2605.10094v2)
- 核心思想:把 VLA 部署重新定义为**持续在线自适应过程**(而非孤立的一次次 trial)。用非参数化的 "retrieve-then-steer":维护一个 progress-calibrated 的成功记忆库,抽取可复用片段,把一致性过滤后的 elite 先验注入生成式采样。对冻结 VLA 做轻量自适应。
- 与 TTT 关系:用"记忆 + 检索"替代梯度更新来实现测试时自适应,思想同源。

### TACO — Steering VLA as Anti-Exploration: A Test-Time Scaling Approach 【冻结+引导】
- 链接:[arXiv:2512.02834](https://arxiv.org/abs/2512.02834)
- 核心思想:把测试时动作选择建模为 anti-exploration(偏向落在训练分布内的动作);用一个轻量 pseudo-count 估计器作为 action chunk 的高保真 verifier,作为 test-time scaling 手段引导冻结策略。
- 与 TTT 关系:推理时对动作分布施加约束以抗漂移,属冻结引导一类。

### PhysMem — Scaling Test-time Physical Memory for Robot Manipulation 【冻结+引导】
- 链接:[arXiv:2602.20323](https://arxiv.org/pdf/2602.20323)
- 核心思想:在测试时扩展一个"物理记忆"库供机器人检索复用,作为部署期增益来源(与 Retrieve-then-Steer 同属 memory 一路,但强调可扩展的物理记忆)。
- 与 TTT 关系:用测试时记忆扩展替代权重更新,memory-based test-time scaling。

### (邻域)Generative Predictive Control(GPC)
- 思想:在部署时把冻结 diffusion policy 与一个预测式 world model 耦合,做推理时增强 / 测试时自适应。非纯 VLA,记录为邻域。

---

## 3. Test-Time Scaling:多采样 + 外部 verifier/value 选动作

共同点:**不改 backbone**,推理时多采样若干候选动作,用一个 reward/value/verifier 打分挑最优——存在"推理时 scaling law"。

### RoboMonkey — Scaling Test-Time Sampling and Verification 【采样+打分】
- 链接:[arXiv:2506.17811](https://arxiv.org/abs/2506.17811) · [项目页](https://robomonkey-vla.github.io/) · [PMLR](https://proceedings.mlr.press/v305/kwok25a.html)
- 机构/会议:Stanford、UC Berkeley、NVIDIA;CoRL 2025。
- 核心思想:部署时从 VLA 采一小批动作,加高斯扰动 + 多数投票构造动作提案分布,再用 **VLM-based verifier** 选最优动作。发现 action error 与采样数呈 **exponentiated power law**(推理时 scaling law)。配套合成数据流水线训练 verifier。OOD 任务 +25%,ID +9%。
- 与 TTT 关系:不更新权重,但用"测试时多花算力 + 验证"换鲁棒性,是 external test-time scaling 的标杆。

### Hume — Introducing System-2 Thinking in VLA 【采样+打分】
- 链接:[arXiv:2505.21432](https://arxiv.org/abs/2505.21432) · [项目页](https://hume-vla.github.io/) · [代码](https://github.com/hume-vla/hume)
- 核心思想:双系统 VLA。System-2 给 backbone 加一个 **value-query head** 估计状态-动作价值,**重复采样多个候选动作并按 value 做 best-of-N 选择**;再用 cascaded action denoising 把 System-1/2 融合,兼顾慢思考与高频控制。
- 与 TTT 关系:value-guided 的测试时 best-of-N,是 internal value + 推理时 scaling 的代表。

### RoVer — Robot Reward Model as Test-Time Verifier 【采样+打分】
- 链接:[arXiv:2510.10975](https://arxiv.org/html/2510.10975)
- 核心思想:用一个机器人 reward model 作为测试时 verifier,对 VLA 候选动作打分选优。
- 与 TTT 关系:同 RoboMonkey 一路的 verifier-based test-time scaling。

### Vision-Language-Action-Critic Model 【采样+打分】
- 链接:[arXiv:2509.15937](https://arxiv.org/pdf/2509.15937)
- 核心思想:在 VLA 上接一个 critic,用价值估计在推理时指导动作选择/评估。
- 与 TTT 关系:internal critic 引导测试时决策,介于 value-guided scaling 与 verifier 之间。

### Value-VLA — Value Vision-Language-Action Planning & Search 【采样+打分】
- 链接:[arXiv:2601.00969](https://arxiv.org/html/2601.00969v1)
- 核心思想:用学到的 value 在推理时做规划/搜索,引导 VLA 的动作展开(planning + search 而非单步 best-of-N)。
- 与 TTT 关系:value-guided 的测试时搜索,external test-time scaling 的"带规划"版本。

---

## 4. Verifier-free Test-Time Scaling:用模型自身不确定性选动作

共同点:**不需要额外 verifier / 不更新权重**,用模型自身的置信/不确定性在推理时调节。

### MG-Select — Verifier-free Test-Time Sampling 【自置信选择】
- 链接:[arXiv:2510.05681](https://arxiv.org/abs/2510.05681) · [HF](https://huggingface.co/papers/2510.05681) · [ICLR 2026](https://openreview.net/pdf?id=UD4Rw8MOEK)
- 核心思想:用**预测动作分布**与**条件被 mask 后的动作分布**之间的 KL 散度,作为自置信信号,从多个候选里选最优动作。参考分布由同一个 VLA 在随机 mask 掉 state/language 条件下生成。配套联合训练:随机 text/state dropout 让模型同时学会条件分布与 mask 分布。低数据设置下相对提升达 168%(RoboCasa / SIMPLER-WidowX / LIBERO / Franka)。
- 与 TTT 关系:verifier-free 的测试时 scaling 新范式,用模型自身不确定性替代外部打分器。

### SCALE — Self-uncertainty Conditioned Adaptive Looking and Execution 【自置信选择】
- 链接:[arXiv:2602.04208](https://arxiv.org/abs/2602.04208)
- 核心思想:受 Active Inference 的"不确定性驱动探索"启发,用 **self-uncertainty** 同时调制视觉感知与动作:不确定时在感知和动作上都加大探索,自信时收敛到 exploitation。无需额外训练、无 verifier、仅一次前向。
- 与 TTT 关系:单次前向的自适应执行,极轻量的测试时自调节。

### Adaptive Action Chunking at Inference-time(AAC)【自置信选择】
- 链接:[arXiv:2604.04161](https://arxiv.org/abs/2604.04161) · [项目页](https://lance-lot.github.io/adaptive-chunking.github.io/)
- 核心思想:用 **action entropy** 作为线索在推理时自适应决定 chunk 大小:熵高(不确定)用小 chunk 保持反应性,熵低用大 chunk。纯推理时、无需额外训练或改架构。
- 与 TTT 关系:不改权重,用自身不确定性在测试时调节"何时重规划",抗漂移的轻量手段。

### PDF — Test-Time Perturbation Learning with Delayed Feedback 【自置信选择 + 轻量模块】
- 链接:[arXiv:2604.18107](https://arxiv.org/abs/2604.18107)
- 核心思想:verifier-free 的测试时自适应,不微调 base 模型。把 VLA 对物体位姿等微小变化的脆弱性归因为 **trajectory overfitting**(过度关注动作与实体的虚假相关、复现记忆中的动作模式)。用基于不确定性的数据增强 + 动作投票缓解虚假相关;自适应调度器分配增强预算;并学一个**轻量 perturbation 模块**,在延迟反馈引导下回溯性地调整 action logits、纠正过自信。
- 与 TTT 关系:测试时只学一个极小的扰动模块(不动 base),思想介于自置信选择与"仅更新小参数"的 TTT 之间,与 TTT-VLA 的 LPO 精神相近。

> 相关邻域:*Adaptive Action Chunking via Multi-Chunk Q Value Estimation*([arXiv:2605.10044](https://arxiv.org/html/2605.10044))——用 Q 值而非熵来自适应 chunk,偏训练侧,作对照记录。

---

## 5. 推理时推理 / Internal Test-Time Scaling(动作前生成中间表征)

共同点:把"测试时多花算力"花在**显式中间推理**(子目标图、视觉 CoT、反思)上,再据此产动作。

### CoT-VLA — Visual Chain-of-Thought Reasoning 【推理时推理】
- 链接:[arXiv:2503.22020](https://arxiv.org/abs/2503.22020) · [项目页](https://cot-vla.github.io/) · CVPR 2025;NVIDIA。
- 核心思想:把显式**视觉 CoT** 引入 VLA——先自回归预测未来图像帧作为视觉目标,再生成一小段动作去达成。部署时:给定观测+指令,先用 causal attention 生成子目标图,再用 full attention 生成动作。7B,基于 VILA-U。真机 +17%,仿真 +6%。
- 与 TTT 关系:internal test-time scaling 的代表——推理时显式"想象未来"作为中间步骤。

### FlowVLA — Visual CoT-based Motion Reasoning 【推理时推理】
- 链接:[项目页](https://irpn-lab.github.io/FlowVLA/)
- 核心思想:在视觉 CoT 基础上用运动(flow)推理,先推理"怎么动"再产动作。
- 与 TTT 关系:同 CoT-VLA 一路的推理时中间表征。

### Counterfactual VLA — Self-Reflective VLA with Adaptive Reasoning 【推理时推理】
- 链接:[arXiv:2512.24426](https://arxiv.org/html/2512.24426v1)
- 核心思想:自反思式 VLA,推理时做反事实/自我反思并自适应决定推理深度。
- 与 TTT 关系:推理时自适应"想多久",internal scaling 与自置信的结合。

> 还可纳入:**OneTwo-VLA**——用自适应 token 决定"何时 reason、何时 act"(在多份综述里与 CoT-VLA / UniVLA 并列为 internal test-time scaling 代表);**UniVLA** 的 future-frame 预测。后续可补全链接。

---

## 6. 速查表

| 方法 | 类别 | 是否改权重 | 测试时信号 / 机制 | 链接 |
|---|---|---|---|---|
| **TTT-VLA** ⭐ | 严格 TTT(仅 prompt) | ✅(仅 latent prompt) | 自监督 proxy 优化 latent prompt | [2606.03127](https://arxiv.org/abs/2606.03127) |
| T³VF | 严格 TTT | ✅ | 预测未来帧 vs. 实际帧自监督 + 更新过滤 | [2605.08215](https://arxiv.org/abs/2605.08215) |
| EVOLVE-VLA | 严格 TTT | ✅ | 环境反馈 + progress estimator | [2512.14666](https://arxiv.org/pdf/2512.14666) |
| TT-VLA(俗称 TTT-VLA) | 严格 TTT(RL) | ✅ | dense task-progress reward,test-time RL | [2601.06748](https://arxiv.org/abs/2601.06748) |
| WorldAgen | 严格 TTT(world model) | ✅ | 短 rollout 做测试时世界模型训练 | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/download/38925/42887) |
| AdaWorldPolicy | 邻域(world model) | ✅ | 在线自适应 diffusion policy | [2602.20057](https://arxiv.org/html/2602.20057v1) |
| Act2Goal | 邻域(world model) | ✅ | on-device 轻量 finetune 自改进 | [2512.23541](https://arxiv.org/html/2512.23541v1) |
| VLS | 冻结+引导 | ❌ | VLM 生成 reward 引导去噪路径 | [2602.03973](https://arxiv.org/abs/2602.03973) |
| Retrieve-then-Steer | 冻结+引导 | ❌ | 成功记忆检索 + elite 先验注入采样 | [2605.10094](https://arxiv.org/html/2605.10094v2) |
| Anti-Exploration TTS | 冻结+引导 | ❌ | anti-exploration 约束动作落入分布内 | [2512.02834](https://arxiv.org/abs/2512.02834) |
| RoboMonkey | 采样+打分 | ❌ | 扰动+投票采样 + VLM verifier | [2506.17811](https://arxiv.org/abs/2506.17811) |
| Hume | 采样+打分 | ❌ | value-query head best-of-N | [2505.21432](https://arxiv.org/abs/2505.21432) |
| RoVer | 采样+打分 | ❌ | robot reward model 作 verifier | [2510.10975](https://arxiv.org/html/2510.10975) |
| VLA-Critic | 采样+打分 | ❌ | critic 价值估计引导 | [2509.15937](https://arxiv.org/pdf/2509.15937) |
| Value-VLA | 采样+打分(planning) | ❌ | value 引导的测试时规划/搜索 | [2601.00969](https://arxiv.org/html/2601.00969v1) |
| MG-Select | 自置信选择 | ❌ | 与 mask 条件分布的 KL 作自置信 | [2510.05681](https://arxiv.org/abs/2510.05681) |
| SCALE | 自置信选择 | ❌ | self-uncertainty 调制感知+动作,单前向 | [2602.04208](https://arxiv.org/abs/2602.04208) |
| AAC | 自置信选择 | ❌ | action entropy 自适应 chunk 大小 | [2604.04161](https://arxiv.org/abs/2604.04161) |
| PDF | 自置信+轻量模块 | ✅(仅扰动模块) | 不确定性增强+动作投票+延迟反馈学扰动 | [2604.18107](https://arxiv.org/abs/2604.18107) |
| PhysMem | 冻结+引导(memory) | ❌ | 可扩展测试时物理记忆检索 | [2602.20323](https://arxiv.org/pdf/2602.20323) |
| CoT-VLA | 推理时推理 | ❌ | 先预测子目标图再产动作 | [2503.22020](https://arxiv.org/abs/2503.22020) |
| FlowVLA | 推理时推理 | ❌ | 视觉 CoT + motion/flow 推理 | [项目页](https://irpn-lab.github.io/FlowVLA/) |
| Counterfactual VLA | 推理时推理 | ❌ | 自反思 + 自适应推理深度 | [2512.24426](https://arxiv.org/html/2512.24426v1) |

---

## 7. 一句话总结与待补

**光谱:** 从"真改权重"(T³VF / EVOLVE-VLA / TT-VLA)→ "冻结但引导采样"(VLS / Retrieve-then-Steer / Anti-Exploration)→ "采样+外部打分"(RoboMonkey / Hume / RoVer / VLA-Critic)→ "采样+自身不确定性"(MG-Select / SCALE / AAC)→ "推理时显式推理"(CoT-VLA / FlowVLA / Counterfactual)。越往右越轻量、越不动权重;越往左越接近经典 TTT。

**共同动机:** 都在救同一件事——**VLA 在部署时 OOD + rollout 复合误差导致的漂移**。区别只在于"用多大代价、动不动权重"来对抗漂移。

> 补充:**TTT-VLA(2606.03127)的 LPO** 又在光谱上插了一档——"做梯度优化、但只优化 latent prompt";**PDF(2604.18107)** 类似,"只学一个轻量扰动模块"。这条"仅更新极小参数子集"的中间路线,可能是兼顾自适应力与稳定性的甜点区,值得重点跟。**这个方向出文速度很快(2026 上半年密集涌现),后续大概率还会持续有新工作,建议定期复扫。**

**待补(看到再加):**
- OneTwo-VLA、UniVLA 的 internal test-time scaling 细节与链接
- **MoS-VLA**(One-Shot Skill Adaptation,[2510.16617](https://arxiv.org/html/2510.16617)):一/少样本技能自适应,确认是否算 test-time
- **HiF-VLA**(Hindsight/Insight/Foresight,[2512.09928](https://arxiv.org/pdf/2512.09928)):foresight 表征,确认 train vs. test-time
- **RobustVLA**(Robustness-Aware RL Post-Training,[2511.01331](https://arxiv.org/pdf/2511.01331)):偏 post-training,作鲁棒性对照
- 纯 diffusion/flow policy 的 test-time 工作(GPC、real-time chunking 等)是否纳入
- 经典 TTT(Sun et al. TTT、TTT-MAE、TENT/test-time entropy minimization)作为"思想源头"是否单列一节
- **TTT-Parkour**(Rapid Test-Time Training for Perceptive Robot Parkour,[arXiv:2602.02331](https://arxiv.org/html/2602.02331v1)):腿足 locomotion 的 TTT,非 VLA 但同属"机器人测试时训练",作邻域候选
- **World-VLA-Loop**(闭环 video world model + VLA,[arXiv:2602.06508](https://arxiv.org/pdf/2602.06508)):偏训练时闭环,待确认是否有 test-time 成分
