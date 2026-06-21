# Sweep 1 — 恒定 velocity-blend (constant w) RoboTwin 结果与分析

> 第一次真·sim 评测(step2):用 `v = w·v_full + (1−w)·v_masked` 的恒定 w 把动作往 language 先验拉,看 OOD/ID 成功率随 w 的变化。
> **结论:干净的负结果——w 越小成功率单调崩塌,OOD 与 ID 同等崩。但有两个混淆因素使其没有真正检验到核心假设(见 §3)。**

---

## 0. 评测设置

- 机器:`fnii-vla2`(hostname GPU01),A100-80GB;env `RoboTwin_wg`(py3.10 / torch2.4.1+cu121),SAPIEN 渲染正常。
- 代码:`/code/wge/ttt_drift`;RoboTwin:`/code/wge/RoboTwin`;ckpt:`tencent/Hy-Embodied-0.5-VLA-RoboTwin`(本地 `ckpts/Hy-VLA-RoboTwin`)。
- 干预:`hy_vla/ttt_blend.py` 的 velocity blend,经 `robotwin_eval/policy_wrapper.py` 的 `guidance_w`(env `HYVLA_GUIDANCE_W`)注入;w=1.0 与原版采样逐 bit 一致。
- 协议:6 个回归任务,每格 **20 episode**(`test_num` 已 patch 为可配);`instruction_type=unseen`,`seed=10000`。
- 轴:`TASK_CONFIG=demo_randomized`(拟作 OOD)对 `demo_clean`(ID);w∈{1.0, 0.75, 0.5(部分), 0.25(未跑)}。
- 原始日志在 fnii-vla2 `/code/wge/ttt_drift/eval_logs/<config>/w_<w>/*.log`(未入 git)。sweep 在 w=0.5 跑到一半时按需手动停止。

---

## 1. OOD:demo_randomized 成功率(%)

| task | w=1.0 | w=0.75 | w=0.5 |
|---|---|---|---|
| adjust_bottle | 100 | 100 | 60 |
| beat_block_hammer | 90 | 40 | 5 |
| blocks_ranking_rgb | 100 | 85 | 5 |
| blocks_ranking_size | 90 | 85 | (未完成) |
| click_alarmclock | 100 | 100 | (未完成) |
| click_bell | 95 | 90 | (未完成) |
| **平均(完成格)** | **95.8** | **83.3** | 已完成 3 个均值 ≈ **23.3** |

## 2. ID:demo_clean 成功率(%)

| task | w=1.0 | w=0.75 | w=0.5 |
|---|---|---|---|
| adjust_bottle | 100 | 100 | 45 |
| beat_block_hammer | 100 | 65 | 30 |
| blocks_ranking_rgb | 100 | 95 | (未完成) |
| blocks_ranking_size | 100 | 70 | (未完成) |
| click_alarmclock | 100 | 100 | (未完成) |
| click_bell | 100 | 100 | (未完成) |
| **平均(完成格)** | **100** | **88.3** | 崩 |

**趋势**:w 1.0 → 0.75 → 0.5,两条都单调下降并在 0.5 急剧崩塌;OOD 与 ID 降幅相近。没有任何一个任务在降 w 时变好。

---

## 3. 分析:为什么这是"负结果但没测到真假设"

两个混淆因素:

1. **demo_randomized 对 Hy-VLA 不构成视觉 OOD。** w=1.0 时 OOD 95.8% ≈ ID 100%,**仅差 ~4 点**。Hy-VLA 用 10k 小时 + 强域随机化训练,demo_randomized 的视觉扰动基本在其分布内——视觉没"坏"。**没有 OOD gap,就没有东西可供"多信先验"去挽救。**

2. **恒定 w 是 idea 明确反对的用法。** `v_masked` 是"无视图像"的先验;全局、无条件地把动作往它拉,在视觉可信处只会丢弃有用信息 → 单调掉点。idea §5.2 早写明:**力必须门控**(仅在动作落入先验近零密度区/视觉不可信时施力),恒定 w 是 strawman。

**自检(idea §6 的"先验幅度↓ 是否与成功率↑ 同向")**:这里是**反向**——往先验拉 → 成功率↓。在视觉可信的场景这恰恰是应当发生的,说明实验没有自欺,只是 naïve 版本 + 错测试床注定负结果。

**因此:本轮证伪的是"恒定 w + demo_randomized",而 idea 从未声称过这个;真正的主张(门控/仅在 vision 不可信时施力)尚未被检验。**

---

## 4. 方法学教训

- step0/step1 用 dummy 零图,只验证了"先验存在 + 旋钮单调",**未验真实场景正确性**。若先做"真实图 + OOD 扰动"的离线实验,本可更早、更便宜地发现"无 OOD gap"问题,而非耗十余小时 sim。
- 单 rollout 受 cuRobo EE 规划支配(分钟级),sim 搜索昂贵;迭代应优先廉价的离线实验。

---

## 5. 下一步(候选)

- **A. 先确认 Hy-VLA 上是否存在 OOD gap**:把 RoboTwin 视觉扰动旋钮(`Messy Table / Random Background / Random Light / Random Camera Distance` + 干扰物)拉满,或上未见物体/指令,重测 w=1 基线。**w=1 明显掉才有戏;掉不动则说明 Hy-VLA 太鲁棒,需换更弱/未经 DR 训练的 base model。**
- **B. 换机制**:恒定 w → 门控/自适应 w(候选门信号 `‖v_full − v_masked‖`,但其有歧义:大值既可能"视觉有用"也可能"视觉是垃圾",需进一步设计);或转向 idea (b) latent drift / 流形投影(只清掉垂直于流形的分量,保留有用视觉)。
- **C. 廉价离线真实图实验**先行,再上 sim。
- **D. plan B**:在 vision 真会崩的弱 base model 上先证明现象存在,再回 Hy-VLA。
