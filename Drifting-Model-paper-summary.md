# Generative Modeling via Drifting — 论文摘录与笔记

> Deng, Mingyang; Li, He; Li, Tianhong; Du, Yilun; **He, Kaiming**. MIT / Harvard. arXiv:2602.04770 (Feb 2026).
>
> - Paper: http://arxiv.org/abs/2602.04770
> - Code (JAX, ImageNet): https://github.com/lambertae/drifting
> - Colab demo (toy 2D): https://colab.research.google.com/github/lambertae/lambertae.github.io/blob/main/projects/drifting/notebooks/drifting_model_demo.ipynb
> - 项目页: https://lambertae.github.io/projects/drifting/
>
> 说明:sandbox 网络封了 arXiv PDF 与 GitHub 二进制下载(curl exit 56 / git 403),所以这里保存的是可读摘录 + 全部链接;PDF 和代码需在本机手动拉取。

## TL;DR(官方)
- **新范式**:Drifting Models 在**训练过程中**演化模型的 pushforward 分布(而不是像 diffusion/flow 那样在推理时迭代)。因此推理是**单步前向(1-NFE)**。
- **Drifting field V**:引入一个"漂移场" V 来移动样本;设计上当生成分布 q 匹配数据分布 p 时 **V 归零(达到平衡 equilibrium)**。
- **算法**:从 mini-batch 估计 V → 构造一个"drifted target"(漂移后的目标)→ 回归网络去匹配它。
- **灵活性**:兼容 representation-space loss 与 classifier-free guidance。
- **性能**:ImageNet 256×256,1-NFE FID **1.54**(latent)/ **1.61**(pixel),SOTA。

## 核心算法(来自官方 demo notebook)
整个方法的"灵魂"就三行:

```python
def drifting_loss(gen, pos, compute_drift):
    V = compute_drift(gen, pos)         # 漂移场:把生成样本 gen 往数据样本 pos 的方向推
    target = (gen + V).detach()         # 漂移后的目标(stop-gradient)
    return F.mse_loss(gen, target)      # 回归网络输出去匹配漂移目标
```

关键点:
- `gen` 是当前网络生成的样本,`pos` 是真实数据样本(positive)。
- `compute_drift` 估计一个把 q 推向 p 的速度场 V(论文用 mini-batch 上的核/配对估计)。
- `(gen + V).detach()` 是一个"比现在好一点点"的自举目标;网络反复回归它,**分布在训练步上被一点点推到数据分布**——这就是"drift"。
- 平衡条件:q==p 时 V==0,target==gen,loss==0,训练自然停。

## 与本项目(ttt_drift)的关系
- Kaiming 的 drift 是**训练时**演化整个分布;**不是**推理时方法,也没有 reward/OOD 的概念。
- 我们借的是它的**思想骨架**:一个"当且仅当落在目标分布上时归零的漂移场 V",用 V 当作把样本拉回流形的力。
- 我们的迁移:把"数据分布 p"换成 **language-only 的 action 流形**,把"训练时演化"换成 **测试时对单个 OOD 样本做几步 drift**,且**无需 0/1 reward**——力直接来自"当前 vision-conditioned action 偏离 language 流形多远"。

## Abstract(原文)
Generative modeling can be formulated as learning a mapping *f* such that its pushforward distribution matches the data distribution. The pushforward behavior can be carried out iteratively at inference time, e.g., in diffusion/flow-based models. In this paper, we propose a new paradigm called *Drifting Models*, which evolve the pushforward distribution during training and naturally admit one-step inference. We introduce a drifting field that governs the sample movement and achieves equilibrium when the distributions match. This leads to a training objective that allows the neural network optimizer to evolve the distribution. On ImageNet 256×256, the one-step generator reaches FID 1.54 (latent) and 1.61 (pixel).

## Citation
```
@article{deng2026drifting,
  title={Generative Modeling via Drifting},
  author={Deng, Mingyang and Li, He and Li, Tianhong and Du, Yilun and He, Kaiming},
  journal={arXiv preprint arXiv:2602.04770},
  year={2026}
}
```
