# H20 workspace — RoboTwin/SAPIEN Vulkan 失败复盘 + 复现流程

> 目的:记录在 coder-workspace(H20)容器上把 RoboTwin 装通、但 **SAPIEN 渲染因 Vulkan 不可用而失败** 的全过程、路径、诊断结论,以及一套可在新窗口重跑的复现/排查流程。
> 结论先行:**NVIDIA Vulkan 在这台 H20 容器里建不出实例(`vkCreateInstance` → `ERROR_INCOMPATIBLE_DRIVER`),用户态拼图全对也没用,容器内修不了。** 已改用 A100 机器 `fnii-vla2`(渲染正常)。

---

## 1. 环境(H20 那台)

- host: `coder-workspace-brown-panther-85`,user `jovyan`
- 代码:`/home/jovyan/code/wge/ttt_drift`;RoboTwin:`/home/jovyan/code/wge/RoboTwin`
- 离线实验(step0/step1)环境:uv,Python 3.11
- RoboTwin 环境:conda env `RoboTwin`,Python 3.10,torch 2.4.1+cu121
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

## 3. 真正的 blocker:Vulkan 建不出实例

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

## 4. 为什么排除了所有常见原因(诊断 + 观测结论)

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

**结论:用户态(驱动版本、设备节点、DRM 节点、ICD、能力位)全部正确,NVIDIA 驱动本身就是建不出 Vulkan 实例。** 最可能是 **H20 在固件/hypervisor 层禁用了图形/Vulkan**,或该容器里开源内核模块对 H20 的 Vulkan 支持不全。非 pip/apt/json 能解决。

---

## 5. 复现流程(新窗口照此重跑,确认/继续排查根因)

### 5.1 复现失败(应当稳定复现)
```
conda activate RoboTwin
export XDG_RUNTIME_DIR=/tmp/xdg-root && mkdir -p $XDG_RUNTIME_DIR
python - <<'PY'
import sapien
sapien.set_log_level("info")
s = sapien.Scene(); s.add_ground(-1); s.set_ambient_light([0.5,0.5,0.5])
c = s.add_camera("c",128,128,1.0,0.01,100); s.update_render(); c.take_picture()
print("RENDER OK", c.get_picture("Color").shape)
PY
```
预期:`RuntimeError: failed to find a rendering device`。

### 5.2 拿底层 Vulkan 错误
```
sudo apt-get install -y vulkan-tools
vulkaninfo --summary 2>&1 | head -40
```
预期:`Could not get 'vkCreateInstance' ... libGLX_nvidia.so.0` / `Found no drivers!` / `ERROR_INCOMPATIBLE_DRIVER`。

### 5.3 全套诊断(确认用户态都正常)
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

### 5.4 还想往根因挖,可继续试的方向
- **完整 loader 跟踪**(看协商在哪断、是否 dlopen 了次级库失败):
```
VK_LOADER_DEBUG=all VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json vulkaninfo 2>&1 | sed -n '1,120p'
```
- **看 GPU 是否报告图形能力 / 有无错误**:`nvidia-smi -q | grep -iE "Graphics|ECC|Fabric|Persistence"`;`dmesg 2>/dev/null | grep -i nvidia | tail -40`(可能无权限)。
- **换一块 GPU 试**:`CUDA_VISIBLE_DEVICES=1 vulkaninfo --summary`(各卡逐一)。
- **更新 Vulkan loader**:系统 `libvulkan1` 是 1.3.275,可试装更新的 loader 再协商。
- **最终判定手段**:同样的 5.1 在一台**普通直通 GPU(A100/4090 等)**上能 `RENDER OK`(已在 `fnii-vla2` 验证),则坐实是 **H20 这台容器/固件**的问题,而非代码或 RoboTwin。

### 5.5 给平台方的诉求(若要他们修)
> H20 直通容器内 NVIDIA Vulkan 无法初始化:`vkCreateInstance` 返回 `ERROR_INCOMPATIBLE_DRIVER`(driver 570.124.06,kernel=userspace 一致,`/dev/nvidia*`、`/dev/nvidia-modeset`、`/dev/dri/renderD*` 均在,`NVIDIA_DRIVER_CAPABILITIES=all`,Pass-Through)。CUDA 正常但 Vulkan 离屏渲染不可用。请确认 H20 是否在固件/驱动层禁用了图形,或提供 Vulkan 可用的镜像/节点(SAPIEN/RoboTwin 仿真渲染需要)。

---

## 6. 可用替代:A100 机器 `fnii-vla2`(GPU01)

同样的 5.1 在该机 `RENDER OK`。RoboTwin eval 已在该机跑通(详见 `SERVER_SETUP.md` / 项目记忆)。结论:**问题定位在 H20 容器的 Vulkan,不在代码。**
