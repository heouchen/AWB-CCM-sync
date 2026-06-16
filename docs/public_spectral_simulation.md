# 公开光谱数据仿真说明

`scripts/run_spectral_simulation.py` 使用公开光谱数据构建虚拟主副摄采集链路，比手写 RGB 合成数据更接近真实物理过程。

## 数据来源

数据通过 `colour-science` Python 包读取，不在运行时下载网络文件。`colour-science` 是 BSD-3-Clause 许可的开源色彩科学库，包含常用色彩科学算法和数据集。

当前仿真使用：

- 光源 SPD：`colour.SDS_ILLUMINANTS`，包括 CIE A/B/C/D/E、荧光灯 FL 系列、高压放电灯 HP 系列、典型 LED、ISO 7589 光源等。
- 色卡反射率：`colour.SDS_COLOURCHECKERS["BabelColor Average"]`，对应 BabelColor ColorChecker 平均光谱数据。
- 相机光谱响应：`colour.MSDS_CAMERA_SENSITIVITIES` 中的 `Nikon 5100 (NPL)` 和 `Sigma SDMerill (NPL)`，来源于 Darrodi、Finlayson、Goodman 和 Mackiewicz 的 NPL 相机光谱灵敏度参考数据。

代码统一使用 400-700 nm、10 nm 间隔，并对插值产生的极小负值做 0 裁剪。

## 虚拟采集模型

对每个光源、色块、相机通道计算：

```math
X_{i,c,l,q}
=
\sum_{\lambda}
E_l(\lambda)
\rho_c(\lambda)
S_{i,q}(\lambda)
\Delta\lambda
```

其中：

- `E_l(lambda)`：光源 SPD。
- `rho_c(lambda)`：ColorChecker 色块反射率。
- `S_i,q(lambda)`：相机通道光谱响应。
- `Delta lambda`：采样间隔。

每个光源和相机使用 `neutral 8 (.23 D)` 灰块的 G 通道做曝光归一化。这样 AWB/CCM 评估集中在色度一致性，而不是 AE 亮度尺度。

## 训练与验证拆分

默认保留 8 个 hold-out 光源验证：

```text
D55, D75, FL4, FL10, LED-B2, LED-V1, HP3, ISO 7589 Photoflood
```

`colour.SDS_ILLUMINANTS` 中其余 51 个光源用于训练 AWB sync 和 CCM sync。

## 运行

```bash
python3 scripts/run_spectral_simulation.py
```

可调参数：

```bash
python3 scripts/run_spectral_simulation.py --noise-std 0.0005 --l2 1e-6 --awb-degree 2 --ccm-degree 2
```

输出指标包括：

- `awb_naive_mean_error`：直接把主摄白点当副摄白点的平均误差。
- `awb_sync_mean_error` / `awb_sync_p95_error`：AWB sync 在 hold-out 光源上的白点误差。
- `chroma_rmse_*`：G 归一化色度 RMSE。
- `color_rmse_*`：在 G 归一化输入上通过 AWB/CCM 后的相对 RGB RMSE。
- `max_condition_number`：验证集中 `M * AWB2` 的最大条件数。

## 当前限制

- 数据来自公开光谱表，不是目标手机模组实测数据。
- 只模拟理想兰伯特反射，没有镜头 shading、flare、IR 泄漏、传感器串扰、噪声模型、压缩或 ISP 非线性。
- 当前只使用两台公开 DSLR 相机响应模拟主副摄差异，不能代表具体手机摄像头。
- 只验证色卡和灰块，不覆盖真实场景的空间结构、混合光、局部光照和切摄时序。
