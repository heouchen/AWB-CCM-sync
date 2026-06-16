# AWB/CCM Sync 数据契约

本文定义参考实现期望的数据形态。工程落地时可以使用 CSV、Parquet、JSONL 或数据库表，但字段语义应保持一致。

## 1. 色块与灰块统计表

每行表示一个相机、一个光源、一个 ROI 的线性 RGB 统计值。

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `illuminant_id` | string | 是 | 单次光源采集的唯一 ID，主副摄同一光源必须一致。 |
| `camera_id` | string | 是 | 相机 ID，例如 `main`、`sub` 或具体模组名。 |
| `sample_type` | string | 是 | `grey` 或 `color_checker`。 |
| `patch_id` | string | 是 | 灰块 ID 或 24 色卡块 ID。 |
| `r` / `g` / `b` | float | 是 | 扣黑、线性化、LSC/通道顺序一致后的 RGB ROI 统计值。 |
| `weight` | float | 否 | 样本权重；真实数据通常高于虚拟数据。 |
| `source` | string | 否 | `real` 或 `virtual`。 |
| `exposure_time_us` | float | 否 | 曝光时间，用于排查主副摄曝光比例和异常帧。 |
| `analog_gain` / `digital_gain` | float | 否 | 增益信息，用于数据清洗。 |
| `illuminant_label` | string | 否 | 例如 `A`、`TL84`、`D65`、`LED_mix`。 |
| `cct` | float | 否 | 色温标签，不能替代光源唯一 ID。 |
| `data_version` | string | 否 | 标定数据、色卡、光谱或 ISP 前处理版本。 |

硬性约束：

- `r/g/b` 必须来自扣黑后的线性域，且 `g > 0`。
- 同一个 `illuminant_id` 下，主副摄应有同一组 `patch_id`。
- AWB sync 至少需要灰块主副摄配对；CCM sync 需要每个光源下的 24 色卡主副摄配对。
- 饱和、欠曝、反光、阴影、ROI 定位异常的数据应在进入训练前剔除。

## 2. 训练中间量

AWB sync 训练样本：

```text
main_white_point = (main_grey.r / main_grey.g, main_grey.b / main_grey.g)
sub_white_point  = (sub_grey.r  / sub_grey.g,  sub_grey.b  / sub_grey.g)
```

CCM sync 单光源训练样本：

```text
main_patch = (main_patch.r / main_patch.g, 1, main_patch.b / main_patch.g)
sub_patch  = (sub_patch.r  / sub_patch.g,  1, sub_patch.b  / sub_patch.g)
```

每个 `illuminant_id` 先拟合一个局部 `M_j`，再使用 `{sub_white_point_j, M_j}` 拟合白点条件化的 `M(p2)`。

## 3. 模型导出建议

导出参数至少应包含：

- `awb_sync`：`f_grey_rg` 和 `f_grey_bg` 的多项式阶数、系数、输入范围。
- `ccm_sync`：`m11/m13/m21/m23/m31/m33` 六个系数面的多项式阶数、系数、输入范围。
- `guards`：白点裁剪范围、`M * AWB2` 条件数阈值、正则化参数、fallback CCM。
- `metadata`：训练光源列表、真实/虚拟数据权重、色卡版本、光谱版本、模组或三刺激值版本、导出时间和模型版本。

参考实现中的 `AWBSyncModel.to_dict()` 与 `CCMSyncModel.to_dict()` 可作为轻量 JSON 导出的基础格式。
