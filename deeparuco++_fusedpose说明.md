# DeepArUco++ 融合位姿识别与渲染项目技术说明

本文档基于 `/home/zsyy/桌面/deeparuco++_fusedpose.py` 生成，用于说明该脚本项目的功能、运行环境、DeepArUco++ 检测后端、codebook 解码机制、融合位姿估计、渲染和调试方法。

## 1. 项目概述

`deeparuco++_fusedpose.py` 是一个集成 DeepArUco++ / OpenCV ArUco 双检测后端的刚体融合位姿估计脚本。它可以从摄像头或单张图片中检测 marker，将检测结果转换成 OpenCV ArUco 兼容格式，再根据预设的 3D 刚体布局计算统一位姿，并将模型投影渲染到画面中。

该脚本的主要特点：

- 支持 DeepArUco++ 神经网络检测后端。
- 支持 OpenCV ArUco 原生检测后端。
- DeepArUco++ 后端支持 `mip36h12` 和 `opencv` 两种 codebook 解码方式。
- 支持多 tag 融合 `solvePnP` 位姿估计。
- 支持姿态滤波、重投影误差拒绝和上一帧姿态保持。
- 支持 OpenCV 画面渲染和 Unity UDP 位姿输出。

默认模型是一个 9-tag 双层 cube 刚体：

- 顶面 tag：`0`
- 上层侧面 tag：`1,2,3,4`
- 下层侧面 tag：`5,6,7,8`
- 默认字典：`DICT_6X6_250`
- 默认 marker 边长：`0.032 m`
- 默认单 cube 边长：`0.04 m`

## 2. 与普通 DeepArUco 版本的主要区别

相比 `/home/zsyy/桌面/deeparuco_fusedpose.py`，该脚本主要增强点是 DeepArUco++ codebook 选择：

```text
--deeparuco-codebook mip36h12
--deeparuco-codebook opencv
```

其中：

- `mip36h12`：使用 DeepArUco++ 仓库内置的 MIP36h12 codebook，即 `impl.aruco.ids_as_bits`。
- `opencv`：使用 OpenCV ArUco 字典模板进行 bit 匹配，字典由 `--dictionary` 指定。

该机制使同一套神经网络检测/角点回归/解码流程可以服务于不同 marker 编码体系。

## 3. 主要功能

### 3.1 DeepArUco++ 检测后端

DeepArUco++ 后端的处理流程：

1. 使用 YOLO 模型检测 marker 外接框。
2. 对检测框做 20% padding。
3. 裁剪并缩放 marker 图像到 `64x64`。
4. 使用 Keras 角点回归模型精修角点。
5. 将角点排序。
6. 基于角点做透视规整，得到标准 marker 图像。
7. 使用 decoder 模型输出 bit 矩阵。
8. 根据 `--deeparuco-codebook` 选择解码模板。
9. 计算 bit 距离，得到 ID、旋转和距离。
10. 使用 `--deeparuco-threshold` 过滤低质量识别。
11. 转成 OpenCV ArUco 的角点顺序和 ID 格式。

### 3.2 OpenCV ArUco 后端

通过以下参数切换：

```bash
--detector-backend opencv
```

该模式直接调用 OpenCV：

```python
cv2.aruco.detectMarkers(...)
```

适合在不加载 DeepArUco++ 模型时快速验证相机、字典和几何配置。

### 3.3 Codebook 解码

DeepArUco++ 后端在 `create_deeparuco_backend_decoder()` 中根据参数选择解码方式：

```text
mip36h12 -> create_deeparuco_bit_decoder(...)
opencv   -> create_aruco_bit_decoder(...)
```

#### `mip36h12`

使用 DeepArUco++ 仓库 `impl/aruco.py` 中的 `ids_as_bits` 作为模板。适合识别 DeepArUco++ 自带的 MIP36h12 marker。

#### `opencv`

使用 OpenCV 生成指定 ArUco 字典下每个 ID 的标准 bit 模板，再与 decoder 输出的 bit 进行距离匹配。适合识别 OpenCV ArUco 字典 marker。

## 4. 运行环境

### 4.1 Python 解释器

脚本默认环境：

```text
/home/zsyy/anaconda3/envs/deeparuco39/bin/python
```

建议直接使用：

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/deeparuco++_fusedpose.py
```

### 4.2 基础依赖

- Python 3
- OpenCV Python
- NumPy

### 4.3 DeepArUco++ 后端依赖

- TensorFlow / Keras
- Ultralytics YOLO
- DeepArUco++ 仓库代码
- YOLO 检测模型 `.pt`
- 角点回归模型 `.h5`
- marker decoder 模型 `dec_new.h5`

默认仓库路径：

```text
/home/zsyy/下载/deeparuco-main
```

脚本会检查仓库内是否存在：

```text
impl/aruco.py
impl/heatmaps.py
impl/losses.py
impl/utils.py
```

默认模型文件：

```text
models/det_luma_bc_s.pt
models/reg_hmap_8.h5
models/dec_new.h5
```

## 5. 常用运行命令

### 5.1 默认 DeepArUco++ 摄像头识别

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/deeparuco++_fusedpose.py \
  --camera 0 \
  --calibration /home/zsyy/camera_calibration.yml
```

### 5.2 指定 MIP36h12 codebook

这是默认模式：

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/deeparuco++_fusedpose.py \
  --camera 0 \
  --calibration /home/zsyy/camera_calibration.yml \
  --deeparuco-codebook mip36h12
```

### 5.3 使用 OpenCV ArUco 字典 codebook 解码

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/deeparuco++_fusedpose.py \
  --camera 0 \
  --calibration /home/zsyy/camera_calibration.yml \
  --deeparuco-codebook opencv \
  --dictionary DICT_6X6_250
```

### 5.4 完全切换为 OpenCV ArUco 检测后端

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/deeparuco++_fusedpose.py \
  --camera 0 \
  --calibration /home/zsyy/camera_calibration.yml \
  --detector-backend opencv \
  --dictionary DICT_6X6_250
```

### 5.5 指定 DeepArUco++ 仓库和模型

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/deeparuco++_fusedpose.py \
  --camera 0 \
  --calibration /home/zsyy/camera_calibration.yml \
  --deeparuco-repo /home/zsyy/下载/deeparuco-main \
  --deeparuco-detector det_luma_bc_s \
  --deeparuco-regressor reg_hmap_8
```

### 5.6 调整性能

降低计算量：

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/deeparuco++_fusedpose.py \
  --camera 0 \
  --calibration /home/zsyy/camera_calibration.yml \
  --process-scale 0.5 \
  --process-every-n-frames 3
```

提高检测精度：

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/deeparuco++_fusedpose.py \
  --camera 0 \
  --calibration /home/zsyy/camera_calibration.yml \
  --process-scale 1.0 \
  --process-every-n-frames 1
```

### 5.7 降低单 tag 姿态翻转风险

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/deeparuco++_fusedpose.py \
  --camera 0 \
  --calibration /home/zsyy/camera_calibration.yml \
  --min-tags-for-pose 2 \
  --hold-last-seconds 0.5 \
  --max-stable-reprojection-error 5.0 \
  --print-pose
```

### 5.8 角点和模型调试

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/deeparuco++_fusedpose.py \
  --camera 0 \
  --calibration /home/zsyy/camera_calibration.yml \
  --draw-detected-corners \
  --draw-corner-index \
  --draw-model-corner-labels \
  --print-pose
```

### 5.9 单张图片处理

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/deeparuco++_fusedpose.py \
  --image /path/to/input.png \
  --output /path/to/output.png
```

### 5.10 Unity UDP 输出

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/deeparuco++_fusedpose.py \
  --camera 0 \
  --calibration /home/zsyy/camera_calibration.yml \
  --send-unity-pose \
  --unity-udp-host 127.0.0.1 \
  --unity-udp-port 5055
```

## 6. 关键参数说明

### 6.1 检测后端与 codebook

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--detector-backend` | `deeparuco` | 检测后端，可选 `deeparuco` 或 `opencv` |
| `--deeparuco-codebook` | `mip36h12` | DeepArUco++ 解码 codebook，可选 `mip36h12` 或 `opencv` |
| `--dictionary` | `DICT_6X6_250` | OpenCV ArUco 字典，在 OpenCV 后端或 `opencv` codebook 下使用 |
| `--deeparuco-id-map` | 空 | 将 DeepArUco++ 解码 ID 映射为物理 tag ID |

### 6.2 DeepArUco++ 模型参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--deeparuco-repo` | `/home/zsyy/下载/deeparuco-main` | DeepArUco++ 仓库根目录 |
| `--deeparuco-detector` | `det_luma_bc_s` | YOLO 检测模型名 |
| `--deeparuco-regressor` | `reg_hmap_8` | 角点回归模型名 |
| `--deeparuco-threshold` | `9.0` | bit 解码距离阈值 |
| `--deeparuco-detector-conf` | `0.03` | YOLO 置信度阈值 |
| `--deeparuco-detector-iou` | `0.5` | YOLO NMS IoU 阈值 |
| `--deeparuco-include-rejected` | 关闭 | 是否把未通过阈值的识别结果也送入位姿估计 |

### 6.3 摄像头与性能参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--camera` | `0` | 摄像头编号，也可用 `auto` |
| `--width` | `640` | 请求采集宽度 |
| `--height` | `480` | 请求采集高度 |
| `--fps` | `30` | 请求帧率 |
| `--camera-buffer` | `1` | 摄像头缓冲区大小 |
| `--autofocus` | `1` | 是否开启自动对焦 |
| `--process-scale` | `0.75` | DeepArUco++ 推理前图像缩放比例 |
| `--process-every-n-frames` | `2` | 每 N 帧执行一次 marker 推理 |

### 6.4 几何布局参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--marker-length` | `0.032` | marker 边长，单位米 |
| `--cube-size` | `0.04` | 单个 cube 边长，单位米 |
| `--vertical-gap` | `0.0` | 上下 cube 间隙 |
| `--upper-ids` | `1,2,3,4` | 上层侧面 ID |
| `--lower-ids` | `5,6,7,8` | 下层侧面 ID |
| `--top-id` | `0` | 顶面 ID |
| `--upper-rotation-deg` | `45.0` | 上层 cube 绕 Z 轴旋转 |
| `--lower-rotation-deg` | `0.0` | 下层 cube 绕 Z 轴旋转 |
| `--id-face-map` | `1:front,2:left,3:back,4:right,5:front,6:left,7:back,8:right,0:top` | ID 到物理面的映射 |
| `--id-rotation-map` | `0:90` | 每个 ID 的面内旋转 |
| `--corner-rolls` | 空 | 静态角点顺序修正 |

### 6.5 位姿稳定参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--min-tags-for-pose` | `1` | 最少参与位姿估计的 tag 数 |
| `--pose-filter` | `one_euro` | 姿态滤波器 |
| `--hold-last-seconds` | `0.25` | 检测丢失时保持上一帧姿态的时间 |
| `--max-stable-reprojection-error` | `8.0` | 平均重投影误差超过该值时拒绝更新 |
| `--reject-outlier-tags` | 开启 | 是否剔除重投影误差异常 tag |
| `--candidate-switch-hysteresis` | `0.18` | 候选 tag 集切换迟滞 |
| `--render-anchor-switch-ratio` | `1.35` | 渲染 anchor 切换门限 |

### 6.6 渲染参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--draw-model-tags` | 开启 | 绘制配置的模型 tag 面 |
| `--draw-prism-wireframe` | 关闭 | 绘制上下 cube 线框 |
| `--draw-detected-corners` | 关闭 | 绘制检测角点 |
| `--draw-corner-index` | 关闭 | 显示检测角点编号 |
| `--draw-model-corner-labels` | 关闭 | 显示模型角点编号 |
| `--render-model-rotation-deg` | `180.0` | 渲染模型绕 Z 轴旋转 |
| `--render-model-translation` | `-0.04,0,-0.004` | 渲染模型平移 |
| `--render-alignment-smoothing-alpha` | `0.25` | 渲染对齐偏移平滑 |

## 7. DeepArUco++ 数据流

```text
输入 BGR 图像
  ↓
可选 process-scale 缩放
  ↓
YOLO 检测 marker bbox
  ↓
bbox padding
  ↓
裁剪并 resize 到 64x64
  ↓
角点回归模型 refine_corners
  ↓
hmap_to_corners 或直接回归角点
  ↓
ordered_corners 排序
  ↓
marker_from_corners 透视规整
  ↓
decoder 输出 marker bit
  ↓
根据 deeparuco-codebook 选择模板
  ↓
find_id 得到 id / distance / rotation
  ↓
deeparuco-threshold 过滤
  ↓
deeparuco-id-map 映射物理 ID
  ↓
转换成 OpenCV ArUco 格式
```

## 8. 融合位姿数据流

```text
marker_corners + marker_ids
  ↓
id-face-map / id-rotation-map / corner-rolls
  ↓
生成刚体 3D tag 角点 layout
  ↓
collect_correspondences 生成 2D-3D 对应点
  ↓
build_stable_candidate_id_sets 生成候选 tag 集
  ↓
estimate_fused_pose 调用 solvePnP
  ↓
compute_tag_reprojection_errors 计算重投影误差
  ↓
refine_pose_by_tag_outliers 剔除异常 tag
  ↓
score_pose_candidate 选择最佳候选
  ↓
PoseTracker 滤波、拒绝和 hold
  ↓
draw_render_state 渲染
  ↓
可选 UDP 发送 Unity
```

## 9. 主要模块说明

### 9.1 `create_deeparuco_backend_decoder`

根据 `--deeparuco-codebook` 创建解码函数。

- `mip36h12` 使用 DeepArUco++ 仓库 codebook。
- `opencv` 使用 OpenCV ArUco 字典生成模板。

这是本脚本区别于普通 DeepArUco 版本的核心模块。

### 9.2 `load_deeparuco_backend`

加载 DeepArUco++ 仓库、YOLO 检测模型、Keras 角点回归模型和 decoder 模型，并返回后端字典。

### 9.3 `detect_deeparuco`

执行 DeepArUco++ 检测、角点回归、marker 规整、decoder 解码和结果封装。

### 9.4 `detect_deeparuco_scaled`

在缩放后的帧上执行检测，再将 bbox 和角点坐标映射回原始图像。

### 9.5 `deeparuco_results_to_aruco`

将 DeepArUco++ 结果转换为 OpenCV ArUco 标准格式，包括角点顺序、ID 映射和 rejected 结果。

### 9.6 `PoseTracker`

负责姿态滤波、重投影误差拒绝、上一帧保持和候选状态记录。

### 9.7 `FrameProcessScheduler`

控制每隔 N 帧执行一次 DeepArUco++ 推理。

## 10. 常见问题与调试

### 10.1 codebook 选错导致 ID 不对

如果使用的是 DeepArUco++ MIP36h12 marker，应使用：

```bash
--deeparuco-codebook mip36h12
```

如果使用的是普通 OpenCV ArUco marker，应使用：

```bash
--deeparuco-codebook opencv --dictionary DICT_6X6_250
```

如果 ID 仍然与实物编号不一致，使用：

```bash
--deeparuco-id-map 解码ID:物理ID
```

例如：

```bash
--deeparuco-id-map 23:0,16:1,18:2
```

### 10.2 DeepArUco++ 仓库或模型找不到

检查仓库：

```bash
ls /home/zsyy/下载/deeparuco-main
```

检查模型：

```bash
ls /home/zsyy/下载/deeparuco-main/models
```

必要时指定：

```bash
--deeparuco-repo /实际/路径/deeparuco-main
```

### 10.3 速度慢

降低推理负载：

```bash
--process-scale 0.5 --process-every-n-frames 3
```

需要更高精度时：

```bash
--process-scale 1.0 --process-every-n-frames 1
```

### 10.4 模型突然翻到 tag 外侧

常见原因：

- 只使用了单个平面 tag。
- 当前帧 DeepArUco++ 角点回归抖动。
- 解码 ID 或 ID 映射错误。
- 某个 tag 的面映射或面内旋转不正确。

建议：

```bash
--min-tags-for-pose 2 \
--hold-last-seconds 0.5 \
--max-stable-reprojection-error 5.0 \
--print-pose
```

### 10.5 模型与实物不重合

优先检查：

- `--marker-length`
- `--cube-size`
- `--vertical-gap`
- `--id-face-map`
- `--id-rotation-map`
- `--corner-rolls`
- `--deeparuco-id-map`
- 相机标定文件

建议打开：

```bash
--draw-detected-corners --draw-corner-index --draw-model-corner-labels
```

### 10.6 标定文件读取失败

建议使用纯英文路径：

```text
/home/zsyy/camera_calibration.yml
```

运行时：

```bash
--calibration /home/zsyy/camera_calibration.yml
```

## 11. DeepArUco++、DeepArUco 和 OpenCV 后端对比

| 项目 | DeepArUco++ | DeepArUco | OpenCV ArUco |
|---|---|---|---|
| 检测方式 | YOLO + 角点回归 + decoder | YOLO + 角点回归 + decoder | OpenCV 图像处理 |
| codebook | `mip36h12` 或 `opencv` | 主要使用仓库内置 bits | OpenCV 字典 |
| 模型依赖 | 高 | 高 | 低 |
| 速度 | 较慢 | 较慢 | 较快 |
| 调试难度 | 高 | 中高 | 低 |
| 适用场景 | 需要 DeepArUco++ codebook 或 OpenCV codebook 混用 | DeepArUco marker | 标准 ArUco marker |

## 12. Unity 集成

脚本可发送 UDP JSON：

```json
{
  "valid": true,
  "timestamp": 12345.678,
  "rvec": [0.1, 0.2, 0.3],
  "tvec": [0.01, 0.02, 0.5],
  "used_ids": [1, 2],
  "mean_error": 1.2,
  "pose_state": "measured"
}
```

Python 端参数：

```bash
--send-unity-pose --unity-udp-host 127.0.0.1 --unity-udp-port 5055
```

Unity 端需要：

1. 监听 UDP 端口。
2. 解析 JSON。
3. 将 OpenCV `rvec/tvec` 转换到 Unity 坐标系。
4. 驱动 Cube、STL 或其他模型。

## 13. 后续改进方向

建议改进：

1. 增加 DeepArUco++ 检测可视化，包括 bbox、dist、rotation、accepted 状态。
2. 对单 tag 使用 `solvePnPGeneric + SOLVEPNP_IPPE_SQUARE`，选择与上一帧更接近的解。
3. 增加姿态跳变拒绝阈值，避免错误帧进入渲染。
4. 增加 `--render-alignment-mode center|edge`。
5. 将检测、解码、PnP、渲染、Unity 输出拆分为独立模块。
6. 增加异常帧保存，记录图像、used ids、mean_err、codebook 和 decoder distance。

## 14. 总结

`deeparuco++_fusedpose.py` 是一个面向 DeepArUco++ marker 和 OpenCV ArUco marker 的融合位姿估计脚本。它的核心价值在于：使用神经网络检测和角点回归增强 marker 检测能力，同时支持不同 codebook 解码，并复用成熟的多 tag 融合 PnP、姿态滤波和渲染流程。

项目稳定性主要依赖：

1. DeepArUco++ 仓库和模型文件是否正确。
2. `--deeparuco-codebook` 是否匹配实际 marker 编码。
3. `--deeparuco-id-map` 是否正确映射到物理 tag ID。
4. 相机标定是否准确。
5. 几何布局参数是否与实物一致。
6. 是否避免单 tag 独立驱动姿态。

调试时应重点观察：`used ids`、`mean_err`、rejected 数量、DeepArUco++ 解码 ID、codebook 类型、角点编号和模型角点编号。
