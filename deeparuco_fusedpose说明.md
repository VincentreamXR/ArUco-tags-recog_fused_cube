# DeepArUco 融合位姿识别与渲染项目技术说明

本文档基于 `/home/zsyy/桌面/deeparuco_fusedpose.py` 生成，用于说明该脚本项目的功能、运行环境、检测后端、位姿估计流程、关键参数和调试方法。

## 1. 项目概述

`deeparuco_fusedpose.py` 是一个基于 DeepArUco / OpenCV ArUco 的刚体位姿估计与实时渲染脚本。脚本可从摄像头或图片中检测多个 marker，将检测结果转换为 OpenCV ArUco 格式，再根据预设的三维刚体布局计算融合位姿，并把模型投影到视频画面中。

该脚本相比普通 OpenCV ArUco 版本增加了 DeepArUco 检测后端：

- 使用 YOLO 模型检测 marker 区域。
- 使用 TensorFlow/Keras 角点回归模型精修角点。
- 使用解码模型识别 marker ID。
- 将 DeepArUco 输出转换为 OpenCV ArUco 的 `marker_corners` / `marker_ids` 格式。
- 后续位姿估计、滤波、渲染流程与 ArUco 融合位姿逻辑共用。

默认场景是一个 9-tag 双层 cube 刚体：

- 顶面 tag：`0`
- 上层侧面 tag：`1,2,3,4`
- 下层侧面 tag：`5,6,7,8`
- 默认字典：`DICT_6X6_250`
- 默认 marker 边长：`0.032 m`
- 默认单 cube 边长：`0.04 m`

## 2. 项目功能

### 2.1 双检测后端

脚本支持两种 marker 检测方式：

```text
--detector-backend deeparuco
--detector-backend opencv
```

默认使用 `deeparuco`。

#### DeepArUco 后端

DeepArUco 后端流程：

1. 使用 YOLO 模型检测 marker 外接框。
2. 对外接框扩大一定 padding。
3. 裁剪 marker 图像并缩放到 `64x64`。
4. 使用角点回归模型预测四个角点。
5. 对 marker 图像做透视归一化。
6. 使用解码模型得到 marker bit。
7. 与字典模板匹配，得到 ID、旋转和距离。
8. 根据阈值决定是否接受该检测。
9. 转换为 OpenCV ArUco 角点顺序。

#### OpenCV ArUco 后端

OpenCV 后端直接使用：

```python
cv2.aruco.detectMarkers(...)
```

该模式不需要 DeepArUco 模型文件，适合对比调试。

### 2.2 多 tag 融合位姿估计

脚本会将多个可见 tag 的 2D 图像角点和预设 3D 物体角点组合起来，通过 `cv2.solvePnP` 估计一个统一刚体姿态。

核心流程：

1. 检测 marker。
2. 转换 marker ID 和角点顺序。
3. 根据 `id-face-map` 找到 marker 在刚体上的面。
4. 生成 3D 角点。
5. 收集 2D-3D 对应点。
6. 生成候选 tag 集合。
7. 对候选调用 `solvePnP`。
8. 计算重投影误差。
9. 根据误差、面积质量、上一帧连续性和 top tag 约束选择最佳姿态。
10. 输出平滑后的 `rvec/tvec`。

### 2.3 实时异步处理

摄像头模式中脚本使用异步架构：

- 采集线程：持续读取摄像头最新画面。
- 处理线程：按设置的频率运行 marker 检测和位姿估计。
- 显示线程：显示最新摄像头帧，并叠加最近一次处理结果。

该架构通过 `LatestFrameSlot` 保存最新帧，避免旧帧堆积。

### 2.4 跳帧与缩放推理

DeepArUco 模型推理成本较高，脚本提供两个性能参数：

```text
--process-scale
--process-every-n-frames
```

- `--process-scale`：先缩小图像再做 DeepArUco 检测，然后把角点映射回原图坐标。
- `--process-every-n-frames`：每 N 帧处理一次 marker，其余帧只显示最近渲染状态。

默认：

```text
--process-scale 0.75
--process-every-n-frames 2
```

### 2.5 姿态滤波和保持

脚本提供姿态稳定机制：

- `ema` 滤波
- `one_euro` 滤波
- 重投影误差拒绝
- 短时间检测丢失时保持上一帧姿态
- 候选 tag 集切换迟滞
- 渲染 anchor 切换迟滞

常用参数：

```text
--pose-filter
--hold-last-seconds
--max-stable-reprojection-error
--candidate-switch-hysteresis
--render-anchor-switch-ratio
```

### 2.6 OpenCV 画面渲染

脚本使用 OpenCV 直接在图像上绘制：

- 检测到的 marker 边框。
- 检测角点和角点编号。
- 模型 tag 面。
- cube 线框。
- 坐标轴。
- 顶部状态信息面板。

### 2.7 Unity UDP 输出

脚本支持通过 UDP 向 Unity 发送最新姿态：

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

相关参数：

```text
--send-unity-pose
--unity-udp-host
--unity-udp-port
```

## 3. 运行环境

### 3.1 Python 环境

脚本指定的 Python 环境：

```text
/home/zsyy/anaconda3/envs/deeparuco39/bin/python
```

建议使用该解释器运行：

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/deeparuco_fusedpose.py
```

### 3.2 主要依赖

基础依赖：

- Python 3
- OpenCV Python
- NumPy

DeepArUco 后端额外依赖：

- TensorFlow / Keras
- Ultralytics YOLO
- DeepArUco 仓库代码
- DeepArUco 模型文件

DeepArUco 默认仓库路径：

```text
/home/zsyy/下载/deeparuco-main
```

脚本会检查以下文件是否存在：

```text
impl/aruco.py
impl/heatmaps.py
impl/losses.py
impl/utils.py
```

模型默认位于：

```text
/home/zsyy/下载/deeparuco-main/models
```

默认模型：

```text
det_luma_bc_s.pt
reg_hmap_8.h5
dec_new.h5
```

## 4. 常用运行命令

### 4.1 使用 DeepArUco 后端打开摄像头

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/deeparuco_fusedpose.py \
  --camera 0 \
  --calibration /home/zsyy/camera_calibration.yml
```

### 4.2 指定 DeepArUco 仓库

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/deeparuco_fusedpose.py \
  --camera 0 \
  --calibration /home/zsyy/camera_calibration.yml \
  --deeparuco-repo /home/zsyy/下载/deeparuco-main
```

### 4.3 使用 OpenCV ArUco 后端

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/deeparuco_fusedpose.py \
  --camera 0 \
  --calibration /home/zsyy/camera_calibration.yml \
  --detector-backend opencv
```

### 4.4 调整 DeepArUco 推理速度

降低分辨率、减少处理帧数：

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/deeparuco_fusedpose.py \
  --camera 0 \
  --calibration /home/zsyy/camera_calibration.yml \
  --process-scale 0.5 \
  --process-every-n-frames 3
```

提高精度但增加计算量：

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/deeparuco_fusedpose.py \
  --camera 0 \
  --calibration /home/zsyy/camera_calibration.yml \
  --process-scale 1.0 \
  --process-every-n-frames 1
```

### 4.5 避免单 tag 姿态翻转

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/deeparuco_fusedpose.py \
  --camera 0 \
  --calibration /home/zsyy/camera_calibration.yml \
  --min-tags-for-pose 2 \
  --hold-last-seconds 0.5 \
  --max-stable-reprojection-error 5.0 \
  --print-pose
```

### 4.6 显示检测角点和模型角点

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/deeparuco_fusedpose.py \
  --camera 0 \
  --calibration /home/zsyy/camera_calibration.yml \
  --draw-detected-corners \
  --draw-corner-index \
  --draw-model-corner-labels \
  --print-pose
```

### 4.7 向 Unity 发送姿态

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/deeparuco_fusedpose.py \
  --camera 0 \
  --calibration /home/zsyy/camera_calibration.yml \
  --send-unity-pose \
  --unity-udp-host 127.0.0.1 \
  --unity-udp-port 5055
```

### 4.8 单张图片处理

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/deeparuco_fusedpose.py \
  --image /path/to/input.png \
  --output /path/to/output.png
```

## 5. 核心参数说明

### 5.1 检测后端参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--detector-backend` | `deeparuco` | 检测后端，可选 `deeparuco` 或 `opencv` |
| `--deeparuco-repo` | `/home/zsyy/下载/deeparuco-main` | DeepArUco 仓库路径 |
| `--deeparuco-detector` | `det_luma_bc_s` | YOLO 检测模型名 |
| `--deeparuco-regressor` | `reg_hmap_8` | 角点回归模型名 |
| `--deeparuco-threshold` | `9.0` | 解码距离阈值，大于等于该值会被拒绝 |
| `--deeparuco-detector-conf` | `0.03` | YOLO 置信度阈值 |
| `--deeparuco-detector-iou` | `0.5` | YOLO NMS IoU 阈值 |
| `--deeparuco-include-rejected` | 关闭 | 是否把低质量解码结果也送入位姿流程 |
| `--deeparuco-id-map` | 空 | 将 DeepArUco 解码 ID 映射到物理 tag ID |

### 5.2 性能参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--process-scale` | `0.75` | DeepArUco 推理前图像缩放比例 |
| `--process-every-n-frames` | `2` | 每 N 帧执行一次 marker 推理 |
| `--width` | `640` | 摄像头请求宽度 |
| `--height` | `480` | 摄像头请求高度 |
| `--fps` | `30` | 摄像头请求帧率 |
| `--camera-buffer` | `1` | 摄像头缓冲区大小 |

### 5.3 几何布局参数

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
| `--id-face-map` | `1:front,2:left,...` | ID 到物理面的映射 |
| `--id-rotation-map` | `0:90` | 每个 ID 的面内旋转 |
| `--corner-rolls` | 空 | 固定角点顺序修正 |

### 5.4 位姿稳定参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--min-tags-for-pose` | `1` | 最少参与位姿估计的 tag 数 |
| `--pose-filter` | `one_euro` | 姿态滤波器 |
| `--hold-last-seconds` | `0.25` | 检测丢失时保持上一帧姿态的时间 |
| `--max-stable-reprojection-error` | `8.0` | 平均重投影误差超过该值时拒绝更新 |
| `--reject-outlier-tags` | 开启 | 是否剔除重投影误差异常 tag |
| `--candidate-switch-hysteresis` | `0.18` | 候选 tag 集切换迟滞 |
| `--render-anchor-switch-ratio` | `1.35` | 渲染 anchor 切换门限 |

### 5.5 渲染参数

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

## 6. DeepArUco 后端数据流

```text
输入 BGR 图像
  ↓
YOLO 检测 marker 框
  ↓
扩大框并裁剪
  ↓
缩放到 64x64
  ↓
角点回归模型输出角点
  ↓
ordered_corners 排序
  ↓
marker_from_corners 透视规整
  ↓
decoder 模型输出 bit
  ↓
模板匹配得到 decoded id / rotation / distance
  ↓
按 deeparuco-threshold 接受或拒绝
  ↓
转换为 OpenCV ArUco marker_corners / marker_ids
```

转换后，DeepArUco 和 OpenCV ArUco 两个后端共用同一套融合位姿逻辑。

## 7. 位姿估计数据流

```text
marker_corners + marker_ids
  ↓
根据 id-face-map 建立 3D 角点
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
PoseTracker 滤波和保持
  ↓
draw_render_state 渲染
  ↓
可选 UDP 发送 Unity
```

## 8. 主要模块说明

### 8.1 `FrameProcessScheduler`

控制每隔多少帧执行一次检测。用于降低 DeepArUco 推理压力。

### 8.2 `load_deeparuco_backend`

加载 DeepArUco 运行时依赖：

- TensorFlow
- DeepArUco 内部模块
- YOLO
- 检测模型 `.pt`
- 角点回归模型 `.h5`
- 解码模型 `dec_new.h5`

返回一个后端字典，供检测函数使用。

### 8.3 `detect_deeparuco`

完整执行 DeepArUco 检测、角点回归、marker 解码和结果封装。

### 8.4 `detect_deeparuco_scaled`

在低分辨率图像上执行 DeepArUco 检测，再将 bbox 和角点恢复到原图尺度。

### 8.5 `deeparuco_results_to_aruco`

将 DeepArUco 输出转换为 OpenCV ArUco 兼容格式。该函数还处理：

- 解码 ID 到物理 ID 的映射。
- 角点顺序转换。
- rejected 结果收集。

### 8.6 `PoseTracker`

负责姿态滤波、低置信度平滑、上一帧保持和错误姿态拒绝。

### 8.7 `PoseRenderState`

保存一帧处理后的渲染状态，包括检测结果、位姿、使用的 tag、渲染 anchor、文字状态和性能信息。

## 9. 常见问题与调试

### 9.1 DeepArUco 仓库找不到

报错类似：

```text
DeepArUco repo not found
```

检查：

```bash
ls /home/zsyy/下载/deeparuco-main
```

或通过参数指定：

```bash
--deeparuco-repo /实际/路径/deeparuco-main
```

### 9.2 DeepArUco 模型文件找不到

脚本需要：

```text
models/det_luma_bc_s.pt
models/reg_hmap_8.h5
models/dec_new.h5
```

如果使用其他模型名，需要通过：

```text
--deeparuco-detector
--deeparuco-regressor
```

指定。

### 9.3 DeepArUco 识别 ID 和物理 ID 不一致

使用：

```bash
--deeparuco-id-map 23:0,16:1,18:2
```

格式为：

```text
解码ID:物理ID
```

如果不设置，脚本会直接使用解码出的 ID。

### 9.4 检测速度慢或画面卡顿

优先调：

```text
--process-scale 0.5
--process-every-n-frames 3
```

如果需要更高精度：

```text
--process-scale 1.0
--process-every-n-frames 1
```

但计算量会明显增加。

### 9.5 模型偶尔翻到 tag 外侧

常见原因：

- 当前帧只使用了单个平面 tag。
- DeepArUco 角点回归抖动。
- 解码 ID 映射错误。
- 某个 tag 的 `id-face-map` 或 `id-rotation-map` 不正确。

建议：

```bash
--min-tags-for-pose 2 \
--hold-last-seconds 0.5 \
--max-stable-reprojection-error 5.0 \
--print-pose
```

### 9.6 模型和实物不重合

重点检查：

- `--marker-length`
- `--cube-size`
- `--vertical-gap`
- `--id-face-map`
- `--id-rotation-map`
- `--corner-rolls`
- 相机标定文件
- DeepArUco ID 映射

建议打开角点调试：

```bash
--draw-detected-corners --draw-corner-index --draw-model-corner-labels
```

### 9.7 标定文件读取失败

建议将标定文件放在纯英文路径：

```text
/home/zsyy/camera_calibration.yml
```

并使用：

```bash
--calibration /home/zsyy/camera_calibration.yml
```

避免中文路径或括号导致 OpenCV `FileStorage` 读取异常。

## 10. OpenCV 后端与 DeepArUco 后端对比

| 项目 | DeepArUco | OpenCV ArUco |
|---|---|---|
| 默认状态 | 默认启用 | 需指定 `--detector-backend opencv` |
| 依赖 | TensorFlow、YOLO、DeepArUco 仓库 | OpenCV |
| 角点来源 | 神经网络回归 | OpenCV 图像处理 |
| ID 解码 | 神经网络 + 模板匹配 | OpenCV ArUco |
| 速度 | 较慢 | 较快 |
| 弱光/模糊鲁棒性 | 可能更好，取决于模型 | 取决于图像质量 |
| 调试复杂度 | 较高 | 较低 |

## 11. Unity 集成

Python 端发送：

```bash
--send-unity-pose --unity-udp-host 127.0.0.1 --unity-udp-port 5055
```

Unity 端需要：

1. UDP 监听 `5055`。
2. 解析 JSON。
3. 将 OpenCV `rvec/tvec` 转换到 Unity 坐标系。
4. 驱动 Cube、STL 或其他 3D 模型。

推荐先用 Unity Cube 验证位姿，再导入 STL 模型。

## 12. 后续改进方向

可以考虑：

1. 增加姿态跳变拒绝逻辑，避免单帧错误解驱动渲染。
2. 为单 tag 使用 `solvePnPGeneric + SOLVEPNP_IPPE_SQUARE` 并选择与上一帧更接近的解。
3. 增加 `--render-alignment-mode center|edge`，支持中心对齐和边对齐切换。
4. 增加 DeepArUco 检测结果可视化，包括 bbox、解码距离和 accepted 状态。
5. 将单文件拆分为检测后端、几何布局、PnP、渲染、UDP 输出等模块。
6. 增加自动记录异常帧图片和对应 `used ids` / `mean_err`，便于复盘。

## 13. 总结

`deeparuco_fusedpose.py` 是一个集成 DeepArUco 检测、OpenCV ArUco 兼容输出、多 tag 融合位姿估计、实时 OpenCV 渲染和 Unity UDP 输出的综合脚本。

项目稳定性主要取决于：

1. DeepArUco 模型是否正确加载。
2. 解码 ID 是否正确映射到物理 tag ID。
3. 相机标定是否准确。
4. `marker_length`、`cube_size`、`id-face-map`、`id-rotation-map` 是否与实物一致。
5. 是否避免单个平面 tag 独立驱动姿态。

调试时建议优先观察 `used ids`、`mean_err`、检测角点编号、DeepArUco rejected 数量和模型角点编号。
