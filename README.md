
# ArUco 融合位姿识别与渲染项目技术说明

本文档用于说明该脚本项目的功能、运行环境、核心流程、关键参数和调试方法。

## 1. 项目概述

本项目是一个基于 OpenCV ArUco 的实时位姿估计与可视化脚本。脚本从摄像头或单张图片中检测多个 ArUco tag，根据预设的刚体几何布局计算一个融合的物体位姿，并将模型投影渲染到视频画面中。

项目主要面向一个带多个 ArUco tag 的双层立方体或棱柱刚体。默认配置包含：

- 顶面 tag：`0`
- 上层侧面 tag：`1,2,3,4`
- 下层侧面 tag：`5,6,7,8`
- 默认 ArUco 字典：`DICT_6X6_250`
- 默认 marker 边长：`0.032 m`
- 默认单个 cube 边长：`0.04 m`

脚本除了本地 OpenCV 画面渲染外，还支持通过 UDP 向 Unity 发送位姿数据。

## 2. 主要功能

### 2.1 ArUco 检测

脚本使用 OpenCV `cv2.aruco` 检测图像中的 marker，支持多种 ArUco 字典，包括：

- `DICT_4X4_50`
- `DICT_5X5_100`
- `DICT_6X6_250`
- `DICT_7X7_1000`
- `DICT_ARUCO_ORIGINAL`

检测参数包括自适应阈值窗口、marker 周长比例、角点亚像素优化和可选 ArUco3 检测。

### 2.2 刚体布局建模

脚本内部定义了一个物体坐标系：

- `+X` 指向 right 面
- `+Y` 指向 front 面
- `+Z` 指向 top 面
- 原点位于上下两个 cube 中心之间

每个 tag 的 3D 角点由以下参数决定：

- `--cube-size`
- `--marker-length`
- `--vertical-gap`
- `--upper-ids`
- `--lower-ids`
- `--top-id`
- `--id-face-map`
- `--id-rotation-map`
- `--corner-rolls`

脚本会将检测到的 2D 角点和预设 3D 角点组成对应关系，然后用于 `solvePnP` 位姿求解。

### 2.3 融合位姿估计

脚本不是简单地逐个 tag 独立估计姿态，而是根据可见 tag 建立 2D-3D 对应点，并计算一个统一的刚体位姿。

核心流程：

1. 检测所有可见 ArUco tag。
2. 根据 `id-face-map` 找到每个 tag 在刚体上的 3D 位置。
3. 收集可见 tag 的 3D 角点和图像 2D 角点。
4. 生成候选 tag 集合，例如上一帧候选、面积最大的单 tag、最优相邻 tag 对。
5. 对每个候选调用 `cv2.solvePnP`。
6. 计算重投影误差。
7. 根据误差、上一帧连续性、tag 面积质量和 top tag 约束选择最佳姿态。
8. 对姿态进行滤波和保持。

### 2.4 姿态稳定与滤波

脚本提供两类姿态滤波：

- `ema`：指数滑动平均
- `one_euro`：One Euro Filter，默认启用

相关参数：

- `--pose-filter`
- `--ema-alpha`
- `--one-euro-min-cutoff`
- `--one-euro-beta`
- `--one-euro-derivate-cutoff`
- `--hold-last-seconds`
- `--max-stable-reprojection-error`

当当前帧姿态质量较差或短暂丢失检测时，脚本可以保持上一帧稳定姿态，避免画面瞬间跳变。

### 2.5 异步摄像头处理

摄像头模式下，脚本使用两个后台线程：

- `capture_loop`：持续从摄像头采集最新帧。
- `processing_loop`：异步处理最新帧，执行检测、位姿估计和渲染状态更新。

主线程负责显示窗口，并使用最新的渲染状态叠加到当前摄像头画面上。

该设计可以降低摄像头采集阻塞对显示流畅度的影响。

### 2.6 模型渲染

脚本使用 OpenCV 将模型 3D 点投影到图像上，支持：

- 绘制检测到的 ArUco 框。
- 绘制检测角点和角点编号。
- 绘制融合位姿坐标轴。
- 绘制配置好的刚体 tag 面。
- 绘制双层 cube 线框。
- 绘制 tag 面半透明填充。

需要注意：脚本中的渲染模型布局 `render_layout` 和位姿求解布局 `layout` 是两套几何：

- `layout` 使用 `marker_length`，表示真实 ArUco tag 的边长。
- `render_layout` 使用 `cube_size`，表示物理 cube 面或渲染模型面的边长。

因此绿色渲染模型不一定贴合黑色 ArUco tag 边框，它默认表达的是刚体模型面。

### 2.7 Unity UDP 位姿输出

脚本支持将最新姿态通过 UDP 发送给 Unity：

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

- `--send-unity-pose`
- `--unity-udp-host`
- `--unity-udp-port`

Unity 端需要接收 JSON，解析 `rvec/tvec`，并完成 OpenCV 坐标系到 Unity 坐标系的转换。

## 3. 运行环境

脚本依赖：

- Python 3
- OpenCV Python
- NumPy
- 摄像头设备
- 可选：OpenCV 标定文件

脚本内置了一个 OpenCV 环境切换逻辑：

```text
/home/zsyy/anaconda3/envs/deeparuco39/bin/python
```

如果当前 Python 缺少 `cv2`，脚本会尝试切换到该环境重新执行。

建议直接使用该解释器运行：

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/aruco_fusedpose121.py
```

## 4. 常用运行命令

### 4.1 摄像头实时识别

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/aruco_fusedpose121.py \
  --camera 0
```

### 4.2 使用相机标定文件

建议将标定文件放到纯英文路径，例如：

```text
/home/zsyy/camera_calibration.yml
```

运行：

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/aruco_fusedpose121.py \
  --camera 0 \
  --calibration /home/zsyy/camera_calibration.yml
```

### 4.3 降低单 tag 姿态翻转风险

单个平面 tag 存在姿态二义性。建议要求至少 2 个 tag 才更新姿态：

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/aruco_fusedpose121.py \
  --camera 0 \
  --calibration /home/zsyy/camera_calibration.yml \
  --min-tags-for-pose 2 \
  --hold-last-seconds 0.5 \
  --max-stable-reprojection-error 5.0 \
  --print-pose
```

### 4.4 显示角点编号和模型角点

用于检查 tag 方向、角点顺序和模型布局：

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/aruco_fusedpose121.py \
  --camera 0 \
  --calibration /home/zsyy/camera_calibration.yml \
  --draw-detected-corners \
  --draw-corner-index \
  --draw-model-corner-labels \
  --print-pose
```

### 4.5 向 Unity 发送姿态

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/aruco_fusedpose121.py \
  --camera 0 \
  --calibration /home/zsyy/camera_calibration.yml \
  --send-unity-pose \
  --unity-udp-host 127.0.0.1 \
  --unity-udp-port 5055
```

### 4.6 单张图片检测

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python /home/zsyy/桌面/aruco_fusedpose121.py \
  --image /path/to/image.png \
  --output /path/to/output.png
```

## 5. 关键参数说明

### 5.1 摄像头参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--camera` | `0` | 摄像头编号，也可用 `auto` |
| `--width` | `640` | 请求采集宽度 |
| `--height` | `480` | 请求采集高度 |
| `--fps` | `30` | 请求帧率 |
| `--camera-buffer` | `1` | 摄像头缓冲区大小 |
| `--autofocus` | `1` | 是否开启自动对焦 |

### 5.2 ArUco 检测参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--dictionary` | `DICT_6X6_250` | ArUco 字典 |
| `--aruco3` | 关闭 | 启用 OpenCV ArUco3 检测 |
| `--adaptive-min` | `3` | 自适应阈值最小窗口 |
| `--adaptive-max` | `53` | 自适应阈值最大窗口 |
| `--adaptive-step` | `10` | 自适应阈值窗口步长 |
| `--min-marker-perimeter-rate` | `0.015` | marker 最小周长比例 |
| `--max-marker-perimeter-rate` | `4.0` | marker 最大周长比例 |

### 5.3 几何布局参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--marker-length` | `0.032` | ArUco tag 边长，单位米 |
| `--cube-size` | `0.04` | 单个 cube 边长，单位米 |
| `--vertical-gap` | `0.0` | 上下 cube 间隙 |
| `--upper-ids` | `1,2,3,4` | 上层侧面 tag ID |
| `--lower-ids` | `5,6,7,8` | 下层侧面 tag ID |
| `--top-id` | `0` | 顶面 tag ID |
| `--upper-rotation-deg` | `45.0` | 上层 cube 绕 Z 轴旋转 |
| `--lower-rotation-deg` | `0.0` | 下层 cube 绕 Z 轴旋转 |
| `--id-face-map` | `1:front,2:left,...` | ID 到物理面的映射 |
| `--id-rotation-map` | `0:90` | 每个 ID 在面内的旋转 |
| `--corner-rolls` | 空 | 固定角点顺序修正 |

### 5.4 位姿稳定参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--min-tags-for-pose` | `1` | 最少参与位姿估计的 tag 数 |
| `--pose-filter` | `one_euro` | 姿态滤波器 |
| `--hold-last-seconds` | `0.25` | 检测失败后保持上一帧姿态的时间 |
| `--max-stable-reprojection-error` | `8.0` | 超过该平均重投影误差时拒绝更新 |
| `--reject-outlier-tags` | 开启 | 剔除重投影误差异常 tag |
| `--candidate-switch-hysteresis` | `0.18` | 候选 tag 集切换迟滞 |
| `--render-anchor-switch-ratio` | `1.35` | 渲染 anchor 切换门限 |

### 5.5 渲染参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--draw-model-tags` | 开启 | 绘制配置好的模型 tag 面 |
| `--draw-prism-wireframe` | 关闭 | 绘制 cube 线框 |
| `--draw-detected-corners` | 关闭 | 绘制检测角点 |
| `--draw-corner-index` | 关闭 | 显示检测角点编号 |
| `--draw-model-corner-labels` | 关闭 | 显示模型角点编号 |
| `--render-model-rotation-deg` | `180.0` | 渲染模型绕 Z 轴旋转 |
| `--render-model-translation` | `-0.04,0,-0.004` | 渲染模型平移 |
| `--render-alignment-smoothing-alpha` | `0.25` | 渲染对齐偏移平滑系数 |

## 6. 核心数据流

### 6.1 摄像头模式

```text
摄像头采集
  ↓
LatestFrameSlot 保存最新帧
  ↓
处理线程取最新帧
  ↓
灰度转换
  ↓
ArUco 检测
  ↓
2D 角点 + 3D 刚体布局
  ↓
solvePnP 融合位姿估计
  ↓
姿态滤波 / 质量判断 / hold
  ↓
生成 PoseRenderState
  ↓
主线程将渲染状态叠加到最新画面
```

### 6.2 图片模式

```text
读取图片
  ↓
如无标定则估算内参
  ↓
检测 ArUco
  ↓
估计位姿
  ↓
绘制结果
  ↓
可选保存输出图片
```

## 7. 主要模块说明

### 7.1 `PoseRenderState`

该数据类保存一帧处理后的渲染状态，包括：

- 检测角点
- 检测 ID
- 实际用于位姿估计的 ID
- 渲染 anchor ID
- 当前姿态
- 姿态状态文本
- 帧率和检测耗时
- 最终用于绘制的姿态
- 渲染对齐偏移

摄像头模式中，处理线程将 `PoseRenderState` 传给显示线程。

### 7.2 `PoseTracker`

负责位姿稳定：

- 记录上一帧原始姿态。
- 记录上一帧输出姿态。
- 使用 EMA 或 One Euro Filter 平滑姿态。
- 根据重投影误差拒绝不稳定更新。
- 在短暂丢失检测时保持上一帧姿态。
- 记录上一帧候选 tag 集合和渲染 anchor。

### 7.3 `LatestFrameSlot`

线程安全的“最新帧槽”。它只保存最新数据，不维护队列，因此处理线程落后时会自动跳过旧帧，降低延迟。

### 7.4 `estimate_best_fused_pose`

负责从多个候选姿态中选择最佳姿态：

- 支持动态角点 roll 搜索。
- 支持 top tag 姿态辅助消歧。
- 支持候选 tag 集稳定选择。
- 支持按重投影误差剔除离群 tag。
- 支持候选切换迟滞，减少来回跳变。

### 7.5 `draw_render_state`

负责最终渲染：

- 绘制检测 marker。
- 绘制角点调试信息。
- 根据 `render_alignment_offset` 平移渲染布局。
- 可选反射或翻转渲染模型。
- 调用 `draw_fused_model` 绘制模型和坐标轴。
- 绘制顶部状态文本面板。

## 8. 常见问题与原因

### 8.1 模型突然翻到 tag 外侧

常见原因：

- 当前帧只使用了单个平面 tag，`solvePnP` 出现平面姿态二义性。
- 角点检测受遮挡、模糊或斜视角影响。
- `used_ids` 在不同候选 tag 集之间切换。
- 某个 ID 的 `id-face-map` 或 `id-rotation-map` 与真实贴法不一致。

建议：

- 使用 `--min-tags-for-pose 2`。
- 增大 `--hold-last-seconds`。
- 降低 `--max-stable-reprojection-error`。
- 使用 `--print-pose` 检查跳变时的 `used ids` 和误差。

### 8.2 同时识别多个 tag 时模型来回切换

常见原因：

- 多个候选 tag 集评分接近。
- tag 面积质量随视角变化，导致 anchor 切换。
- 不同 tag 的真实贴法与脚本几何布局存在小误差。

建议：

- 检查 `used ids` 是否在跳变。
- 提高 `--candidate-switch-hysteresis`。
- 提高 `--render-anchor-switch-ratio`。
- 确认每个 tag 的物理面和旋转配置。

### 8.3 模型和实际物体不重合

常见原因：

- `marker_length`、`cube_size` 或 `vertical_gap` 与实物不一致。
- `id-face-map` 与实际贴纸所在面不一致。
- `id-rotation-map` 或 `corner-rolls` 与贴纸方向不一致。
- 渲染模型使用的是 `cube_size` 面，而不是 `marker_length` tag 边框。
- 相机标定不准或分辨率与标定时不一致。

建议：

- 打开 `--draw-detected-corners --draw-corner-index --draw-model-corner-labels`。
- 检查检测角点 `c0-c3` 和模型角点 `a0-a3` 的对应关系。
- 使用真实标定文件。
- 确认运行分辨率与标定分辨率匹配。

### 8.4 `cv2.FileStorage` 读取标定文件报错

常见原因：

- 标定文件路径包含中文或特殊字符。
- 当前 Python/OpenCV 环境和脚本预期环境不一致。
- 标定文件格式不是 OpenCV YAML/XML。

建议：

- 将标定文件复制到纯英文路径，例如 `/home/zsyy/camera_calibration.yml`。
- 使用 `/home/zsyy/anaconda3/envs/deeparuco39/bin/python` 运行。
- 确认文件中包含 `camera_matrix` 和 `distortion_coefficients`。

## 9. Unity 集成说明

脚本可通过 UDP 向 Unity 输出位姿。推荐架构：

```text
Python/OpenCV
  摄像头采集
  ArUco 检测
  solvePnP 位姿估计
  UDP 输出 rvec/tvec

Unity
  UDP 接收 JSON
  OpenCV 坐标系转 Unity 坐标系
  驱动 Cube/STL/模型姿态
```

注意坐标系差异：

- OpenCV 图像/相机坐标：`X` 向右，`Y` 向下，`Z` 向前。
- Unity 坐标：通常 `X` 向右，`Y` 向上，`Z` 向前。

Unity 中通常需要对 Y 轴取反，并正确转换 Rodrigues 旋转向量到四元数。

建议先在 Unity 中用简单 Cube 验证姿态跟随，再替换为 STL 模型。

## 10. 调试建议

### 10.1 观察 `used ids`

运行时加：

```bash
--print-pose
```

如果模型跳变时 `used ids` 变化，说明是候选 tag 集切换导致。

### 10.2 观察重投影误差

终端输出中的 `mean_err` 和 `max_err` 可以判断当前姿态质量。误差突然升高通常意味着角点检测或几何配置有问题。

### 10.3 检查角点顺序

使用：

```bash
--draw-detected-corners --draw-corner-index --draw-model-corner-labels
```

若检测角点和模型角点顺序不一致，需要调整：

- `--id-rotation-map`
- `--corner-rolls`

### 10.4 避免单 tag 更新姿态

建议使用：

```bash
--min-tags-for-pose 2
```

这可以显著降低单个平面 tag 导致的姿态翻转。

### 10.5 使用稳定的相机配置

建议：

- 使用固定分辨率。
- 使用真实标定文件。
- 尽量关闭频繁对焦变化。
- 保证 tag 清晰、无遮挡、有足够像素面积。

## 11. 后续改进方向

可以考虑的工程改进：

1. 增加姿态跳变拒绝逻辑：当新姿态相对上一帧旋转突变超过阈值时拒绝更新。
2. 对单 tag 使用 `solvePnPGeneric + SOLVEPNP_IPPE_SQUARE`，选择与上一帧更接近的解。
3. 增加 `--render-alignment-mode center|edge` 参数，支持中心对齐或边对齐。
4. 将 STL 模型加载与投影绘制集成进脚本。
5. 将 Unity 接收端做成标准组件，直接消费 UDP 位姿。
6. 将单文件脚本拆分为检测、几何、位姿、渲染、Unity 输出等模块，便于维护和测试。

## 12. 总结

`aruco_fusedpose121.py` 是一个功能完整的 ArUco 刚体融合位姿估计脚本。它集成了摄像头采集、marker 检测、刚体几何建模、PnP 位姿估计、姿态稳定、模型渲染和 Unity UDP 输出。

项目效果高度依赖三项配置的准确性：

1. 相机标定是否准确。
2. tag 的物理尺寸和空间布局是否与脚本参数一致。
3. 每个 tag 的面映射、旋转和角点顺序是否与实物一致。

在调试渲染偏移、姿态跳变或模型不贴合时，应优先检查 `used ids`、重投影误差、角点编号和 `id-face-map` / `id-rotation-map` 配置。
