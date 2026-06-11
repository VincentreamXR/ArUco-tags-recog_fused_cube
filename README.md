# JuMarker / ArUco Rigid-Tag Workspace

这个目录已经不是单一的上游 JuMarker 源码仓库，而是一个混合工作区，包含三类内容：

1. 原始 JuMarker C++ 库与工具。
2. 基于 OpenCV ArUco / ChArUco / DeepArUco 的 Python 检测脚本。
3. 围绕 `4cm x 8cm x 4cm` 刚性多 Tag 长方体做的融合位姿实验脚本与生成资产。

当前最重要、也是最近持续修改的主入口是：

`detect_aruco_cube_rigid_async_pose.py`

这个脚本负责：

- 检测 9 个 ArUco Tag（侧面 0-7，顶面 8）。
- 把所有可见 Tag 角点统一送进一个整体 `solvePnP`。
- 在双线程视频管线中输出整体刚体位姿。
- 对最终位姿做滤波、短时保姿、顶面姿态约束和半透明模型渲染。

## 当前项目结构

### 1. 核心 C++ JuMarker

- `CMakeLists.txt`
- `src/`
- `utils/`

这部分是上游 JuMarker 的核心实现，提供：

- 自定义 SVG fiducial marker 的解析与检测。
- `utils/create_marker`
- `utils/jumarker_test`

### 2. 当前主用 Python 位姿脚本

- `detect_aruco_cube_rigid_async_pose.py`

推荐优先使用这个脚本。它已经集成了：

- `4x8x4cm` 长方体刚体模型。
- 9 Tag 融合位姿。
- 异步采集/处理。
- One Euro / EMA 平滑。
- 顶面 `ID=8` 整姿态锁定。
- 半透明 ID 面片渲染。
- 自动探测摄像头索引。

### 3. 旧版或实验版 ArUco 刚体脚本

- `detect_aruco_cube_fused_pose.py`
  早期 8 Tag、同步处理版本。

- `detect_aruco5x5_100_cube_fused_pose.py`
  固定 `DICT_5X5_100` 的变体。

- `detect_aruco_9tag_cube_rigid_async_pose.py`
  9 Tag 异步刚体脚本的旧版本。

- `detect_aruco_9tag_cube_dual_thread_fused_pose.py`
  双线程 9 Tag 融合位姿实验版，参数较多，适合继续对比调试。

这些脚本仍然有参考价值，但如果没有特殊原因，优先用 `detect_aruco_cube_rigid_async_pose.py`。

### 4. 通用检测脚本

- `detect_aruco3.py`
  通用 ArUco 检测基类脚本。

- `detect_aruco5x5_100_tags_cam.py`
- `detect_aruco6x6_250_tags_cam.py`
  分别是对 `detect_aruco3.py` 的固定字典包装。

- `detect_charuco_cam.py`
  ChArUco 板检测。

- `detect_deeparuco_tags_cam.py`
  OpenCV ArUco 风格的 Tag 检测/位姿脚本。

- `deeparuco/detect_deeparuco_cam.py`
  与外部 DeepArUco 仓库联动的检测脚本。

### 5. JuMarker / 自定义标记检测包装脚本

- `detect_jumarker_cam.py`
- `detect_seabery_cam.py`
- `detect_chimeta.py`
- `detect_rubik_8.py`

这类脚本的作用不是自己做视觉检测，而是：

- 组织 SVG 输入。
- 调 JuMarker 可执行文件。
- 生成或选择相应的 marker 资源。

### 6. Tag / SVG / 资产生成脚本

- `generate_marker_designs_8_id4.py`
  从模板批量生成 8 个 ID 的 SVG marker 设计。

- `generate_deeparuco_tags.py`
  生成普通 ArUco PNG 标签。

- `generate_deeparuco_mip_tags.py`
  生成 DeepArUco MIP36h12 PNG 标签。

- `generate_seabery_8.sh`
- `detect_rubik_8_ubuntu.sh`

### 7. 模板与输出目录

- `marker_designs/`
  各类原始 SVG 模板。

- `output_marker_designs_8_id4/`
- `output_rubik_8_id4/`
- `output_seabery_8/`
- `output_chimeta/`
- `output_chimed/`
- `output_deeparuco_tags_id0_31/`
- `output_deeparuco_mip36h12_tags_id0_31/`
- `test_output/`

这些目录主要存放生成后的 SVG / PNG / 测试输出，不是核心源码。

### 8. 第三方或派生内容

- `opencv4/`
  本地放了一份 OpenCV 4.9.0 源码快照，体积大，主要用于构建或参考，不属于本项目自写逻辑。

- `build/`
  CMake 构建产物目录。

- `jumarker_release.zip`
  打包文件。

## 当前主流程

以 `detect_aruco_cube_rigid_async_pose.py` 为准，流程如下：

1. 采集层  
   摄像头线程单独采集最新帧，避免处理阻塞视频。

2. 检测层  
   在处理线程中识别当前帧可见 ArUco ID 和角点。

3. 模型层  
   维护 9 个 Tag 在物体坐标系下的 3D 角点：
   - 侧面 `ID 0-7`
   - 顶面 `ID 8`
   - 物体模型为 `0.04 x 0.08 x 0.04 m`

4. 融合位姿层  
   用所有可见 Tag 的 2D/3D 对应点统一求解 `cv2.solvePnP(..., flags=cv2.SOLVEPNP_ITERATIVE)`。

5. 姿态约束层  
   若 `ID 8` 可见，保持融合平移 `tvec`，并用 `ID 8` 的单 Tag 姿态锁定最终 `rvec`，即：
   - `X/Y/Z` 朝向由顶面 Tag 决定
   - 位置仍由多 Tag 融合结果决定

6. 滤波层  
   使用 One Euro / EMA、重投影误差拒绝、短时 hold-last，降低抖动和瞬时丢失。

7. 渲染层  
   默认显示：
   - 每个 ID 的半透明面片
   - 一个整体坐标轴
   - 不显示外部黄色框线  
   可选显示长方体外框。

## 推荐运行方式

### 1. 使用当前主脚本

```bash
python3 /home/zsyy/桌面/JuMarker/detect_aruco_cube_rigid_async_pose.py \
  --camera auto \
  --calibration /home/zsyy/桌面/JuMarker/utils/camera_calibration.yml
```

如果摄像头索引不明确，优先用 `--camera auto`。

### 2. 只在需要时显示长方体线框

```bash
python3 /home/zsyy/桌面/JuMarker/detect_aruco_cube_rigid_async_pose.py \
  --camera auto \
  --draw-prism-wireframe
```

### 3. 打印每帧位姿

```bash
python3 /home/zsyy/桌面/JuMarker/detect_aruco_cube_rigid_async_pose.py \
  --camera auto \
  --print-pose
```

## 关键参数说明

`detect_aruco_cube_rigid_async_pose.py` 中最常用的参数：

- `--camera auto|N`
  自动搜索摄像头，或指定索引。

- `--camera-scan-max`
  自动搜索时扫描的最大索引上界。

- `--prism-width`
- `--prism-height`
- `--prism-depth`
  刚体模型尺寸，默认就是 `0.04 / 0.08 / 0.04`。

- `--top-id`
  顶面 Tag ID，默认 `8`。

- `--top-rotation-deg`
  顶面 Tag 在其平面内的模型旋转。

- `--corner-rolls`
  对单个 ID 的角点顺序做静态旋转修正。

- `--auto-roll-ids`
  动态搜索角点顺序的 ID 集合，默认包含 `0,3,8`。

- `--lock-top-pose-to-top-tag`
  当 `ID 8` 可见时，用顶面 Tag 锁定整体朝向。

- `--pose-filter`
  `none` / `ema` / `one_euro`。

- `--hold-last-seconds`
  短时丢失时保持上一帧稳定姿态的时间。

- `--draw-model-tags`
  显示半透明 ID 面片。

- `--draw-prism-wireframe`
  显示长方体外框。

## 依赖与环境

### Python 运行环境

当前项目里的多个 ArUco 脚本会优先尝试使用：

`/home/zsyy/anaconda3/envs/deeparuco39/bin/python`

也就是说，如果当前 Python 缺少 `cv2`，脚本会尝试切换到这个 conda 环境运行。

### DeepArUco 相关依赖

- `requirements_deeparuco_infer.txt`
- `setup_deeparuco_env.sh`

`setup_deeparuco_env.sh` 会创建 `deeparuco39` 环境并安装外部 DeepArUco 仓库依赖。

## C++ JuMarker 编译

项目根目录已有 CMake 配置：

```bash
cd /home/zsyy/桌面/JuMarker
mkdir -p build
cd build
cmake .. -DOPENCV_PATH=/path/to/opencv/install
make -j
```

构建后常用产物：

- `build/src/libjumarker.so`
- `build/utils/create_marker`
- `build/utils/jumarker_test`

## 已知问题与注意事项

1. `detect_puzzlepole_cam.py` 依赖 `calib_targets` 模块，但该模块不在当前目录中。  
   这意味着它不是一个完整自包含入口，README 只把它列为实验脚本。

2. `opencv4/` 是第三方源码快照，不建议在阅读项目逻辑时把它当成本项目源码。

3. 当前根目录里存在多个名字接近的 ArUco 刚体脚本。  
   如果只是要跑当前最新版本，直接用：
   `detect_aruco_cube_rigid_async_pose.py`

4. `README.md` 现在以“当前工作区的实际用途”为主，不再沿用上游 JuMarker 的论文式说明。

## 建议阅读顺序

如果要快速理解项目，建议按下面顺序读：

1. `detect_aruco_cube_rigid_async_pose.py`
2. `detect_aruco_9tag_cube_dual_thread_fused_pose.py`
3. `detect_aruco_cube_fused_pose.py`
4. `detect_aruco3.py`
5. `generate_marker_designs_8_id4.py`
6. `src/` 和 `utils/` 中的 JuMarker C++ 实现

## 当前结论

这个工作区的重点已经从“纯 JuMarker 库源码”转成了“基于多种 marker 技术路线的检测实验平台”，其中当前主线是：

- OpenCV ArUco
- 9 Tag 刚体建模
- 融合 PnP
- 双线程实时处理
- 顶面 `ID 8` 姿态约束
- 半透明模型可视化

如果后续继续整理，优先建议做两件事：

1. 把旧版实验脚本移入 `legacy/` 或 `experiments/`。
2. 给 `detect_aruco_cube_rigid_async_pose.py` 单独拆一个配置文件，而不是继续把物理建模参数全部留在命令行。
