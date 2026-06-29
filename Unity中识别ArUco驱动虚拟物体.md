# ArUco 姿态 UDP 到 Unity 项目技术细节文档

文档依据：

- Unity 接收端：`/home/zsyy/桌面/unity/PoseUdpReceiver.cs`
- 姿态发送端：`/home/zsyy/桌面/aruco_fusedpose.py`
- 运行命令记录：`/home/zsyy/桌面/unity/unity12026.odt`

## 1. 项目概述

本项目用于把 OpenCV ArUco 多标签刚体位姿实时传入 Unity。Python 端通过摄像头检测 9 标签双层立方体模型，融合多个 ArUco 标签的 2D 图像角点和 3D 刚体模型点，使用 `solvePnP` 计算刚体相对相机的位姿，然后通过 UDP 发送 JSON 数据。Unity 端的 `PoseUdpReceiver` 监听 UDP 端口，解析 `rvec` 和 `tvec`，完成 OpenCV 坐标系到 Unity 坐标系的转换，并把结果应用到指定 `Transform`。

整体链路如下：

```text
摄像头图像
  -> Python / OpenCV ArUco 检测
  -> 9 标签刚体模型融合位姿估计
  -> One Euro / EMA 等姿态稳定处理
  -> UDP JSON: rvec + tvec + 状态字段
  -> Unity PoseUdpReceiver
  -> 坐标系转换、偏移、平滑
  -> 更新 trackedRoot 的位置和旋转
```

## 2. 文件组成

| 文件 | 作用 |
| --- | --- |
| `PoseUdpReceiver.cs` | Unity `MonoBehaviour` 脚本，负责 UDP 接收、JSON 解析、姿态转换、目标物体更新和调试显示。 |
| `unity12026.odt` | 记录了启动 Python 发送端并向 Unity 发送姿态的命令。 |
| `Unity_lic.alf` | Unity 许可证申请文件，与姿态通信逻辑无直接关系。 |
| `/home/zsyy/桌面/aruco_fusedpose.py` | Python 端 ArUco 检测、融合位姿估计、滤波、可视化和 Unity UDP 发送脚本。 |

## 3. 运行命令

`unity12026.odt` 中记录的当前 Unity 对接命令为：

```bash
/home/zsyy/anaconda3/envs/deeparuco39/bin/python '/home/zsyy/桌面/aruco_fusedpose.py' \
    --camera 0 \
    --width 640 \
    --height 480 \
    --send-unity-pose \
    --unity-udp-host 127.0.0.1 \
    --unity-udp-port 5055
```

关键参数说明：

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `--camera` | `0` | 使用编号为 0 的摄像头。 |
| `--width` / `--height` | `640` / `480` | 请求摄像头分辨率。 |
| `--send-unity-pose` | 开启 | 允许 Python 端向 Unity 发送 UDP 姿态包。 |
| `--unity-udp-host` | `127.0.0.1` | Unity 接收端主机，当前为本机回环地址。 |
| `--unity-udp-port` | `5055` | Unity 接收端端口，需与 `PoseUdpReceiver.listenPort` 一致。 |

如果 Unity 和 Python 不在同一台机器，需要把 `--unity-udp-host` 改为 Unity 所在机器的局域网 IP，并确保防火墙允许 UDP 端口通信。

## 4. Python 发送端技术细节

### 4.1 姿态计算来源

`aruco_fusedpose.py` 默认检测一个 9 标签双层立方体刚体：

- 顶面标签 ID：`0`
- 上层侧面标签 ID：`1,2,3,4`
- 下层侧面标签 ID：`5,6,7,8`
- 默认单个立方体边长：`0.04 m`
- 默认标签边长：`0.032 m`
- 默认 ArUco 字典：`DICT_6X6_250`

Python 端从摄像头图像中检测 ArUco 标签，按刚体布局收集每个可见标签的 3D 模型角点和 2D 图像角点，然后调用 OpenCV `solvePnP` 求解刚体从物体坐标系到相机坐标系的旋转向量 `rvec` 和平移向量 `tvec`。

### 4.2 姿态稳定策略

发送给 Unity 的不是简单的单帧原始检测结果，而是经过稳定策略处理后的 `draw_pose`：

- 默认姿态滤波为 `one_euro`，用于降低抖动并保持快速运动跟随。
- `--max-stable-reprojection-error` 默认 `8.0`，平均重投影误差过大时拒绝更新。
- `--hold-last-seconds` 默认 `0.25`，短时丢失检测时保持上一帧稳定姿态。
- 标签候选选择使用质量评分、重投影误差、候选切换迟滞和顶部标签方向辅助，减少相邻面切换时跳变。

当 `render_state.draw_pose` 存在时，Python 端发送 `valid=true`；否则发送 `valid=false`，并保留当前 `used_ids` 与 `pose_state`。

### 4.3 UDP 发送实现

Python 端使用 `UnityPoseSender`：

```python
self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
self.socket.sendto(payload.encode("utf-8"), self.address)
```

数据由 `build_unity_pose_payload()` 生成，使用紧凑 JSON 格式：

```json
{
  "valid": true,
  "timestamp": 123.456,
  "rvec": [0.01, 0.02, 0.03],
  "tvec": [0.10, -0.02, 0.55],
  "used_ids": [1, 2],
  "mean_error": 2.4,
  "pose_state": "measured"
}
```

字段含义：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `valid` | `bool` | 当前姿态是否有效。无可用姿态时为 `false`。 |
| `timestamp` | `float` | Python 端 `time.monotonic()` 时间戳，用于调试，不直接驱动 Unity 更新。 |
| `rvec` | `float[3]` 或 `null` | OpenCV Rodrigues 旋转向量，表示物体坐标系到相机坐标系的旋转。 |
| `tvec` | `float[3]` 或 `null` | OpenCV 平移向量，单位与刚体模型一致，当前模型按米建模。 |
| `used_ids` | `int[]` | 本次参与融合位姿求解的标签 ID。 |
| `mean_error` | `float` 或 `null` | 平均重投影误差，单位为像素。 |
| `pose_state` | `string` | 姿态状态，如 `measured`、`held_no_detection`、`held_bad_reproj`、`lost`。 |

## 5. Unity 接收端技术细节

### 5.1 组件职责

`PoseUdpReceiver` 是一个 `MonoBehaviour`，主要职责为：

1. 在 `OnEnable()` 中创建 `UdpClient` 并启动后台接收线程。
2. 后台线程阻塞接收 UDP 数据，把 UTF-8 JSON 解析为 `PosePacket`。
3. 在 Unity 主线程 `LateUpdate()` 中读取最新姿态。
4. 校验 `valid`、`rvec`、`tvec` 和过期时间。
5. 把 OpenCV 坐标和 Rodrigues 旋转转换为 Unity 的 `Vector3` 与 `Quaternion`。
6. 根据姿态空间配置，把结果应用到 `trackedRoot` 的本地坐标或世界坐标。
7. 在 `OnGUI()` 中显示接收状态、标签数量、误差、空间、参考对象、姿态年龄、转换后位置和旋转。

### 5.2 Inspector 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `listenPort` | `5055` | Unity UDP 监听端口，必须与 Python `--unity-udp-port` 一致。 |
| `trackedRoot` | 未指定 | 被驱动的 Unity 物体根节点。必须手动拖入。 |
| `hideWhenInvalid` | `false` | 姿态无效或超时时是否隐藏 `trackedRoot`。 |
| `applyPose` | `true` | 是否实际应用姿态。关闭后只接收数据，不更新物体。 |
| `incomingPoseSpace` | `ReferenceLocalToWorld` | 输入姿态空间解释方式。 |
| `poseReference` | 未指定 | 参考坐标系。为空时优先使用 `Camera.main`。 |
| `positionScale` | `1.0` | 平移缩放。Python 当前按米输出，Unity 单位也按米时保持 `1.0`。 |
| `markerLocalPositionOffset` | `(0,0,0)` | 沿标记自身局部旋转方向施加的位置偏移。 |
| `positionOffset` | `(0,0,0)` | 在转换后的参考局部坐标中追加的位置偏移。 |
| `eulerOffsetDegrees` | `(0,0,0)` | 对转换后的旋转追加欧拉角偏移。 |
| `smoothPose` | `true` | 是否在 Unity 端做插值平滑。 |
| `positionLerpSpeed` | `30.0` | 位置平滑速度。越大越跟手，越小越稳。 |
| `rotationSlerpSpeed` | `30.0` | 旋转平滑速度。越大越跟手，越小越稳。 |
| `staleTimeoutSeconds` | `0.0` | 姿态超时阈值。`0` 表示不按年龄判定过期。 |
| `showDebugOverlay` | `true` | 是否显示 `OnGUI` 调试信息。 |

### 5.3 姿态空间模式

`PoseUdpReceiver` 内部定义了三种 `PoseSpace`：

| 模式 | 应用方式 | 适用场景 |
| --- | --- | --- |
| `TargetParentLocal` | 直接写入 `trackedRoot.localPosition` 和 `trackedRoot.localRotation`。 | 外部发送的数据已经是目标父节点局部空间。 |
| `ReferenceLocalToWorld` | 先把接收到的姿态视为 `poseReference` 的局部空间，再通过 `TransformPoint` 和 `reference.rotation` 转到世界空间。 | 当前默认模式，适合把 OpenCV 相机空间挂到 Unity 相机或某个参考节点下。 |
| `World` | 直接写入 `trackedRoot.position` 和 `trackedRoot.rotation`。 | 外部发送的数据已经是 Unity 世界空间。 |

默认 `ReferenceLocalToWorld` 的处理逻辑：

```text
OpenCV 相机空间姿态
  -> Unity 参考局部姿态
  -> poseReference.TransformPoint(position)
  -> poseReference.rotation * rotation
  -> trackedRoot 世界姿态
```

如果 `poseReference` 未设置，脚本会尝试使用 `Camera.main.transform`。如果场景中没有 Main Camera，会回退到 `TargetParentLocal`，并输出一次警告。

## 6. 坐标系与旋转转换

### 6.1 平移转换

OpenCV 图像/相机坐标系和 Unity 坐标系的 Y 轴方向不同。Unity 脚本使用：

```csharp
return new Vector3(tvec[0], -tvec[1], tvec[2]) * positionScale;
```

含义：

- OpenCV `x` -> Unity `x`
- OpenCV `y` -> Unity `-y`
- OpenCV `z` -> Unity `z`
- 最后乘以 `positionScale`

由于 Python 刚体模型以米为单位，`positionScale=1.0` 时 Unity 中 1 个单位对应 1 米。

### 6.2 旋转转换

Unity 端使用 `ConvertOpenCvRodriguesToUnity()` 处理 `rvec`：

1. `RodriguesToMatrix()` 把 OpenCV Rodrigues 旋转向量转为 3x3 旋转矩阵。
2. `FlipY()` 对矩阵做 Y 轴翻转，以匹配 Unity 坐标方向。
3. `QuaternionFromMatrix()` 取矩阵的 forward 和 upwards 向量，用 `Quaternion.LookRotation()` 生成 Unity 四元数。

转换后的旋转还会追加：

```csharp
referenceLocalRotation = markerLocalRotation * Quaternion.Euler(eulerOffsetDegrees);
```

因此如果 Unity 模型的本地朝向和物理标记坐标系不一致，应优先通过 `eulerOffsetDegrees` 做固定校正。

### 6.3 位置偏移顺序

Unity 端位置偏移顺序如下：

```text
referenceLocalPosition = ConvertOpenCvPositionToUnity(tvec)
markerLocalRotation = ConvertOpenCvRodriguesToUnity(rvec)
referenceLocalRotation = markerLocalRotation * Euler(eulerOffsetDegrees)
referenceLocalPosition += markerLocalRotation * markerLocalPositionOffset
referenceLocalPosition += positionOffset
```

两类偏移区别：

- `markerLocalPositionOffset` 会先乘以标记旋转，适合表达“模型中心相对标记中心”的局部偏移。
- `positionOffset` 不随标记旋转变化，适合表达参考空间中的固定平移修正。

## 7. 线程与生命周期

### 7.1 接收线程

Unity 接收使用后台线程：

```csharp
receiveThread = new Thread(ReceiveLoop)
{
    IsBackground = true,
    Name = "ArUco pose UDP receiver"
};
```

线程内调用 `udpClient.Receive(ref remote)` 阻塞等待数据。收到数据后：

1. 使用 UTF-8 解码。
2. 使用 `JsonUtility.FromJson<PosePacket>()` 解析。
3. 在 `poseLock` 锁内更新 `latestPose`、`latestPoseTicks` 和 `hasPose`。

### 7.2 主线程应用

Unity 的 Transform 只能在主线程安全更新，因此脚本没有在接收线程中操作场景对象，而是在 `LateUpdate()` 中读取最新姿态并更新 `trackedRoot`。

使用 `LateUpdate()` 的好处是：如果场景中还有其它逻辑在 `Update()` 修改相机或父节点，姿态应用会尽量发生在同一帧较靠后的位置。

### 7.3 关闭清理

`OnDisable()` 执行：

```csharp
running = false;
udpClient?.Close();
receiveThread.Join(100);
```

`UdpClient.Close()` 会打断阻塞中的 `Receive()`，接收线程捕获 `ObjectDisposedException` 后退出。这样可以避免停止播放或禁用组件时遗留后台线程。

## 8. 平滑与延迟

Unity 端平滑使用指数形式的帧率无关插值：

```csharp
alpha = 1.0f - Mathf.Exp(-speed * Time.deltaTime);
```

位置用 `Vector3.Lerp()`，旋转用 `Quaternion.Slerp()`。当 `speed <= 0` 时，`alpha=1`，即每帧直接跳到目标姿态。

需要注意：Python 端已经有姿态滤波和短时保持，Unity 端又默认开启 `smoothPose`。如果画面延迟明显，可以按以下顺序排查：

1. 先把 Unity 的 `smoothPose` 关闭，看原始跟随是否正常。
2. 再调整 Python 的 `--pose-filter`、`--one-euro-min-cutoff`、`--one-euro-beta`。
3. 最后再适当恢复 Unity 端平滑，用于视觉表现而不是修正算法抖动。

## 9. Unity 场景接入步骤

1. 把 `PoseUdpReceiver.cs` 放入 Unity 项目的 `Assets` 目录。
2. 在场景中新建一个空物体，例如 `PoseUdpReceiverHost`。
3. 给该空物体挂载 `PoseUdpReceiver` 组件。
4. 把需要跟随 ArUco 刚体的模型根节点拖到 `trackedRoot`。
5. 确认 `listenPort` 为 `5055`，与 Python 命令一致。
6. 如果使用 Unity 相机作为 OpenCV 相机参考，设置 `incomingPoseSpace=ReferenceLocalToWorld`，并把 `poseReference` 指向对应相机；未设置时脚本会自动尝试 `Camera.main`。
7. 如果模型中心和 ArUco 刚体原点不一致，调整 `markerLocalPositionOffset` 或 `positionOffset`。
8. 如果模型朝向不一致，调整 `eulerOffsetDegrees`。
9. 启动 Unity Play Mode 后，再启动 Python 发送端，观察左上角调试信息。

## 10. 调试信息说明

`showDebugOverlay=true` 时，Unity 左上角显示两行信息。

第一行：

```text
ArUco UDP pose: measured ids=2 err=2.40
```

含义：

- `measured`：Python 端姿态状态。
- `ids=2`：本次参与融合的标签数量。
- `err=2.40`：平均重投影误差，像素单位。

第二行：

```text
space=ReferenceLocalToWorld ref=Main Camera age=0.016s pos=(...) rot=(...)
```

含义：

- `space`：当前 Unity 姿态空间模式。
- `ref`：参考 Transform 名称。
- `age`：Unity 收到该 UDP 包到当前帧经过的时间。
- `pos`：坐标转换后的参考局部位置。
- `rot`：坐标转换后的欧拉角。

## 11. 常见问题与处理

### 11.1 Unity 没有任何反应

检查项：

- Unity 是否处于 Play Mode。
- `PoseUdpReceiver` 组件是否启用。
- `trackedRoot` 是否已经拖入。
- Python 是否带了 `--send-unity-pose`。
- Python 的 `--unity-udp-port` 是否等于 Unity `listenPort`。
- Unity 和 Python 是否在同一机器；如果不在同一机器，`--unity-udp-host` 不能使用 `127.0.0.1`。

### 11.2 物体方向不对

优先检查：

- `incomingPoseSpace` 是否选择正确。
- `poseReference` 是否指向正确相机或参考节点。
- Unity 模型本地轴向是否和 ArUco 刚体物体坐标系一致。
- 使用 `eulerOffsetDegrees` 做固定旋转补偿。

### 11.3 物体位置尺度不对

Python 模型尺寸按米定义，Unity 默认也可按 1 unit = 1 m 处理。若 Unity 模型或场景使用其它比例，调整：

- `positionScale`
- Python 端 `--cube-size`
- Python 端 `--marker-length`
- 实际相机标定文件 `--calibration`

如果没有使用真实相机标定，`tvec` 的尺度和精度会受近似内参影响，平移量不适合作为严格测量值。

### 11.4 姿态抖动或延迟

处理顺序：

1. 查看 Python 窗口中的重投影误差，优先保证检测和模型布局正确。
2. 确认标签打印尺寸、粘贴位置、`--marker-length`、`--cube-size` 与实物一致。
3. 用真实相机标定文件启动 Python。
4. 临时关闭 Unity `smoothPose`，判断延迟来自 Unity 端还是 Python 端。
5. 调整 Python 滤波和候选迟滞参数。

### 11.5 姿态丢失后物体不隐藏

默认 `hideWhenInvalid=false` 且 `staleTimeoutSeconds=0`，所以姿态无效或旧包不会自动隐藏。需要隐藏时：

- 开启 `hideWhenInvalid`
- 设置 `staleTimeoutSeconds`，例如 `0.2`

## 12. 推荐验收标准

用于确认 Unity 对接是否成功：

1. Python 启动后终端打印 `Unity pose UDP: 127.0.0.1:5055`。
2. Unity 左上角状态从 `waiting` 变为 `measured` 或 `held_no_detection`。
3. `ids` 数量随可见标签变化，`err` 在合理范围内波动。
4. 缓慢移动实体 ArUco 刚体时，Unity 中 `trackedRoot` 平移和旋转连续变化。
5. 遮挡标签时，Unity 按配置保持上一姿态或隐藏目标；恢复可见后能够重新跟踪。
6. 调整 `eulerOffsetDegrees` 和偏移后，Unity 模型与实体刚体坐标方向一致。

## 13. 后续工程化建议

当前项目核心逻辑已经能完成 Python 到 Unity 的实时姿态传输。若后续要长期维护，建议补充：

- 把 UDP JSON 协议单独写成版本化协议文档，后续字段变更时保持兼容。
- 在 Unity 中增加最近包时间、丢包计数、平均接收频率等诊断指标。
- 在 Python 端增加发送频率限制或序号字段，便于排查网络抖动。
- 为 `PoseUdpReceiver` 增加编辑器预设，例如“相机参考模式”“世界空间模式”“禁用平滑调试模式”。
- 保存每次实物模型尺寸、相机标定文件、Unity 偏移参数，避免不同设备重复调参。
