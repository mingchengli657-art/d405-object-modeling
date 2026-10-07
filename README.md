# D405 Object Modeling

**Piper 视觉抓取系列 · 02 / 物体建模** · [系列总入口与整套运行指南](https://github.com/mingchengli657-art/piper-vision-grasping)

本模块输出米制模型，供 [FoundationPose 跟踪](https://github.com/mingchengli657-art/foundationpose-d405-runtime)使用；在线抓取另需[手眼标定](https://github.com/mingchengli657-art/piper_handeye_calibration)与 [Piper 控制](https://github.com/mingchengli657-art/piper-known-object-control)。

基于 **Intel RealSense D405 + ChArUco 标定板** 的 RGB-D 物体建模工具，从机器人比赛工程中提取。物体与板保持固定，操作者手持相机改变视角；程序根据板恢复相机位姿，将多帧深度融合到统一坐标系，并输出彩色点云及粗略凸包网格。

**流程：RGB-D 同步采集 → ChArUco 位姿与质量筛选 → 多帧融合 → 背景清理 → PLY / OBJ 导出。** 不需要机械臂、手眼外参或转盘反馈，不包含 FoundationPose 推理和抓取控制。

## 输出与能力边界

| 文件 | 内容 |
|---|---|
| `object_cleaned_centered.ply` | 带 RGB 颜色的点云，可在 CloudCompare 等工具查看 |
| `object_foundationpose.obj` | 面绕序朝外的凸包三角网格，米制、无纹理 |
| `model.yaml` | 尺寸、坐标原点、相对文件名和建模参数 |
| `quality_report.yaml` | 点数、包围盒和离群点清理统计 |

OBJ 是**凸包**，会填平凹槽、孔洞和内腔，不是高精度表面重建，也没有纹理贴图。可用于后续位姿估计的初步模型输入，但实际效果需要在目标物体上验证。板上多个物体、低矮物体、细薄结构、透明或反光表面需要特别检查裁剪和深度质量。

## 安装

基线环境为 Ubuntu 22.04、Python 3.10、OpenCV 4.5.x；实时采集使用 ROS 2 Humble。离线融合不需要 ROS，也不需要 GPU。

### 只运行离线示例 / 建模

进入本仓库根目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install '.[test]'
object-modeling --help
```

依赖由 `pyproject.toml` 声明：NumPy、SciPy、PyYAML、带 ArUco/ChArUco 的 `opencv-contrib-python`；测试额外使用 pytest。项目目前约束 OpenCV `<4.6`，保留原工程的标定板姿态约定；升级前须重新验证板坐标轴及裁剪方向。不要在同一环境叠装不同的 OpenCV wheel。

### 连接 ROS 2 采集

先安装 ROS 2 Humble 和相机驱动，然后准备系统 Python 依赖：

```bash
sudo apt update
sudo apt install python3-pip python3-setuptools python3-opencv python3-numpy \
  python3-scipy python3-yaml ros-humble-cv-bridge ros-humble-message-filters
source /opt/ros/humble/setup.bash
```

采集优先使用系统 Python 和系统 OpenCV，与 `cv_bridge` 保持一致；退出上述离线虚拟环境后运行。脚本通过本仓库的 `src/` 加载代码，不强制安装 Python 包：

```bash
bash scripts/capture_d405.sh datasets/my_object
```

也可在依赖已经齐全的环境执行 `bash install_editable.sh` 安装开发版本。该脚本使用 `--no-deps`，只安装项目本身，不负责安装依赖。包装脚本不再自动读取个人电脑的比赛环境文件，ROS 环境由使用者显式加载。

## 五分钟离线示例

仓库提供 `object_003` 原始采集的前 8 帧彩色图、深度图、内参和板位姿。安装完成后执行：

```bash
object-modeling validate --dataset examples/object_003_subset
object-modeling build \
  --dataset examples/object_003_subset \
  --config configs/d405_charuco.yaml \
  --output outputs/example
python -m pytest tests -v
```

本次本地运行得到 13,183 个点、166 个网格顶点、328 个三角面。依赖版本变化可能造成少量数值差异，因此测试主要检查有效输出和几何性质。

验证器提示“不足 30 帧”是预期现象：这个示例只验证数据到模型的完整通路，视角有限，不代表完整扫描结果。完整约 1.8 GB 历史数据未收入源码仓库。

## 实际采集与建模

### 1. 准备固定场景

默认板参数在 `configs/d405_charuco.yaml`：12 × 9 个方格、15 mm 方格边长、11.25 mm 标记边长、字典 `DICT_5X5_100`。实际打印尺寸必须与配置一致，本仓库不包含原始打印板文件。

物体放在板附近的配置裁剪范围内，并与板刚性固定。移动相机时不要移动物体或板，避免手部长期停留在物体区域。板必须有足够标记可见，物体表面需要相邻视角重叠。

当前裁剪保留板坐标系 `z > 0` 一侧的物体。先检查少量数据的位姿和点云方向，再采集整圈。默认配置针对原工程物体，不能盲目套用于所有尺寸。

### 2. 启动 D405 驱动

驱动由使用者单独安装和启动，需要提供：

| 默认话题 | 消息类型 | 要求 |
|---|---|---|
| `/camera/d405/color/image_raw` | `sensor_msgs/msg/Image` | 彩色图像 |
| `/camera/d405/aligned_depth_to_color/image_raw` | `sensor_msgs/msg/Image` | **已对齐彩色图像**的深度图 |
| `/camera/d405/color/camera_info` | `sensor_msgs/msg/CameraInfo` | 对应分辨率的彩色内参 |

同分辨率不等于已完成深度对齐。深度和颜色必须对应相同像素，时间戳同源且非零。默认深度比例 `0.001` 表示 uint16 每单位 1 mm；若相机设置不同必须修改。浮点深度按米转换为 uint16 保存。

采集订阅采用默认 reliable QoS；如果驱动只发布 best-effort，需要匹配 QoS 后再采集。相机标定参数和分辨率应在一次采集过程中保持不变。

### 3. 绕物体采集

```bash
source /opt/ros/humble/setup.bash
bash scripts/capture_d405.sh datasets/my_object --max-frames 300
```

预览窗口显示 ACCEPT / REJECT 及原因，`q` 或 `Ctrl+C` 结束。达到 `--max-frames` 后自动退出。**输出目录必须为空**，避免覆盖已有数据；继续采集请换新目录，目前不支持断点续采。

默认每 230 ms 尝试接受一帧，检查角点数、重投影误差、角点图像覆盖、清晰度、有效深度比例和 RGB-D 时间差。方位角 / 仰角分区限制重复视角，不等于严格的物体表面覆盖检测。

常用覆盖参数：

```bash
bash scripts/capture_d405.sh datasets/my_object_second \
  --interval-ms 500 --min-corners 16 --max-frames 200
```

话题可用 `--color-topic`、`--depth-topic`、`--camera-info-topic` 修改；也可编辑配置，或通过 `OBJECT_MODELING_CONFIG` 指定另一个 YAML。默认显示窗口，无桌面环境时通过 Python CLI 运行并省略 `--show`：

```bash
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" python3 -m object_modeling.cli capture \
  --config configs/d405_charuco.yaml --output datasets/headless --max-frames 100
```

### 4. 检查和建模

```bash
bash scripts/validate_dataset.sh datasets/my_object
bash scripts/build_model.sh datasets/my_object --output outputs/my_object
```

也可以使用安装后的 `object-modeling validate/build` 命令。验证报告是检查提示，不是自动验收；应检查可用帧、视角覆盖及实际点云。默认融合只使用至少 12 角点的帧，可通过 `build --min-corners` 调整。

处理顺序为：针孔深度反投影、变换到板坐标系、空间裁剪、每帧体素降采样、多帧体素支持度筛选、板面/最大连通物体清理、最终降采样、离群点过滤和凸包导出。

| 主要参数 | 含义 |
|---|---|
| `voxel_size_m` | 点云降采样体素边长，默认 2 mm |
| `board_z_min_m` / `board_z_max_m` | 板坐标系的高度裁剪 |
| `board_xy_margin_m` | 板边界外保留的 XY 范围 |
| `min_voxel_views` | 同一支持体素需要的不同帧数，默认 2 |
| `support_voxel_size_m` | 支持体素边长，默认 6 mm，不是球形搜索半径 |
| `board_plane_clearance_m` | 删除低于此高度的点，默认 12 mm |
| `component_seed_z_m` / `component_eps_m` | 物体种子高度与连通网格尺寸 |
| `base_xy_margin_m` | 核心物体 XY 范围外扩尺寸 |

背景清理保留最大连通核心及其 XY 包围范围内的部分低处点，**不会恢复已被高度裁掉的真实底部**。低矮物体应调整阈值；`min_voxel_views: 1` 可关闭多帧筛选，但也会保留更多噪声。固定背景重复出现时仍可能通过多帧筛选。

融合使用针孔反投影，沿用原工程实现，不做畸变模型专用深度反投影、相机位姿优化或 TSDF 重建。精度依赖相机内参、深度、板检测和视角质量。

## 数据与坐标

```text
dataset/
├── color/000000.png        彩色图
├── depth/000000.png        uint16 深度图
├── overlays/              可选检测预览
├── poses.csv              每帧 board_T_camera 和质量指标
├── camera_info.yaml       内参与 depth_scale
└── manifest.yaml          采集配置和覆盖记录
```

`board_T_camera` 将相机点变换到板坐标系。`poses.csv` 的 `tx,ty,tz` 单位为米，四元数为 `qx,qy,qz,qw`。模型原点移到清理后点云的包围盒中心，坐标轴仍平行于板坐标轴；`model.yaml` 中 `board_T_model_translation_m` 记录这个平移。不能直接把模型原点当作抓取 TCP。

```python
from object_modeling import build_model, validate_dataset

report = validate_dataset("examples/object_003_subset")
result = build_model(
    "examples/object_003_subset",
    "configs/d405_charuco.yaml",
    output="outputs/api_example",
)
print(result.point_count, result.extents_m)
```

## 测试与仓库结构

```text
src/object_modeling/    采集、融合、验证及命令行实现
configs/                默认板、相机和建模参数
scripts/                采集 / 建模 / 检查快捷脚本
examples/               8 帧真实离线示例
tests/                 原有算法测试和新增流程回归测试
.github/workflows/      GitHub 上自动运行离线测试
docs/                  来源、修改说明及原 README 存档
```

`tests/` 是实际测试代码；`.github/workflows/offline-tests.yml` 负责在提交更新时安装项目并运行它。使用 pytest，避免用 unittest 导致未发现这些函数式测试。测试不连接相机或机器人；本次仅验证本地离线流程、安装包与 ROS 模块导入，尚未重做实时硬件采集。GitHub 工作流要上传后才会运行。

## 上传 GitHub

将**本目录内部内容**作为独立仓库根目录，建议仓库名 `d405-object-modeling`。`.gitignore` 排除采集数据、运行输出、缓存和安装产物，保留小型 `examples/`。网页上传时隐藏的 `.gitignore` 和 `.github/workflows/offline-tests.yml` 可以用 Create new file 创建；ZIP 是方便分发的附件，不替代源码上传。

本次未替作者选择开源许可证。需要授权他人复用时，请在发布前选择并添加合适的 `LICENSE`。
