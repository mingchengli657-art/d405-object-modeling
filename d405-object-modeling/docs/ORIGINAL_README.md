# object_modeling

可迁移的 D405 + 单块 ChArUco 板 RGB-D 物体建模库。物体和标定板保持刚性固定，D405 可由操作者手持绕物体移动。程序根据 ChArUco 几何质量、RGB-D 时间差、清晰度、深度有效率和视角覆盖决定是否保存帧，不依赖转盘角度反馈，也不控制 Piper 或转盘。

## 安装

在本目录执行：

```bash
./install_editable.sh
```

安装脚本默认不联网、不重复下载依赖，适合当前代理受限的电脑；它会使用系统或当前虚拟环境中已有的 NumPy、SciPy、PyYAML 和 OpenCV。若迁移到一台尚未准备这些依赖的电脑，再单独安装它们即可。实时 ROS 采集还需要系统已有的 ROS 2 Humble、`cv_bridge` 和 `message_filters`。

## 采集

先单独启动 D405，确认以下话题存在：

```text
/camera/d405/color/image_raw
/camera/d405/aligned_depth_to_color/image_raw
/camera/d405/color/camera_info
```

让“标定板 + 物体”保持不动，手持 D405 依次从侧面、上方绕物体移动，执行：

```bash
./scripts/capture_d405.sh datasets/bottle_001 --max-frames 300
```

脚本名保留为旧名称以兼容已有流程，但现在不需要电子转盘。默认每 230 ms 尝试保存一帧；只有 RGB-D 同步、ChArUco 角点、重投影误差、角点覆盖、清晰度和深度质量均通过的帧才会保存。程序还会按相机在标定板坐标系中的方位和高度限制重复视角。`--show` 已由脚本默认打开，按 `q` 或 `Ctrl+C` 结束。可以通过 `--interval-ms 500` 降低采样频率，或通过 `--min-corners 16` 提高角点门槛。

建议相机光轴相对桌面向下约 30°～40° 采集一圈侧面，再用约 55°～70° 采集顶部；不要强行使用接近水平或完全 90° 的极端角度。单帧只看到物体的一部分没有问题，但每个表面应在多个相邻视角中有重叠。

采集期间不要移动物体或标定板；相机可以手持移动。标定板应和物体刚性固定在防滑底座上，并尽量在每个视角保留一部分完整标记。

数据目录还会生成 `manifest.yaml`，记录采集模式、质量阈值、视角覆盖、话题、时间间隔、深度比例和坐标约定；`poses.csv` 还会记录每个已保存帧的质量指标。其中不保存主机绝对路径。

## 检查、建模和查看

```bash
./scripts/validate_dataset.sh datasets/bottle_001
./scripts/build_model.sh datasets/bottle_001
```

输出在 `datasets/bottle_001/model/`，包括：

```text
object_cleaned_centered.ply  清理后的带颜色点云
object_foundationpose.obj   面绕序一致、法向朝外的轻量凸包网格
model.yaml                  单位、原点、包内文件名和尺寸
quality_report.yaml         点数、包围盒和去噪统计
```

PLY 是主要检查对象，可直接用 CloudCompare 打开。模型包内只使用文件名和相对元数据，不记录机器绝对路径，因此可以整体复制到机器人电脑或队友电脑。

建模阶段还会按多帧体素支持度抑制只出现一次的手部、支架和背景点；默认支持半径为 6 mm、至少由 2 帧观测。若要保留只在单一视角出现的细薄结构，可在配置中把 `min_voxel_views` 改为 `1`。

建模阶段还会自动清理 ChArUco 板平面：先从标定板上方的最大连通物体估计物体投影范围，再删除投影范围之外的近板面点，同时保留物体底部。清理参数位于配置的 `model.board_plane_clearance_m`、`component_seed_z_m`、`component_eps_m` 和 `base_xy_margin_m`；当前默认值适合本次瓶子数据。若要保留原有结果并另建清理版，可指定输出目录：

```bash
./scripts/build_model.sh \
  datasets/object_003 \
  --output datasets/object_003/model_plane_cleaned
```

清理版的 `object_foundationpose.obj` 可直接作为后续 FoundationPose 的模型输入；导出时会自动统一三角面绕序和法向方向，`object_cleaned_centered.ply` 用于 CloudCompare 检查。对薄片、碗底等贴近标定板的物体，应按该物体单独调整 `board_plane_clearance_m`，避免误删真实底部。

## Python API

```python
from object_modeling import build_model, validate_dataset

report = validate_dataset("datasets/bottle_001")
result = build_model(
    "datasets/bottle_001",
    "configs/d405_charuco.yaml",
)
print(result.model_dir, result.extents_m)
```

命令行和 Python API 使用相同的核心实现。FoundationPose 和抓取流程目前没有耦合进来，后续只需读取 `model/model.yaml` 中的 `mesh` 文件即可。

## 当前版本的边界

- 手持采集没有编码器角度反馈，程序使用 ChArUco 恢复的相机方位和高度划分视角区域；覆盖是否完整仍需操作者在预览窗口中确认。
- 当前 OBJ 是保守凸包，适合瓶子、方块等初步定位验证；PLY 用于 CloudCompare 人工检查。
- 后续如有需要，可以增加 TSDF/Poisson 网格后端和更严格的物体分割，但不影响现有数据格式。
