# EgoNavForge 大模块与 I/O 约定

对齐 `OPEN_SOURCE_FRAMEWORK.md`：YAML 拼工作流；模块用 `type` 注册；模块之间只靠 `EpisodeContext` 对接。

思路类似 MMLab：**换算法 = 换/加模块 + 改 YAML**，不为每个数据集重写流水线。

## 上下文（对接总线）

`EpisodeContext` 是唯一跨模块载体：

| 字段 | 谁写 | 谁读 | 含义 |
|------|------|------|------|
| `streams` | Ingest | Preprocess / Trajectory / Export | 命名数据流路径字典 |
| `pack` | Ingest（可被 Export 覆盖） | Export | 进数据集的流名列表 |
| `meta` | 各模块 | 各模块 | stride、prepared_dir、ann_path… |
| `extras` | Preprocess 等 | Trajectory | mask / yolo 配置 |
| `suite_dir` / `poses` | Trajectory | Scale / Export / Viz | 重建结果 |
| `export_meta` | Export | Runner 汇总 | 单集导出信息 |

## 已封装大模块（官方）

### 1. 加载 — `ParsedRootLoader` / `DictStreamLoader`（episode_start）

- **输入**：`root` + `stream_files`（或绝对 `streams`）
- **输出**：`ctx.streams`、`ctx.pack`、为轨迹准备的 `prepared/rgb.mp4`（指向 `recon_image_key`）
- **不负责**：解畸变算法本身（若 dump 已是去畸变视频则直接用；以后可加 `Undistort` 预处理模块）

单路 / 三路只改 `stream_files` 与 `pack`，**不换加载类、不换轨迹类**。

### 2. 预处理 — `FixedGripperMask` / `YOLOPersonFilter`（frame）

- **输入**：`image_key` → 对应 `streams[...]`；mask/weights 路径
- **输出**：`ctx.extras['mask']` / `ctx.extras['yolo']`（供轨迹模块在抽帧时应用）
- **可替换**：用户可注册 `SAMDynamicFilter` 等同阶段模块，保证仍写约定好的 `extras` 或直接改帧缓存即可

### 3. 轨迹 — `Pi3XRecon`（chunk）≡ 文档中的 Pi3XRecon

大模块，内部步骤（实现细节，可不在 YAML 展开）：

`ChunkScheduler → Pi3Infer → ChunkStitch(pose-aware Sim3) → PointMerge`，并在 `rgb_moge` 模式下带 MoGe 定尺。

- **输入**：`image_key` 一路图；`chunk_size` / `overlap`；mask/yolo extras
- **输出**：`ctx.suite_dir`、`ctx.poses`（米制 `pi3_poses_metric.npy`）
- **可替换**：用户注册 `MyMultiViewPose`，只要写出同样的 `poses` / `suite_dir` 约定，下游 Export 不用改

### 4. 深度 / 尺度 — `MoGeDepth` / `MoGeScale`（after_pose，可选）

文档要求深度与尺度可拆开以便对照。当前默认工作流里 **尺度已在 `Pi3XRecon(rgb_moge)` 内完成**；这两个类型已注册，需要「双尺度对照」时再挂进 YAML。  
以后接双目时：换/加 `StereoDepth` + `StereoDepthScale`，轨迹模块可不动。

### 5. 导出 — `LeRobotExport`（finalize）

内部步骤：`GravityOrImuAlign → PoseToXYZYaw → SquatZMap → PackObservations → WriteLeRobotLayout`。

- **输入**：`poses` / `suite_dir`；`pack` 流；可选 squat `ann_path`
- **输出**：`work_dir/lerobot/`（parquet + videos + meta）
- **要点**：`pack` 与轨迹读哪些流独立——Pi3 一路，数据集可三路

### 6. 可视化 — `TrajXYZViz`（finalize）

- **输入**：`suite_dir` 位姿  
- **输出**：`work_dir/viz_xyz/*.png`

### 7. 场景孪生 — `IndoorTwinScene`（finalize，可选）

室内数字孪生（vendored `backends/indoor_twin_agent`，资产库检索，替代 v37）：

- **输入**：`streams[image_key]` 视频；`ctx.poses` / `suite_dir/.../pi3_poses_metric.npy`；内参（`intrinsics_path` 或 episode `calib.json` + `calib_camera`）；可选 `points_metric_textured.ply`（`use_pointcloud: true` 且文件存在才传入）；VLM `api_base` + `api_key_env`；检索库 `backends/asset_bank`
- **输出**：`work_dir/episodes/<ep>/scene_twin/Scene/IndoorTwin.blend`、`Scene/Scene.usda`、观察与检索 JSON
- **开关**：工作流默认 `enabled: false`；数据集用 `dataset.scene_twin: {enabled: true, ...}` 打开
- **检查**：`check_only: true` 只写轨迹和内参，不调 API、不启动 Blender

## 文档有、尚未做完整产品的模块

| 文档名 | 状态 |
|--------|------|
| ObjectMemory | 未实现（accumulate 预留） |
| SceneBuild / VLNVersePackager | 由 `IndoorTwinScene`（资产库 agent）覆盖场景侧；VLNVerse 专用打包仍可另加 |
| TrajCompareViz / EgoPointCloudViz / QualityGate | 未做成独立 type（旧脚本留在 pi3_moge_lerobot，不暴露为本仓库模块） |
| SensorDepthScale / StereoDepthScale | 未注册 |

## 用户加新方法（MMLab 式）

```python
# my_ext.py
from ego_nav_forge.modules.base import TrajectoryModule
from ego_nav_forge.registry import register

@register("MyMultiViewPose")
class MyMultiViewPose(TrajectoryModule):
    def on_chunk(self, ctx, chunk_index, span):
        # read ctx.streams[...]
        # write ctx.poses, ctx.suite_dir
        ...
```

YAML 里把 `Pi3XRecon` 换成 `MyMultiViewPose`，加载 / 导出可不动。

## 配置怎么拆

| 文件 | 内容 |
|------|------|
| `configs/workflows/pi3_moge_lerobot.yaml` | 阶段 + 模块列表（工作流） |
| `configs/datasets/*.yaml` | 只绑 `dataset:`（流名、pack、mask、路径） |

不为 UMI / 三视角各写一套代码或两套完整 pipeline。
