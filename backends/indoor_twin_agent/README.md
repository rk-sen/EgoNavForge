# IndoorTwinAgent

用 Kimi K2.6 做室内视频的多视角理解、资产判断和渲染审阅，用 Python、Blender 和 OpenUSD 做可验证的度量建模与物理导出。

## 运行环境

- Python 3.11 或 3.12，安装 `requirements.txt`。本次使用 Python 3.12 和 Blender 4.3.2。
- Blender、FFmpeg、FFprobe 可在 PATH 中找到。Windows 可用 `--blender "C:\Program Files\Blender Foundation\Blender 4.3\blender.exe"` 指定。
- 使用环境变量 `KIMI_API_KEY`。默认接口为 `https://ai.zj-computility.com/maas/v1/chat/completions`，模型 `kimi-k2.6`，可用 `--endpoint` 指定同协议的 API 基地址。唯一在线模型调用是此接口。FFmpeg 用于媒体解码和 PTS，NumPy/SciPy 做度量计算，Blender 做程序建模，OpenUSD Python 负责场景组合、物理属性与验证。

Linux/macOS 示例：

```bash
python -m pip install -r requirements.txt
read -rs KIMI_API_KEY
export KIMI_API_KEY
python agent.py --video video.mp4 --trajectory trajectory.json \
  --intrinsics intrinsics.json --pointcloud cloud.ply \
  --catalog catalog --output runs/first
python agent.py --video video.mp4 --trajectory trajectory.json \
  --intrinsics intrinsics.json --pointcloud cloud.ply \
  --catalog catalog --output runs/second
```

PowerShell 可用 `$env:KIMI_API_KEY = Read-Host "Kimi API key"` 设置当前进程环境变量。每次用新 `--output`，跨场景重复使用同一 `--catalog`。长视频增大 `--frames` 以覆盖更多观察位置。省略 `--pointcloud` 时通过相机上方向、地面先验和图像落地锚点估计位置。相机高度已知时可传 `--camera-height`（米），默认 0.65 m 是弱先验。若高度、视线或深度不受约束，估计误差可能较大。

### 输入协议

| 输入 | 支持格式 |
|---|---|
| 室内视频 | FFmpeg 可解码的视频。PTS 根据第一张解码图像归零 |
| 轨迹 | 本样例 Pi3 JSON，或下方时间戳 JSON。位置为米，至少两个有效完整位姿 |
| 内参 | `{"K":[[fx,0,cx],[0,fy,cy],[0,0,1]],"width":672,"height":478}` 或样例原格式。视频应已去畸变 |
| 点云（可选） | PLY，XYZ，RGB 可选，点云和轨迹共世界坐标。支持 ASCII 和常见二进制顶点编码 |

时间戳轨迹示例：

```json
{
  "units": "meters",
  "convention": "blender_c2w",
  "poses": [
    {"time_s": 0, "matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0.6],[0,0,0,1]]},
    {"time_s": 1, "matrix": [[1,0,0,0.5],[0,1,0,0],[0,0,1,0.6],[0,0,0,1]]}
  ]
}
```

也支持 `opencv_c2w`、`opencv_w2c`。相机矩阵是逐行 4×4，第 4 列为平移。基线轨迹采样单位和视频时间偏移需要提前对应。Pi3 样例按 `row × image_stride` 对到解码索引。

## 输出与资产库

- `runs/<job>/Scene/IndoorTwin.blend` 用于继续编辑、观看相机漫游和检查同机位渲染。
- `runs/<job>/Scene/Scene.usda` 是物理仿真入口。默认根 `/World`，Z-up，米与千克，重力 -9.81 m/s²，含动画相机、独立实例、物理材质和碰撞代理。优先打开这个文件。
- `catalog/AT_<id>/Asset.usda` 是一个可独立加载的数字资产，默认根 `/Asset`。同目录含 `Geometry.usda`、`Materials.usda`、PNG 纹理、`Prototype.blend`、缩略图与元数据。
- 资产在主场景以视觉引用实例化。每件物体根 Xform 可单独移动，视觉比例和物理碰撞代理分开作者化。调整到新物理尺度后应重新导出相应碰撞范围与质量。
- 每个任务记录 `InputAudit.json`、`Observations.json`、`AssetQueries.json`、`Retrieval.json`、`RunSummary.json`、模型的请求文本和响应、源图与缩略图。场景目录含 `SimulationManifest.json`、`AlignmentLog.json`、`Validation.json` 和 `VideoFrameMap.csv`。
- 资产库用 SQLite 索引。用 `Metadata.json` 查看语义类别、几何指纹、外观定义、摩擦、弹性、质量先验、碰撞范围与分支来源。备份时保留索引与所有资产目录。

### Isaac Sim

在 Isaac Sim 用 **File → Open** 打开 `Scene.usda`，检查 `/World/Physics` 后进入 Play。地面、墙、固定家具使用静态碰撞体。球、抱枕、部分盒子与设备例子使用动态刚体。物品可作为独立 USD 引用给其它场景。导出的纹理以 UsdPreviewSurface 表示，摩擦/密度/回弹使用独立物理材质。

本次的额外运行验证使用 NVIDIA PhysX CPU SDK。在另一个虚拟环境可执行：

```bash
python -m pip install -r requirements.txt -r requirements-physx.txt
python physx_smoke.py runs/first/Scene/Scene.usda --steps 120 --output physx.json
```

完整 Isaac Sim 的 Kit 启动、机器人接触与传感器测试需在有 Isaac Sim 的机器运行。碰撞代理和物理数值为模拟先验，动态物品的控制与材质参数可按目标任务标定。

## 复现三个受控分支

`benchmark/lab_seed.json` 是上一轮经关键帧比较修正过的样例测量布置。它控制位姿和视角，使本次实验单独检验资产库、材料和导出。先以空目录运行，第二次使用同目录资产库；在首次命令的基础上仅改 `--output`。两个受控实验都追加：

```text
--survey benchmark/lab_seed.json --frames 2 --samples 44 --threads 8
```

第三次使用同一资产库，并追加 `--finish-override benchmark/finish_override.json --skip-review`，将球的绿色橡胶改成蓝色，同样的球形几何和质量模型。对原始输入的自动实验省略 `--survey` 以及 `--finish-override`。实验中的 `--resume-observations` 只在该任务目录已成功保存独立真实 API 输出时使用，避免工作进程重启重复计费。

## 代码入口

| 文件 | 功能 |
|---|---|
| `agent.py` | CLI、任务编排、渲染反馈和受约束的修正 |
| `evidence.py` | 解码、PTS、轨迹及内参标准化、可选点云和坐标估计 |
| `kimi.py`、`prompts/` | Kimi 请求、JSON 提示词、原样用户提示词及图片指纹日志 |
| `planner.py` | 图像锚点回投、几何先验、程序化部件配方和建筑包络 |
| `catalog.py` | 本地候选召回、多模态判别、三分支和事务入库 |
| `build_twin.py`、`usd_writer.py` | Blender 组装、PBR、可编辑实例、相机和网格导出 |
| `physical.py`、`physx_smoke.py` | 独立数字资产、组合物理场景、OpenUSD 检查、PhysX CPU 验证 |
