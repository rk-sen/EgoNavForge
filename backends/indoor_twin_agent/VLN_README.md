# IndoorTwinAgent for VLN scene preparation

本版本复用程序化 Blender 资产和 Kimi API。新增外墙闭合、室内隔墙开口验收、远景语义补全、桌面支持关系和 USD 通行检查。结果适用于准备 Isaac Sim/VLNVerse 的场景资产。日志中的导航网格与地标是候选标注，完整 VLNVerse task registry、机器人 embodiment 和语言任务验证在本次执行环境之外。

## 输入与运行

使用 Python 3.11/3.12、Blender 4.3.2 或兼容版本、FFmpeg/FFprobe、`requirements.txt`。API 读取 `KIMI_API_KEY`。轨迹文件为带相对原片首帧时间戳的米制相机矩阵。若有可靠的世界重力方向，在轨迹 JSON 中设 `"world_up":[0,0,1]` 或填写对应单位向量。普通视频若无点云可用相机高度先验，但应降低对墙面和远物定位的信心。

```bash
python -m pip install -r requirements.txt
python agent.py --video source.mp4 --trajectory trajectory.json \
  --intrinsics intrinsics.json --pointcloud cloud.ply \
  --catalog IndoorAssetBank --output Runs/NewCase \
  --frames 12 --nav-radius 0.27
```

提高 `--frames` 可覆盖更长轨迹。将 `--nav-radius` 调整到目标机器人平面外廓。可保留同一个资产库来复用独立 USD 和 Blender 原型。运行参数及旧版数据协议也见 `README.md`。所有在线推理只使用 Kimi。没有使用额外检测、分割、重建、嵌入或生成模型。

## 房型、远区和通行

1. **轴向标定。** 低层点带估计地板，并拒绝有过多有效几何落在其下的平面。点云竖直表面沿水平转角搜索结构轴。地板、相机和点云统一在米制 Z-up。
2. **结构证据。** 某一水平偏移处的多层高度覆盖和连续长度形成墙候选。结合相机路径和云范围形成外壳。提供给 Kimi 的是带墙候选 ID 的俯视图、原始帧、已定位物体和相机位置。模型提出空间功能、边界关系和室内材质候选。
3. **隔墙与门。** 外壁是连续墙段。窗口有实体窗体，外部或未验证去向的门有封闭门扇。内部通行开口需要高贯通竖直面、视频明确门口证据、两侧相机点和单一局部穿越区。若录像只证明一间房，保留一个可闭合区域和未决假设。当前支持正交、单层布局近似。
4. **可见与远景补全。** 物体根据图像和点云定位，显示器、书本等小物优先依附已找到的桌面。候选补全来自室内功能和目视证据，放在远端或遮挡的周边。边界门窗、已有物体、近相机实测空区和轨迹缓冲区域排除添加。植物、储物柜、装饰画、垃圾桶有独立的资产和来源标记。所有补全实例的 `placement_evidence` 为 `inferred_context`，不会加入观测地标候选。模型只能建议类型，几何和占位仍由本地校核。
5. **物理通路检查。** 使用实际 USD 碰撞体世界边界进行域内膨胀栅格、连通组件、房间安全点和封墙探针。路迹检查使用两种口径：原相机三维中心碰撞、手持路径在地面上的机器人宽度可走率。手持拍摄时可能探过桌面，后一数字通常不能当作机器人导航成功率。
6. **复用。** 先判几何和外观，优先复用，表面不同则换材质，需要新形体时按验证后的部件配方构建。猜测的摆放位姿与资产库中的规范几何分别保存。

## 输出接口

| 文件 | 用途 |
|---|---|
| Scene/Scene.usda | Isaac Sim 导入入口及物理资产 |
| Scene/IndoorTwin.blend | Blender 编辑和原时间轴回放 |
| Layout.json | 房间矩形、墙面来源、隔墙、门窗状态、补全来源 |
| NavigationGrid.npz | 平面米制轴、保守可走栅格、组件标签和净距 |
| Navigation.json | 身体半径、拓扑、室内安全点、门联通和语义候选 |
| NavigationMap.jpg | 通行区域、相机路径与补全影响的俯视图 |
| ModelLogs/ | 实际 Kimi 请求与公开响应 |
| Catalog/.../Asset.usda | 单件可引用物理资产和其 PBR 材质 |

机器人安全点参考 Z=0 地面。Isaac Sim 装载时需要根据机器人原点高度和碰撞半径转换。导航检查使用初始静态布置下的碰撞代理包围盒，无法替代动态规划或机器人接触试验。

## 基准与回放

`benchmarks/tum_clip.py` 使用 TUM RGB-D 片段、标定、实测深度和动作捕捉位姿，生成适配器输入。官方数据下载见 [TUM RGB-D Benchmark](https://webshare.cvg.cit.tum.de/g/rgbd/dataset/)。输入模型仅接收 RGB 关键帧与局部摘要，不接触完整测试真值资产。

```bash
python benchmarks/layout_regression.py
python benchmarks/walkthrough.py --video source.mp4 \
  --blend Runs/NewCase/Scene/IndoorTwin.blend \
  --output comparison.mp4 --work scratch/replay --fps 6
```

同机位对照片右侧按左侧被选中帧的实际 PTS 渲染，另附逐帧索引表。实验使用 6 fps 以减少逐帧渲染成本，原始轨迹和 Blender 相机动画保留全部样本。三段公开视频主要覆盖单房间或一个台面区域，不足以证明真实多房间的恢复精度。隔墙分支另有可复现数值回归用例。
