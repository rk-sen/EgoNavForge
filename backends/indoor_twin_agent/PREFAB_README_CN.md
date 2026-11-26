# IndoorTwin Kimi 预制资产与位姿优化版

解压两个档案到同一位置。它们共享顶层目录 `IndoorTwin_Prefab_Refinement`。代码和四个案例在 `Code` 和 `Cases`，资产库在 `AssetBank`。各案例的 `Scene/Scene.usda` 使用相对引用，场景主文件和资产库的目录关系需要保留。Blender 项目另存为 `Scene/IndoorTwin.blend`。各案例下有原始样本输入、关键帧左右对照、相机、导航和模型审阅日志。

## 安装与复跑

使用 Python 3.11/3.12、Blender 4.3.2 或兼容版本、FFmpeg/FFprobe。安装 `requirements.txt`。通过本机环境变量 `KIMI_API_KEY` 配置代理密钥。代码不保存密钥。其余推理使用几何、Python 和 Blender。

从共同顶层目录执行：

```bash
python -m pip install -r Code/requirements.txt
python Code/agent.py --video Cases/lab/Scene/Reference/source_video.mp4 \
  --trajectory Cases/lab/Inputs/trajectory.json --intrinsics Cases/lab/Inputs/intrinsics.json \
  --pointcloud Cases/lab/Inputs/cloud.ply --catalog AssetBank \
  --output Runs/lab_retest --frames 9
```

将案例名换成 `room`、`desk` 或 `household` 可复跑 TUM 三段。复跑读取同一个库，已发布的新资产即可参与下一次检索。就地修改库前可复制索引和资产目录作实验快照。使用新输出目录以避免覆盖既有场景结果。

## 内容和约定

| 阶段 | 实现 |
|---|---|
| 多视角证据 | Kimi 识别物体、紧致框、支撑关系、遮挡标记。同帧分离的对象不因类别相同被合并 |
| 资产匹配 | 比较语义、观测尺寸、公共资产缩略图和轮廓。匹配通过时直接引用一个完整资产。架构壳使用闭合所需的精确模块 |
| 位置和朝向 | 根据标定相机的投影框和 PLY 点局部支持优化 XY、偏航及尺寸。地面根维持落地。遮挡框只施加单向覆盖约束。至少三个完整视角时做一帧留出验证 |
| 支撑 | 支撑件先拟合。桌上件通过共视射线与面积约束匹配桌面，其根结点高度贴合支撑件 |
| 人体 | 使用下载的 Kenney 骨架与皮肤，原模型摆成自然站姿并整组烘焙，统一调整实例根位置、偏航和比例。只看到头肩时优先使用接近给定位姿锚点的相机帧。用于场景中姿态近似，人体关节动作和动态轨迹尚未估计 |
| 新物体 | 未命中才请求部件配方。小缝通过接合修复，结构距离超阈值的方案改为单一连通代理。所有部件烘焙到统一米制根 |
| 输出资产 | `Asset.usda` 是带碰撞代理的 USD 入口，支持引用。 `Prototype.blend` 保存可追加的集合。预制单件保留原部件和 PBR 绑定 |

碰撞和装配接合使用保守包围体，因此精确接合、重叠和关节物理需要在最终应用中进一步验证。复跑报告中的 IoU 来源于观测框，与实测三维物体位姿误差是不同指标。

## 从公开来源重建库

```bash
python Code/fetch_public_assets.py Downloaded
blender -b -t 3 --python-exit-code 4 --python Code/import_public_blender.py -- --sources Downloaded --staging Imported
python Code/publish_public_assets.py Imported NewAssetBank
```

`AssetBank/PublicInventory.json` 记录来源、原始输入哈希、许可和尺寸。 `Inventory.csv` 还包括用于闭合和布局的本地模块，以及四次复跑发布的物件。Poly Haven 的模型均以 CC0 发布。Powered by Poly Haven。Kenney 入库来源为公开 Furniture Kit 和 Animated Characters 1。Innerscene 补充窗帘和篮球。原始人物 FBX、皮肤和许可另在 `AssetBank/HumanSource`。

```bash
python Code/benchmarks/assembly_regression.py
python Code/benchmarks/verify_refinement.py Cases/lab AssetBank
```
