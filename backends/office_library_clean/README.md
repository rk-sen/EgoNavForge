# office_common 外部来源资产库 clean-v3.1

本库用于“先检索、后装配”的室内数字孪生流程：优先从 USD 资产库直接取模型，只有检索不到时才退回 Blender 程序化建模。

## 规模
- **83 个规范化 USD**
- **249 张缩略图**（front / iso / top，512 PNG）
- 覆盖用户要求的 **20 个小类**，含沙发、椅凳、桌柜、置物架、包装、冰箱、运动球/隔断、盆栽、标识、门窗、机器人、灯具
- `category_path` 为受控分类；`tags` 与 `search_text` 用于 FTS5 检索
- 每个资产带 `asset_meta.json` 元数据、`world_bbox`、`material_profiles`、来源 URL/许可

## 规范
- Z-up，meters_per_unit = 1.0
- 原点：包围盒底部中心
- Front：-Y
- USD 导出保留材质/纹理；缩略图直接渲染导出的 USD，不覆盖材质

## 目录
```
office_library/
├── usd/                 # 规范化 USD
├── thumbs/              # 三视图 PNG
├── raw_glb/             # GitHub/生成类 GLB 源文件
├── raw_usd/             # Poly Haven 原生 USD + 贴图
├── asset_meta.json      # 资产元数据
├── catalog.json         # 检索名录
├── catalog.sqlite       # SQLite + FTS5
├── CATALOG_SUMMARY.json
├── COVERAGE.md          # 20 类覆盖矩阵
├── VALIDATION.json
└── source/              # 来源、SHA256、许可
```

## 检索示例
```bash
python query_office_library.py --q "三人沙发"
python query_office_library.py --category furniture/sofa
python query_office_library.py --q "ergonomic office chair" --limit 5
```

## 主要来源
- **Poly Haven** — 55 个原生 USD 资产，CC0-1.0；1k 贴图，来源逐项写入 `source/asset_sources.json`
- **GitHub 公开仓库** — 办公场景 GLB（ivoirh/dp-sm-vr、fthuggy/aframeproject、jj-143/assets、psandis/office-3d-exp、threelab/web3d-file-server、raghava7261/3D-Models 等），许可按上游标注
- **URDF** — Franka Panda、UR5、ANYmal 由公开 URDF/DAE 在 Blender 中按零位装配后转 USD/GLB
- **程序化补缺** — 安全出口 / 灭火器 / 禁烟标识、PC 主机、显示器、双门冰箱、中国传统板凳；均为本项目生成，CC0-1.0

## 说明
- 因体积上限再删除 7 个外部资产，并已从名录/元数据/缩略图同步移除；另已删除 40 个程序化生成、V37/V39 场景派生、材质变体和自行装配 URDF 资产；只保留外部下载与用户提供人物 `v39_person_supplied`。`REMOVED_GENERATED.json` 是删除清单。

本库是 **asset/variant 库**，`placements=[]`，不保存场景摆放；装配阶段按房间布局实例化。
- 部分 GLB 资产的单位/比例由源文件决定，已在规范化阶段做统一缩放；详见每个资产的 `bbox_m`。
- 目标不是“凑数量”，而是让检索器能按类别、尺寸、风格直接命中；每个小类尽量提供 3–4 个可替代资产。
