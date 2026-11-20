# indoor_twin_agent v14.0

用「一个多模态大模型 + 本地 Blender/ffmpeg 执行器」把走拍视频 + 轨迹 + 内参，变成可编辑的
室内数字孪生。外部只调用 Kimi K2.6（或任意 OpenAI 兼容多模态 API）；建模、渲染、几何
检查、像素对比都在本机完成。

## v9 修的是什么

v8 能跑通，但实跑暴露了五类“看起来像、其实不对”的问题：

1. **场景理解**：只认 10 个模板，背景架子上的东西、台灯/鼠标/冰箱/饭桌/百叶等没有类别；
2. **尺度**：`bs_layout` 从 `calibration.room_bbox_m` 取房间尺寸，plan 的 architecture 被忽略，
   轨迹长度被误当户型；房间偏小；
3. **材质**：`fabric` 生成竖条纹，地板 `node_brick` 像金属贴片；
4. **摆放**：物品都塞进同一个 cell；只有“单件正确”，没有同层 AABB 互穿检查；
5. **拓扑**：靠枕只是倒角方盒，无法表达变形软垫。

v9 的落点：

- **30 类语义模板**（`blender_lib/bs_assets.py`）：sofa/barrier/desk/table/dining_table/bench/
  task_chair/office_chair/ergonomic_chair/pressure_stool/stool/monitor/laptop/keyboard/mouse/pc/
  fridge/cabinet/drawer_cabinet/shelf/box/cardboard/ac/lamp/blind/ball/cushion/pillow/beanbag/
  robot/light/door/safety_door/window/default。部件先 parent，再 join 成**一个 mesh**。
- **关系摆放**：`relation: {type: on|near|against, target, offset_m}`。显示器/键鼠/台灯落桌面，
  椅子在桌边，架子/柜子靠墙；`scaffold_assets` 按目标尺寸和 `size_m` 自动算 z/偏移。
- **尺度检查**：同层 AABB 互穿检查（`bs_stage.placement_report`），`on` 支撑对豁免；
  重叠会进入 `state.assets.placement.overlaps`，S3 门禁直接失败并列出 `a×b`。
- **语义材质**：`bs_mat.preset(family, color)` 只调颜色，布料/皮革/橡胶/木/金属/屏/塑料/纸板/
  玻璃/涂料各一套；不再生成条纹贴图。`floor_rubber()` 是近乎同色的大块橡胶地面，不用砖格。
- **点线面**：`bs_geo.pillow()` 两片 8×8 网格 + 边缘下垂 + 正弦皱褶；`soft_sphere()` 做豆袋/
  靠垫；不是缩放立方体。模板可 `patch_file` 改 `build_<kind>` 里的顶点/面。
- **户型**：`bs_layout.build()` 优先读 plan 的 `architecture.room_bbox_m` / `ceiling_z`，**不再
  回退轨迹 bbox**。窗支持 `blinds`（百叶），门叶加把手；`check_openings` 只数 `*_glass` 和
  门叶，不再把 `Light_Window` / 门框算成门窗。
- **相机**：`init_scene()` 先放 1 个 `Cam_Overview`（看整间户型/场外物体）+ 6 个 `Cam_Traj_*`；
  壳体、物品、QA 每个阶段都有非平渲染可看，不等到最后才放相机。





## V14：窗帘不是默认百叶，灯条按视频长度/数量，组件必须接触

- **逐窗窗帘证据**：`bs_visual_struct.inspect_window_crop` 新增 `curtain={draw,side}`，用低饱和亮面占比 + 竖向褶皱证据与 `blind` 分开；`bs_floorplan` 按每个窗的 `curtain` 字段生成米白布面、褶皱、帘杆和挂钩。实际场景中 N1/N3 两扇窗是半拉窗帘，其余是 plain glass，不再统一百叶。
- **长条灯**：`architecture.lights={count:6,length_m:3.2,rows:2,mode:"long_strip"}` 由 plan 驱动；`bs_floorplan` 生成 6 条 3.2 m 灯体 + 两端吊杆 + 面光源，替代原来的“每 2.2 m 一个短盒”。
- **组件连接**：桌子 top/legs/modesty/brace 的接触面按高度计算；显示器 panel/neck/base 在桌面上，屏幕旋转按度数；椅子 gas column 贯通座垫与五星脚；单件对象图重新跑 17 件，全部 joined。
- **复杂网格接口**：圆润件使用 `bs_geo.pillow` 的 bulged grid 与圆辊臂；窗帘用多段折面表达褶皱，提示词要求模型为看得清的物体生成这种更细的脚本，而不是直接复制模板。
## V13：图像决定结构、时序决定布局、碰撞先行

- **窗户不再默认百叶**：`blender_lib/bs_visual_struct.py` 对窗裁块做亮度/水平带/竖线统计，输出 `plain_glass / blind_window / curtain` 候选、`blind`、`mullions`、frame/bbox/confidence。实际这条视频的窗户是连续亮玻璃，plan 全部 `blind=none`，不再强行加“上半天窗百叶”。
- **沙发圆润化**：`build_sofa` 改成 pillow 软体底座 + Y 轴圆滚扶手 + 座/背靠垫 + 顶部圆枕 + 短脚，保留 2.0×0.9×0.8 m 尺寸但轮廓不再是四个倒角盒子。
- **显示器组件位姿修复**：`bs_geo.plane` 接收的是**度数**，V12 的 `build_monitor` 又传了 `math.radians(90)`，导致屏幕被转了 90 弧度、变成横插的矩形片；V13 改为 `rot_xyz=(90,0,0)`，实测单件屏幕已经竖在 bezel 前面。
- **桌面色来自图像证据**：`inspect_flat_color` 可对桌面裁块取中位色；spec 支持 `material="paint", color="white"`，desk 不再因为“desk” token 被强制套木纹。实际白桌 `O5/O13` 已改为白色。
- **相机时序布局**：`blender_lib/bs_timeline.py` 的 `annotate(plan,path)` 给每个实例写 phase/side/bearing；`collision_pairs()` 报 AABB 相交；`push_out_of_barriers()` 把货架/桌子从屏障带推出去。实际把货架/纸箱移到屏障后侧、桌 2 深度缩到 0.6 m，并把豆袋从西侧屏障边挪开。
- **实际 QA**：没有家具插进屏障的硬碰撞；报告只保留“显示器在桌上、纸箱在货架上、球轻贴屏障”这类有意接触。
## V12：从“摆模板”到“按组件树重建”

- **可见物体先做树状分解**：`blender_lib/bs_structure.py` 给出 barrier / chair / bench / desk / table / monitor / sofa / shelf / cabinet / window 的 part-count-parent-connect 契约；`SCENE_PLAN` 里保留 `component_tree`，`validate_tree()` 会在写 builder 前报出缺失/断开的部件。模板仍可参考，但连接关系必须显式。
- **单个物体修正**：屏障每端改成“黑色橡胶底 + 红色顶板”的两只脚，而不是一黑一红；长凳加两根通长横梁；办公椅气杆贯通到座面并加座板，扶手加立柱；这些都能在 `previews/objects_sheet.png` 复核。
- **窗不是一块白玻璃**：plan 的 window 有 `mullions={cols,rows}` 和 `blind=none|top_half|full`，`bs_floorplan` 真的生成格子条和百叶；视频里“上半截百叶挡住”的场景可以直接建模。
- **材质有视频证据**：`bs_video_mat.sample_patch(frame,bbox,hint)` 从原视频裁块、按色相族取色并估计 albedo；`material_evidence.json` 记录 frame/bbox/dominant/albedo，`apply_materials.py` 把采样色扩到同类实例，不把 “blue/orange” 写成硬编码 RGB。
- **相机不再钻进物体**：`bs_trajvideo` 的 path 对 barrier/sofa/beanbag/ball 等做 AABB 退出，并向房间中心退；末段对 `camera_focus="end"` 的前景物（绿球）保持 1.4 m 站距并俯视，避免球掉出画面或相机埋进沙发/屏障。
- **轨迹方位估计**：`bs_trajorient.py` 把每件物品投到最近 path sample，输出 phase_t / side / bearing / distance，用来放前景物，而不是随意填坐标。
## V11 的关键变更（上一轮真实重建）

- **房间尺度由“场景理解”决定，不由轨迹长度决定**：v11 真实 plan 把 L 形户型扩到 10×8 m，含 5 个窗、1 个门、9 段蓝色屏障；轨迹只提供相机位置/时序。
- **规划物品与背景补充**：17 个 id / 26 个实例，新增第二工位、货架纸箱、柜体；每个 id 仍经过 `inspect_assets()` 单件检查，不能凭空乱加。
- **屏障布面-金属架连接**：`build_barrier` 的布面从 82%×70% 改为 96%×88%，上缘贴顶管，两端有金属夹片，下摆有 hem bar，红/黑配重脚与立柱相连；单件近景可查 `previews/barrier_close.png`。
- **地面是 plan 数据，不是硬编码**：`architecture.floor = {style, colors, tile_m}` → `bs_mat.floor_pattern()`；`foam_interlock` 支持黄蓝交错泡沫地，深灰橡胶/木/瓷砖也是同一个入口。真实测试见 `previews/yellowblue_floor.png`。
- **轨迹到 Z-up 的修复**：`bs_trajcoord` 做 R 旋转 + footprint 归一化 + 可选 yaw 轴符号 calib；`bs_trajvideo` 的内容感知 look-at 现在把 `place`（单实例）也算进目标，不再只盯屏障；实测同轨迹首段能看到办公区、中段木桌/长凳、末段屏障+绿球。
- **通用 2D 布局**：`t_scaffold_assets` 的碰撞分离从“只推 x”改为按最小穿透轴推 x/y，避免屏障/桌子/柜体在同一墙边硬穿。
- **模板是参考，不是终点**：`bs_assets` 提供构件语法；`build_assets.py` 允许按视频改写每个 builder；V11 的屏障 builder 已按视频结构改写，布面/管架/配重脚在一个 root + 一个 joined mesh 内。

## 四个阶段（运行时状态机，模型不能跳）

| 阶段 | 做什么 | 通过条件 |
|---|---|---|
| `S1_recon` | 整段视频抽 12 帧，先写 `architecture`：`room_bbox_m`、`ceiling_z`、`windows[]`、`doors[]`；再逐 id 清点主要前景 + 合理脑补背景（`INF_`） | `room_bbox_m` 必填；`windows`/`doors` 键必须存在；每个 id 有 `id/count/size_m/seen_in`；位置用 `place/cell/instances` 或 `relation`；多件 `instances` 长度=count；`relation.target` 必须存在；`inferred` 前缀 `INF_` |
| `S2_shell` | `init_scene()` + `build_shell()`：地板 z=0、墙、真实门窗洞、窗框/玻璃/百叶、门叶+把手、顶、默认灯、7 相机 | `scene.blend`、`shell_built`、≥3 相机、至少一张非平 `shell_*.png`、`state["shell"]` |
| `S3_assets` | `scaffold_assets()` 生成 `SPECS` + `bs_assets.build_instance`；`build_assets()` 建所有 id，逐件临时相机渲染后删相机 | `build_assets.py`、`state.assets.ok`（coverage 无 missing/short/extra、无散件、单件检查 ok、`placement.overlaps` 为空）、`objects_sheet.png` |
| `S4_qa` | `qa_scene()` 全场景对账；按 `errors` 回阶段修；够了 `finish` | QA hard checks：`floor_zero/solid_envelope/openings/camera_fov/duplicate_objects/plan_coverage/camera_rig`；至少一张非平预览 |

## 场景理解：不是“整段视频一次性丢给模型”

- `video_frames(count=12)` 在本机用 ffmpeg 沿整条时间轴等距抽帧；文件名是秒数。
- 每轮最多 4 张图：S1 带 4 帧，S2/S3/S4 带关键帧 + 当前渲染/对比/逐件检查表。
- S1 提示词明确要求“从头扫到尾”：绿色瑜伽球、橙色沙发、只出现半程的物体都要清点；
  同款屏障/椅子要写 `count` 和逐个 `instances`。
- 轨迹只给**相机姿态参考**，不参与户型尺寸；户型由视频中的门宽/桌高/人高和整段空间关系估。
- 背景物体：架子上的纸箱/工具、桌下主机、墙上空调、角落纸箱、桌上键鼠/台灯都要脑补；
  看不清也给合理尺寸和位置，不能只建 7 件前景。

## 每个物品怎么保证不是散件

`t_scaffold_assets()` 生成的 `build_assets.py` 只做两件事：

```python
import bs_assets
for spec in SPECS:
    bs_assets.build_instance(spec)
```

每个 `build_instance`：

1. 建一个 Empty root（位置/朝向/`z0`）；
2. 按 `kind` 调 `build_<kind>`，所有部件 `parent_all(root)`；
3. `join_children(root)` 合成一个 mesh；
4. 多实例用 `bs_geo.clone_asset`，不重新建一遍（v4.4 的 `.001` 重叠对就是重跑造成的）。

`build_assets()` 保存后对每个 id：

1. 创建临时相机取景；2) Workbench 320px/4 samples 渲染；3) 立刻删除相机；
4) 算世界 bbox、`meshes==1`、是否 joined；5) 拼 `previews/objects_sheet.png`。
只改一件用 `inspect_object(id)`，同样临时相机，渲完即删。

## 摆放与尺度

- `size_m = [宽, 深, 高]` 是 S1 必填；没有尺寸就无法判断“显示器是否在桌上、椅子是否在桌边、
  架/柜是否靠墙”，也无法避免同格重叠。
- `scaffold_assets` 解析 `on/near/against`：`on` 放到目标顶面（`z = target.z + target.h`），
  `near` 保持地面并偏移，`against` 贴墙并旋转 90°（E/W 墙）。
- 同层 AABB 互穿检查会输出 `PLACEMENT_JSON`，例如 `{"ok": false, "overlaps":[...]}`；
  S3 门禁把 `O7_OfficeDesk×O9_Monitor` 这类重叠直接拦下，模型必须改 relation/placements。
- `build_assets` 里的 `placement.overlaps` 是**真实世界 bbox**，不是只看 plan 坐标。

## 本机实跑（v9 e2e，2 核 CPU）

输入：v8 实跑 `SCENE_PLAN` 的 11 个 id / 16 实例，改成 v9 的 architecture + size + relation
（13 id / 18 实例，含 6 块蓝屏障、绿球、橙沙发、豆袋、机器人、木凳、桌/椅/显示器、铁架、
蒙布物、2 个 `INF_Box`）。

- `S2_shell`：43 网格、7 相机、`shell_overview.png` 非平；1 窗（W，带百叶）/ 1 门（E，带把手）
- `S3_assets`：49 网格、13 id 全部 `meshes==1`、coverage `missing=[] short=[] extra=[]`、
  `placement.overlaps=[]`、`objects_sheet.png`
- `S4_qa`：`floor_zero / solid_envelope / openings / duplicate_objects / plan_coverage / camera_rig`
  全过；仅 `camera_clearance` 一条 info；`material_texture_wiring` 在语义材质模式下自动跳过
- 测试：`33 passed`（含 v9 的 architecture/size/relation 门禁与 `scaffold_assets` 模板测试）

> 上面的 e2e 是**本地夹具**，不是 Kimi 自动清点结果；真实 Kimi 仍需按 S1 prompt 跑一遍。

## 快速开始

```bash
cd indoor_twin_agent
python run.py --video /path/to/walkthrough.mp4 --project ./project
# 需要：OPENAI_BASE_URL / OPENAI_API_KEY / MODEL
# 本机：--blender /path/to/blender（4.5.1 LTS）、ffmpeg、numpy、Pillow
```

`run.py --resume ./project` 从 `state.json` 继续；`--max-steps` 控制预算。

## 目录

```
indoor_twin_agent/
  agent.py          # 阶段机 + 门禁 + 少量 patch 协议
  prompts.py        # S1-S4 提示词（v9 已写入 architecture/size/relation/材质/摆放要求）
  tools.py          # 工具层：video_frames/init_scene/build_shell/scaffold_assets/build_assets/qa_scene...
  executor.py       # Blender / ffmpeg 本机执行器
blender_lib/
  bs_layout.py      # 户型/门窗/百叶/默认灯/相机（plan architecture 为准）
  bs_assets.py      # 30 类语义模板 + relation/size 摆放
  bs_geo.py         # 点线面基元：pillow/soft_sphere/wall/clone_asset
  bs_mat.py         # 语义材质 preset / 橡胶地面 / 玻璃 / 发光
  bs_stage.py       # init/finalize/coverage/placement_report/inspect_assets
  bs_qa.py          # 全场景 QA（门窗/地板/外壳/相机/遮挡/覆盖）
tests/              # 33 个 host 测试（不启动 Blender）
```

## 已知边界

- 安全门/窗在 v9 里是通用 `door/window` builder；S1 prompt 让门窗主要放 `architecture`，
  也可作为 object 补细节，但不要两处都建同一扇门（否则 QA 会数重）。
- `camera_fov` 只有提供内参时才严格比较；没有内参时跳过并记 info。
- 模板能表达“变形靠枕”，但不是交互式雕刻器；更复杂的拓扑用 `patch_file` 改
  `bs_assets.build_<kind>` 的顶点/面，或加新的 builder。
- 默认灯只是“能看清”的起点，亮度/色温后续阶段再调。
