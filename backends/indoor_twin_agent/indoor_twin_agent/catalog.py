"""Generic indoor-asset catalogue.

The acceptance criterion is not "this one office video works": every category here has a
default size, material family, placement zone, builder kind and aliases.  New videos can
produce categories that were never in the V15 office scene (bed, TV, TV cabinet, fish tank...)
without editing the scene plan by hand.
"""
from __future__ import annotations

CATEGORIES = {
    "sofa": dict(size=[2.10, .90, .80], material="fabric", color="orange", zone="living", yaw=270,
                group="seating", kind="sofa", aliases=["sofa", "couch", "沙发"]),
    "sectional_sofa": dict(size=[2.80, 1.60, .80], material="fabric", zone="living", yaw=270,
                group="seating", kind="sofa", aliases=["sectional", "l-shaped sofa", "转角沙发"]),
    "armchair": dict(size=[.85, .85, .85], material="fabric", zone="living", yaw=270,
                group="seating", kind="armchair", aliases=["armchair", "扶手椅"]),
    "coffee_table": dict(size=[1.10, .60, .42], material="wood", zone="living", yaw=0,
                group="seating", kind="table", aliases=["coffee table", "茶几"]),
    "dining_table": dict(size=[1.40, .80, .75], material="wood", color="brown", zone="dining", yaw=0,
                group="dining", kind="table", aliases=["dining table", "餐桌", "table", "桌子"]),
    "desk": dict(size=[1.50, .75, .75], material="paint", color="white", zone="work", yaw=0,
                group="work", kind="desk", aliases=["desk", "办公桌", "书桌"]),
    "monitor": dict(size=[.55, .08, .40], material="screen", color="black", zone="work", yaw=0,
                group="work", kind="monitor", aliases=["monitor", "display", "screen", "显示器"]),
    "tv": dict(size=[1.10, .08, .65], material="screen", zone="living", yaw=0,
                group="media", kind="tv", aliases=["tv", "television", "电视"]),
    "tv_cabinet": dict(size=[1.40, .40, .50], material="wood", zone="living", yaw=0,
                group="media", kind="tv_cabinet", aliases=["tv cabinet", "tv stand", "电视柜"]),
    "bed": dict(size=[1.60, 2.00, .60], material="fabric", zone="bedroom", yaw=0,
                group="sleep", kind="bed", aliases=["bed", "床"]),
    "wardrobe": dict(size=[1.20, .60, 2.00], material="wood", zone="bedroom", yaw=0,
                group="sleep", kind="wardrobe", aliases=["wardrobe", "closet", "衣柜"]),
    "fish_tank": dict(size=[1.20, .40, .60], material="glass", zone="living", yaw=0,
                group="media", kind="fish_tank", aliases=["fish tank", "aquarium", "鱼缸"]),
    "shelf": dict(size=[1.20, .40, 1.80], material="metal", color="black", zone="storage", yaw=0,
                group="storage", kind="shelf", aliases=["shelf", "rack", "货架", "置物架"]),
    "cabinet": dict(size=[.90, .45, .75], material="wood", zone="storage", yaw=0,
                group="storage", kind="cabinet", aliases=["cabinet", "drawer", "柜子", "抽屉柜"]),
    "chair": dict(size=[.55, .55, .90], material="fabric", color="black", zone="work", yaw=180,
                group="work", kind="chair", aliases=["chair", "office chair", "椅子", "办公椅"]),
    "bench": dict(size=[1.20, .40, .45], material="wood", zone="living", yaw=0,
                group="seating", kind="bench", aliases=["bench", "长凳"]),
    "beanbag": dict(size=[.75, .75, .55], material="fabric", color="gray", zone="living", yaw=0,
                group="seating", kind="beanbag", aliases=["bean bag", "beanbag", "豆袋"]),
    "ball": dict(size=[.65, .65, .65], material="rubber", color="green", zone="living", yaw=0,
                group="seating", kind="ball", aliases=["yoga ball", "exercise ball", "球"]),
    "barrier": dict(size=[1.20, .08, 1.30], material="fabric", color="blue", zone="perimeter", yaw=0,
                group="perimeter", kind="barrier", aliases=["barrier", "partition", "屏障", "围栏"]),
    "robot": dict(size=[.60, .45, .55], material="plastic", zone="work", yaw=0,
                group="work", kind="robot", aliases=["robot", "rover", "机器人"]),
    "person": dict(size=[.35, .25, 1.65], material="paint", zone="work", yaw=0,
                group="work", kind="person", aliases=["person", "mannequin", "人"]),
    "plant": dict(size=[.50, .50, 1.20], material="fabric", zone="living", yaw=0,
                group="decor", kind="plant", aliases=["plant", "绿植", "植物"]),
    "rug": dict(size=[2.00, 3.00, .02], material="fabric", zone="living", yaw=0,
                group="decor", kind="rug", aliases=["rug", "carpet", "地毯"]),
    "fridge": dict(size=[.70, .70, 1.80], material="paint", zone="kitchen", yaw=0,
                group="kitchen", kind="fridge", aliases=["fridge", "refrigerator", "冰箱"]),
    "washer": dict(size=[.60, .60, .85], material="paint", zone="utility", yaw=0,
                group="utility", kind="washer", aliases=["washer", "washing machine", "洗衣机"]),
    "box": dict(size=[.45, .35, .30], material="cardboard", color="brown", zone="storage", yaw=0,
                group="storage", kind="box", aliases=["box", "carton", "纸箱"]),
    "keyboard": dict(size=[.44, .15, .025], material="plastic", zone="work", yaw=0,
                group="work", kind="keyboard", aliases=["keyboard", "键盘"]),
    "mouse": dict(size=[.10, .06, .035], material="plastic", zone="work", yaw=0,
                group="work", kind="mouse", aliases=["mouse", "鼠标"]),
    "pc": dict(size=[.18, .45, .50], material="plastic", color="black", zone="work", yaw=0,
                group="work", kind="pc", aliases=["pc", "computer tower", "主机", "电脑主机"]),
    "lamp": dict(size=[.35, .35, .70], material="paint", color="white", zone="work", yaw=0,
                group="decor", kind="lamp", aliases=["lamp", "台灯", "落地灯"]),
    "ac": dict(size=[.85, .25, .30], material="paint", color="white", zone="living", yaw=0,
                group="decor", kind="ac", aliases=["air conditioner", "空调"]),
    "curtain": dict(size=[1.40, .08, 1.20], material="fabric", color="beige", zone="living", yaw=0,
                group="decor", kind="curtain", aliases=["curtain", "窗帘"]),
    "trash": dict(size=[.35, .35, .55], material="plastic", zone="work", yaw=0,
                group="work", kind="trash", aliases=["trash", "bin", "垃圾桶"]),
}


def classify(name: str, kind: str = "") -> str | None:
    name = str(name or "")
    # The generic English "table" alias otherwise wins over "办公桌" because it is longer
    # in the concatenated string.  The work desk is not a dining table.
    if any(k in name for k in ("办公桌", "书桌", "电脑桌", "工作台")):
        return "desk"
    if any(k in name for k in ("长凳", "长条凳", "长桌")):
        return "bench"
    s = f"{name} {kind}".lower()
    best = None
    for key, c in CATEGORIES.items():
        for alias in c["aliases"]:
            if alias.lower() in s:
                if best is None or len(alias) > len(best[0]):
                    best = (alias, key)
    return best[1] if best else None


def parse_color(value) -> str:
    if isinstance(value, str):
        v = value.lower()
        for c in ("navy", "blue", "orange", "green", "brown", "gray", "grey", "black", "white", "red", "yellow", "beige"):
            if c in v or (c == "blue" and "蓝" in v) or (c == "orange" and "橙" in v) or (c == "green" and "绿" in v):
                return c
        if "#" in value and len(value.strip()) == 7:
            return value.strip()
    return ""


def spec(cat: str, count: int = 1, evidence: dict | None = None) -> dict:
    c = CATEGORIES[cat]
    ev = evidence or {}
    size = ev.get("size_m") or c["size"]
    return {
        "kind": c["kind"], "category": cat, "size_m": list(size), "count": int(count),
        "material": ev.get("material") or c["material"],
        "color": parse_color(ev.get("color")) or c.get("color") or "",
        "zone": c["zone"], "group": c["group"], "yaw": c["yaw"],
        "source": ev.get("source") or "observed",
    }
