#!/usr/bin/env python3
"""导入 Blender 导出资产到 Isaac Sim。"""
import os, sys, shutil, pathlib, asyncio

WORK_DIR  = pathlib.Path("/home/dgx/air/isaac_sim_test/blander")
OBJ_DIRS  = {"grate": WORK_DIR / "world/栅格", "sofa": WORK_DIR / "world/沙发"}
USDZ_DIRS = {"grate_01": WORK_DIR / "栅格资产", "sofa_pbr": WORK_DIR / "沙发资产", "sign_01": WORK_DIR / "立牌usd资产"}
OUTPUT    = WORK_DIR / "isaac_sim_input"
OUTPUT.mkdir(parents=True, exist_ok=True)
stage_path = str(OUTPUT / "composed_scene.usd")
print(f"[INFO] Output: {OUTPUT}")

print("[INFO] Starting Isaac Sim...")
from isaacsim import SimulationApp
if not os.environ.get("DISPLAY"): os.environ["DISPLAY"] = ":0"
app = SimulationApp({"headless": True, "hide_ui": True})
print("[INFO] SimulationApp started.")

import omni.usd
from pxr import Usd, UsdGeom, Sdf, UsdShade, Gf, UsdLux
from omni.kit.asset_converter import AssetConverterContext
from omni.kit.asset_converter.impl.task_manager import AssetConverterTaskManager as ACTM

ctx = omni.usd.get_context()

# ── OBJ → USD 转换 ────────────────────────────────────────────
print("\n[INFO] ── OBJ → USD 转换 ────────────────────────")
async def convert():
    for name, src in OBJ_DIRS.items():
        if not src.exists(): print(f"[WARN] {src}"); continue
        out_dir = OUTPUT / name; out_dir.mkdir(exist_ok=True)
        for f in sorted(src.iterdir()):
            if f.is_file(): shutil.copy2(f, out_dir / f.name)
        obj = out_dir / "base.obj"
        if not obj.exists(): print(f"[WARN] {obj}"); continue
        u = str(OUTPUT / f"{name}.usd")
        print(f"  [CONVERT] {name} ...")
        c = AssetConverterContext(); c.ignore_materials = False; c.merge_all_meshes = False
        t = ACTM.create_converter_task(str(obj), u, None, c, None, True)
        ok = await t.wait_until_finished()
        print(f"  [{'OK' if ok else 'FAIL'}] {name}.usd")

loop = asyncio.new_event_loop(); loop.run_until_complete(convert())

# ── 创建文件绑定的 stage ──────────────────────────────────────
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
UsdGeom.SetStageMetersPerUnit(stage, 0.01)

g = UsdGeom.Mesh.Define(stage, "/World/ground")
g.CreatePointsAttr([(-500,0,-500),(500,0,-500),(500,0,500),(-500,0,500)])
g.CreateFaceVertexCountsAttr([4]); g.CreateFaceVertexIndicesAttr([0,1,2,3])
g.CreateExtentAttr([(-500,-0.1,-500),(500,0.1,500)])
stage.GetRootLayer().Export(stage_path)
ctx.open_stage(stage_path)
stage = ctx.get_stage()
print(f"[INFO] Stage opened from: {stage_path}")

# ── 重建环境 ──────────────────────────────────────────────────
g = UsdGeom.Mesh.Define(stage, "/World/ground")
g.CreatePointsAttr([(-500,0,-500),(500,0,-500),(500,0,500),(-500,0,500)])
g.CreateFaceVertexCountsAttr([4]); g.CreateFaceVertexIndicesAttr([0,1,2,3])
g.CreateExtentAttr([(-500,-0.1,-500),(500,0.1,500)])
mat = UsdShade.Material.Define(stage, "/World/ground/Mat")
sh = UsdShade.Shader.Define(stage, "/World/ground/Mat/Shader")
sh.CreateIdAttr("UsdPreviewSurface")
sh.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.2,0.2,0.2))
sh.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(0.8)
UsdShade.MaterialBindingAPI(g).Bind(mat)
d = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
d.CreateIntensityAttr(500)
print("[OK] Ground + lights")

# ── 引用转换好的 USD ─────────────────────────────────────────
print("\n[INFO] ── 引用 USD ─────────────────────────────")
for name in OBJ_DIRS:
    usd = OUTPUT / f"{name}.usd"
    if usd.exists():
        p = stage.DefinePrim(f"/World/{name}", "Xform")
        p.GetReferences().AddReference(f"./{name}.usd")
        # 清除默认的 xform op，重新设置
        x = UsdGeom.Xformable(p)
        x.ClearXformOpOrder()
        x.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
        print(f"  [REF] {name}.usd -> /World/{name}")

# ── 引用 USDZ ────────────────────────────────────────────────
print("\n[INFO] ── 引用 USDZ ────────────────────────────")
for eng_name, src in USDZ_DIRS.items():
    if not src.exists(): print(f"[WARN] {src}"); continue
    for f in sorted(src.glob("*.usdz")):
        dst = OUTPUT / f.name
        if not dst.exists(): shutil.copy2(f, dst)
        safe = f.name.replace(" ", "%20")
        p = stage.DefinePrim(f"/World/usdz_{eng_name}", "Xform")
        p.GetReferences().AddReference(f"./{safe}")
        print(f"  [OK] {f.name} -> /World/usdz_{eng_name}")

# ── 保存 ──────────────────────────────────────────────────────
print(f"\n[INFO] ── 保存 ──────────────────────────────────")
ctx.save_stage()
print(f"  [OK] saved: {stage_path}")

# ── 清单 ──────────────────────────────────────────────────────
print(f"\n[INFO] ── {OUTPUT} 内容 ─────────────────────────")
for f in sorted(OUTPUT.iterdir()):
    if f.is_dir(): continue
    print(f"  {f.name} ({f.stat().st_size/1024:.1f} KB)")

print("\n[INFO] Closing...")
app.close()
print("[DONE] 导入完成！")
