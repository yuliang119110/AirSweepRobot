import sys
import os
import time
import argparse
import traceback

import carb
from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument(
    "--usd-path",
    default="/model_test/so_arm101_usd/so_arm101/so_arm101.usda",
    help="Robot USDA file path",
)
parser.add_argument("--headless", action="store_true")
args_cli, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args_cli.headless})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
import omni.usd
import omni.kit.commands

TABLE_HEIGHT = 0.72
ROBOT_BASE_Z = TABLE_HEIGHT


def create_material(stage, path, color):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def add_box(stage, path, pos, size, mat, collision=True, kinematic=True):
    xform = UsdGeom.Xform.Define(stage, path)
    cube = UsdGeom.Cube.Define(stage, path + "/geometry")
    cube.CreateSizeAttr().Set(1.0)
    xform.AddScaleOp().Set(Gf.Vec3d(*size))
    xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        rb = UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        if kinematic:
            rb.CreateKinematicEnabledAttr().Set(True)
    return xform


def setup_robot(stage, usd_path):
    robot_xform = UsdGeom.Xform.Define(stage, "/World/Robot")
    robot_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, ROBOT_BASE_Z))
    robot_prim = robot_xform.GetPrim()
    if not os.path.isfile(usd_path):
        raise FileNotFoundError(f"Robot USDA not found: {usd_path}")
    robot_prim.GetReferences().AddReference(usd_path)
    print(f"[clean_scene] Robot USDA referenced: {usd_path}", flush=True)
    simulation_app.update()
    set_count = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            prim.CreateAttribute("drive:angular:physics:stiffness", Sdf.ValueTypeNames.Float).Set(1e5)
            prim.CreateAttribute("drive:angular:physics:damping", Sdf.ValueTypeNames.Float).Set(1e3)
            prim.CreateAttribute("drive:angular:physics:maxForce", Sdf.ValueTypeNames.Float).Set(100.0)
            set_count += 1
    print(f"[clean_scene] Joint drives set on {set_count} revolute joints", flush=True)


def setup_camera(stage):
    cam = UsdGeom.Camera.Define(stage, "/World/ViewCamera")
    cam.AddTranslateOp().Set(Gf.Vec3d(1.2, 1.0, 1.3))
    cam.AddRotateXYZOp().Set(Gf.Vec3f(-35, 0, 50))
    try:
        import omni.kit.viewport.utility as vp_util
        vp = vp_util.get_active_viewport()
        if vp:
            vp.set_active_camera("/World/ViewCamera")
            print("[clean_scene] Viewport camera set", flush=True)
    except Exception as e:
        print(f"[clean_scene] Auto camera setup skipped: {e}", flush=True)


def setup_scene():
    stage = omni.usd.get_context().get_stage()
    print("[clean_scene] === Building scene ===", flush=True)
    ps = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    ps.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    ps.CreateGravityMagnitudeAttr().Set(9.81)
    ground = UsdGeom.Plane.Define(stage, "/World/Ground")
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    m_ground = create_material(stage, "/World/Mat/Ground", Gf.Vec3f(0.22, 0.24, 0.27))
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(m_ground)
    m_table = create_material(stage, "/World/Mat/Table", Gf.Vec3f(0.50, 0.42, 0.32))
    m_sign = create_material(stage, "/World/Mat/Sign", Gf.Vec3f(0.06, 0.50, 0.75))
    m_dirt = create_material(stage, "/World/Mat/Dirt", Gf.Vec3f(0.55, 0.20, 0.08))
    add_box(stage, "/World/Table",
            pos=(0.22, 0.0, TABLE_HEIGHT / 2),
            size=(0.80, 0.55, TABLE_HEIGHT),
            mat=m_table)
    print("[clean_scene] Table done", flush=True)
    add_box(stage, "/World/SignBoard",
            pos=(0.40, 0.0, TABLE_HEIGHT + 0.10),
            size=(0.015, 0.25, 0.20),
            mat=m_sign)
    print("[clean_scene] Sign board done", flush=True)
    for i, (dx, dy) in enumerate([(0.12, -0.06), (0.22, 0.04), (0.32, -0.02)]):
        add_box(stage, f"/World/Dirt_{i}",
                pos=(dx, dy, TABLE_HEIGHT + 0.002),
                size=(0.05, 0.05, 0.003),
                mat=m_dirt, collision=False)
    print("[clean_scene] Dirt spots done", flush=True)
    UsdLux.DistantLight.Define(stage, "/World/DistantLight").CreateIntensityAttr().Set(800)
    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr().Set(400)
    setup_robot(stage, args_cli.usd_path)
    setup_camera(stage)
    print("[clean_scene] === Scene ready ===", flush=True)


try:
    setup_scene()
    for _ in range(3):
        simulation_app.update()
    omni.kit.commands.execute("Play")
    print("[clean_scene] === Physics playing ===", flush=True)
    frame = 0
    t0 = time.time()
    target_dt = 1.0 / 30.0
    while simulation_app.is_running():
        loop_start = time.time()
        simulation_app.update()
        frame += 1
        if frame % 300 == 0:
            elapsed = time.time() - t0
            fps = frame / elapsed if elapsed > 0 else 0
            print(f"[clean_scene] frame {frame}  ({fps:.1f} fps)", flush=True)
        spent = time.time() - loop_start
        if spent < target_dt:
            time.sleep(target_dt - spent)
except Exception as e:
    print(f"[clean_scene] FATAL: {e}", file=sys.stderr, flush=True)
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
    print("[clean_scene] === Closed ===", flush=True)
