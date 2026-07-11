# clean_scene.py - SO-ARM101 cleaning scene visualization for Isaac Sim 6.0.1
# Standalone SimulationApp: loads robot USD, builds scene, animates joints.
# Uses default viewport camera, servo-scale drive gains, runs at 20 fps.

import sys
import os
import time
import math
import argparse
import traceback

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument(
    "--usd-path",
    default="/model_test/so_arm101_usd/so_arm101/so_arm101.usda",
)
parser.add_argument("--headless", action="store_true")
args_cli, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args_cli.headless})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
import omni.kit.commands
import carb

# Lower CPU: cap threads, reduce render quality for this simple scene
settings = carb.settings.get_settings()
settings.set_int("/plugins/carb.tasking.plugin/threadCount", 4)
settings.set_int("/plugins/omni.tbb.globalcontrol/maxThreadCount", 4)
settings.set_int("/rtx/resolution/width", 960)
settings.set_int("/rtx/resolution/height", 540)

TABLE_HEIGHT = 0.72


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


def find_joint_prims(stage):
    joints = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joints.append((str(prim.GetPath()), prim))
    return joints


def setup_robot(stage, usd_path):
    robot_xform = UsdGeom.Xform.Define(stage, "/World/Robot")
    # 5 mm above table to avoid initial penetration
    robot_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, TABLE_HEIGHT + 0.005))
    robot_prim = robot_xform.GetPrim()
    if not os.path.isfile(usd_path):
        raise FileNotFoundError(f"Robot USDA not found: {usd_path}")
    robot_prim.GetReferences().AddReference(usd_path)
    print(f"[clean_scene] Robot USDA referenced: {usd_path}", flush=True)

    import omni.usd
    omni.usd.get_context().get_stage().Save()

    # USD ships maxForce=10 N, too low for visible motion.
    # Use servo-scale PD gains.
    joints = find_joint_prims(stage)
    for jpath, jprim in joints:
        jprim.CreateAttribute("drive:angular:physics:type", Sdf.ValueTypeNames.Token).Set("force")
        jprim.CreateAttribute("drive:angular:physics:stiffness", Sdf.ValueTypeNames.Float).Set(1500.0)
        jprim.CreateAttribute("drive:angular:physics:damping", Sdf.ValueTypeNames.Float).Set(80.0)
        jprim.CreateAttribute("drive:angular:physics:maxForce", Sdf.ValueTypeNames.Float).Set(100.0)
        jprim.CreateAttribute("drive:angular:physics:targetPosition", Sdf.ValueTypeNames.Float).Set(0.0)
        jprim.CreateAttribute("drive:angular:physics:targetVelocity", Sdf.ValueTypeNames.Float).Set(0.0)
        print(f"[clean_scene]   Joint: {jpath}", flush=True)
    print(f"[clean_scene] Joint drives configured on {len(joints)} joints", flush=True)
    return joints


def setup_scene():
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    print("[clean_scene] === Building scene ===", flush=True)

    # Physics
    ps = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    ps.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    ps.CreateGravityMagnitudeAttr().Set(9.81)

    # Ground
    ground = UsdGeom.Plane.Define(stage, "/World/Ground")
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    m_ground = create_material(stage, "/World/Mat/Ground", Gf.Vec3f(0.22, 0.24, 0.27))
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(m_ground)

    # Table
    m_table = create_material(stage, "/World/Mat/Table", Gf.Vec3f(0.50, 0.42, 0.32))
    add_box(stage, "/World/Table",
            pos=(0.15, 0.0, TABLE_HEIGHT / 2),
            size=(0.80, 0.55, TABLE_HEIGHT),
            mat=m_table)
    print("[clean_scene] Table done", flush=True)

    # Sign board
    m_sign = create_material(stage, "/World/Mat/Sign", Gf.Vec3f(0.06, 0.50, 0.75))
    add_box(stage, "/World/SignBoard",
            pos=(0.32, 0.0, TABLE_HEIGHT + 0.10),
            size=(0.015, 0.25, 0.20),
            mat=m_sign)

    # Dirt spots
    m_dirt = create_material(stage, "/World/Mat/Dirt", Gf.Vec3f(0.55, 0.20, 0.08))
    for i, (dx, dy) in enumerate([(0.10, -0.06), (0.18, 0.04), (0.25, -0.02)]):
        add_box(stage, f"/World/Dirt_{i}",
                pos=(dx, dy, TABLE_HEIGHT + 0.002),
                size=(0.05, 0.05, 0.003),
                mat=m_dirt, collision=False)

    # Lights
    UsdLux.DistantLight.Define(stage, "/World/DistantLight").CreateIntensityAttr().Set(600)
    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr().Set(400)

    # Robot
    joints = setup_robot(stage, args_cli.usd_path)
    print("[clean_scene] === Scene ready ===", flush=True)
    return joints


def animate_joints(joints, t):
    for idx, (jpath, jprim) in enumerate(joints):
        amp = 0.35
        freq = 0.4 + idx * 0.12
        angle = amp * math.sin(2.0 * math.pi * freq * t)
        target_attr = jprim.GetAttribute("drive:angular:physics:targetPosition")
        if target_attr:
            target_attr.Set(angle)


try:
    joints = setup_scene()
    for _ in range(5):
        simulation_app.update()

    try:
        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
    except Exception:
        omni.kit.commands.execute("Play")
    print("[clean_scene] === Physics playing ===", flush=True)

    frame = 0
    t0 = time.time()
    target_dt = 1.0 / 20.0

    while simulation_app.is_running():
        loop_start = time.time()
        animate_joints(joints, time.time() - t0)
        simulation_app.update()
        frame += 1
        if frame % 150 == 0:
            elapsed = time.time() - t0
            fps = frame / elapsed if elapsed > 0 else 0
            tgts = []
            for _, jp in joints[:3]:
                ta = jp.GetAttribute("drive:angular:physics:targetPosition")
                if ta:
                    tgts.append(f"{ta.Get():.3f}")
            print(f"[clean_scene] frame {frame}  ({fps:.1f} fps)  targets=[{', '.join(tgts)}]", flush=True)
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
