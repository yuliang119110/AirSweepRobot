"""
SO-ARM101 清洁场景可视化脚本（纯 SimulationApp + pxr，不依赖 isaaclab）。

v2: 增加完整错误捕获，简化 URDF 导入。
"""
import sys
import traceback
import argparse
import os

import carb
from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--load-usdz", action="store_true", help="加载 USDZ 视觉资产")
args_cli, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": False})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

import omni.kit.commands
import omni.usd

MODEL_TEST_DIR = "/model_test"
URDF_PATH = os.path.join(MODEL_TEST_DIR, "机械臂", "SO-ARM101", "urdf", "so_arm101.urdf")
USDZ_DIR = os.path.join(MODEL_TEST_DIR, "机械臂")
USDZ_ASSETS = {
    "sign": os.path.join(USDZ_DIR, "黑客送立牌.usdz"),
    "grid": os.path.join(USDZ_DIR, "天上的栅格.usdz"),
}
USDZ_TARGET_SIZES = {
    "sign": (0.015, 0.28, 0.20),
    "grid": (0.30, 0.20, 0.015),
}
TABLE_HEIGHT = 0.72


def create_material(stage, path, color):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def add_static_cuboid(stage, path, pos, size, mat, collision=True):
    xform = UsdGeom.Xform.Define(stage, path)
    cube = UsdGeom.Cube.Define(stage, path + "/geometry")
    cube.CreateSizeAttr().Set(1.0)
    xform.AddScaleOp().Set(Gf.Vec3d(*size))
    xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        rb = UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        rb.CreateKinematicEnabledAttr().Set(True)
    return xform


def import_robot_urdf(stage):
    print("[clean_scene] Enabling URDF importer extension...", flush=True)
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("omni.scene.optimizer.core", True)
    ext_manager.set_extension_enabled_immediate("isaacsim.robot.schema", True)
    ext_manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)

    from isaacsim.asset.importer.urdf.impl import URDFImporter, URDFImporterConfig

    import_config = URDFImporterConfig()
    import_config.urdf_path = URDF_PATH
    import_config.usd_path = "/tmp/so_arm101_usd"
    import_config.merge_fixed_joints = False
    import_config.fix_base = True
    import_config.collision_from_visuals = True
    import_config.collision_type = "Convex Hull"
    import_config.joint_drive_type = "force"
    import_config.joint_target_type = "position"
    import_config.override_joint_stiffness = {".*": 1000.0}
    import_config.override_joint_damping = {".*": 100.0}

    print(f"[clean_scene] Importing URDF: {URDF_PATH}", flush=True)
    importer = URDFImporter(import_config)
    usd_path = importer.import_urdf()
    print(f"[clean_scene] Robot USD generated: {usd_path}", flush=True)

    robot_xform = UsdGeom.Xform.Define(stage, "/World/Robot")
    robot_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, TABLE_HEIGHT))
    robot_prim = robot_xform.GetPrim()
    robot_prim.GetReferences().AddReference(usd_path)
    print("[clean_scene] Robot loaded to stage", flush=True)

    return robot_xform


def setup_scene():
    stage = omni.usd.get_context().get_stage()
    print("[clean_scene] Stage acquired, building scene...", flush=True)

    # Physics
    phys_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    phys_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    phys_scene.CreateGravityMagnitudeAttr().Set(9.81)

    # Ground
    ground = UsdGeom.Plane.Define(stage, "/World/Ground")
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    ground_mat = create_material(stage, "/World/Materials/Ground", Gf.Vec3f(0.25, 0.27, 0.30))
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(ground_mat)

    # Materials
    mat_table = create_material(stage, "/World/Materials/Table", Gf.Vec3f(0.42, 0.42, 0.38))
    mat_sign = create_material(stage, "/World/Materials/Sign", Gf.Vec3f(0.05, 0.45, 0.70))
    mat_grid = create_material(stage, "/World/Materials/Grid", Gf.Vec3f(0.18, 0.54, 0.38))
    mat_dirt = create_material(stage, "/World/Materials/Dirt", Gf.Vec3f(0.58, 0.18, 0.06))

    # Table
    add_static_cuboid(stage, "/World/Table", pos=(0.20, 0.0, 0.36), size=(0.75, 0.55, 0.72), mat=mat_table)
    print("[clean_scene] Table created", flush=True)

    # Sign collision
    add_static_cuboid(stage, "/World/SignCollision", pos=(0.22, 0.0, 0.88), size=(0.015, 0.28, 0.20), mat=mat_sign)
    print("[clean_scene] Sign collision created", flush=True)

    # Grid collision
    add_static_cuboid(stage, "/World/GridCollision", pos=(0.20, 0.0, 0.96), size=(0.30, 0.20, 0.015), mat=mat_grid)
    print("[clean_scene] Grid collision created", flush=True)

    # Dirt markers
    add_static_cuboid(stage, "/World/Dirt0", pos=(0.211, -0.075, 0.91), size=(0.003, 0.030, 0.030), mat=mat_dirt, collision=False)
    add_static_cuboid(stage, "/World/Dirt1", pos=(0.211, 0.000, 0.86), size=(0.003, 0.024, 0.024), mat=mat_dirt, collision=False)
    add_static_cuboid(stage, "/World/Dirt2", pos=(0.211, 0.080, 0.825), size=(0.003, 0.020, 0.020), mat=mat_dirt, collision=False)
    print("[clean_scene] Dirt markers created", flush=True)

    # Lights
    UsdLux.DistantLight.Define(stage, "/World/DistantLight").CreateIntensityAttr().Set(800)
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr().Set(400)

    # Robot
    import_robot_urdf(stage)

    # Optional USDZ
    if args_cli.load_usdz:
        print("[clean_scene] Loading USDZ visual assets...", flush=True)
        for name, key in [("SignVisual", "sign"), ("GridVisual", "grid")]:
            usdz_path = USDZ_ASSETS[key]
            target = USDZ_TARGET_SIZES[key]
            xform = UsdGeom.Xform.Define(stage, f"/World/{name}")
            if key == "sign":
                xform.AddTranslateOp().Set(Gf.Vec3d(0.22, 0.0, 0.88))
            else:
                xform.AddTranslateOp().Set(Gf.Vec3d(0.20, 0.0, 0.96))
            xform.GetPrim().GetReferences().AddReference(usdz_path)
            print(f"[clean_scene] {name} USDZ referenced (unscaled)", flush=True)

    print("[clean_scene] === Scene setup complete ===", flush=True)


try:
    setup_scene()
    omni.kit.commands.execute("Play")
    print("[clean_scene] === Physics playing ===", flush=True)

    frame = 0
    while simulation_app.is_running():
        simulation_app.update()
        frame += 1
        if frame % 500 == 0:
            print(f"[clean_scene] Running frame {frame}", flush=True)

except Exception as e:
    print(f"[clean_scene] FATAL ERROR: {e}", file=sys.stderr, flush=True)
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
    print("[clean_scene] === Simulation closed ===", flush=True)
