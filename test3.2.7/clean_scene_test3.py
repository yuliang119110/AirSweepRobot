# clean_scene_test3.py - SO-ARM101 + drone_with_arm 清洁仿真场景
# 将 drone_with_arm URDF 和 base_basic_pbr(2).usdz 导入场景
# 使用 test3 conda 环境 + Docker 运行

import sys
import os
import time
import math
import argparse
import traceback

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
args_cli, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args_cli.headless})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
import omni.kit.commands
import omni.usd
import carb
import omni.timeline

# CPU / render 限制
settings = carb.settings.get_settings()
settings.set_int("/plugins/carb.tasking.plugin/threadCount", 4)
settings.set_int("/plugins/omni.tbb.globalcontrol/maxThreadCount", 4)
settings.set_int("/rtx/resolution/width", 960)
settings.set_int("/rtx/resolution/height", 540)

TABLE_HEIGHT = 0.72

# 模型路径（容器内 /workspace -> /home/dgx/air/test3）
DRONE_URDF = "/workspace/blander/drone_with_arm/drone_with_arm.urdf"
BASE_USDZ  = "/workspace/blander/base_basic_pbr(2).usdz"


def log(msg):
    print(f"[test3] {msg}", flush=True)


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


def find_joint_prims(stage, prefix="/World/Robot"):
    joints = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            path = str(prim.GetPath())
            if path.startswith(prefix):
                joints.append((path, prim))
    return joints


def setup_joint_drives(joints, stiffness=1500.0, damping=80.0, max_force=100.0):
    for jpath, jprim in joints:
        jprim.CreateAttribute("drive:angular:physics:type", Sdf.ValueTypeNames.Token).Set("force")
        jprim.CreateAttribute("drive:angular:physics:stiffness", Sdf.ValueTypeNames.Float).Set(stiffness)
        jprim.CreateAttribute("drive:angular:physics:damping", Sdf.ValueTypeNames.Float).Set(damping)
        jprim.CreateAttribute("drive:angular:physics:maxForce", Sdf.ValueTypeNames.Float).Set(max_force)
        jprim.CreateAttribute("drive:angular:physics:targetPosition", Sdf.ValueTypeNames.Float).Set(0.0)
        jprim.CreateAttribute("drive:angular:physics:targetVelocity", Sdf.ValueTypeNames.Float).Set(0.0)
        log(f"  Joint drive: {jpath}")


def import_drone_urdf(stage):
    """导入 drone_with_arm URDF"""
    if not os.path.exists(DRONE_URDF):
        log(f"WARN: URDF 不存在 {DRONE_URDF}")
        return []

    log("启用 URDF 导入扩展...")
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("omni.scene.optimizer.core", True)
    ext_manager.set_extension_enabled_immediate("isaacsim.robot.schema", True)
    ext_manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)

    try:
        from isaacsim.asset.importer.urdf.impl import URDFImporter, URDFImporterConfig
    except ImportError:
        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

    config = URDFImporterConfig()
    config.urdf_path = DRONE_URDF
    config.usd_path = "/tmp/drone_arm_usd"
    config.merge_fixed_joints = False
    config.fix_base = True
    config.collision_from_visuals = True
    config.collision_type = "Convex Hull"
    config.joint_drive_type = "force"
    config.joint_target_type = "position"
    config.override_joint_stiffness = {".*": 1000.0}
    config.override_joint_damping = {".*": 100.0}
    config.convert_rest_state = True

    log(f"导入 URDF: {DRONE_URDF}")
    importer = URDFImporter(config)
    generated_usd = importer.import_urdf()
    log(f"URDF 转换完成: {generated_usd}")

    if not generated_usd or not os.path.exists(generated_usd):
        log("ERROR: URDF 导入失败")
        return []

    # 放置到场景中 - 在桌面上方
    robot_xform = UsdGeom.Xform.Define(stage, "/World/Robot")
    robot_xform.AddTranslateOp().Set(Gf.Vec3d(0.15, 0.0, TABLE_HEIGHT + 0.05))
    robot_prim = robot_xform.GetPrim()
    robot_prim.GetReferences().AddReference(generated_usd)
    log("drone_with_arm 加载到 /World/Robot")

    # 等待几帧让引用解析
    for _ in range(10):
        simulation_app.update()

    # 配置关节驱动
    joints = find_joint_prims(stage, "/World/Robot")
    log(f"找到 {len(joints)} 个关节")
    setup_joint_drives(joints)

    return joints


def import_base_usdz(stage):
    """导入 base_basic_pbr(2).usdz 作为视觉基座"""
    if not os.path.exists(BASE_USDZ):
        log(f"WARN: USDZ 不存在 {BASE_USDZ}")
        return

    log(f"导入 USDZ: {BASE_USDZ}")
    base_prim = stage.DefinePrim("/World/BasePlatform", "Xform")
    base_prim.GetReferences().AddReference(BASE_USDZ)
    base_xform = UsdGeom.Xform(base_prim)
    base_xform.AddTranslateOp().Set(Gf.Vec3d(-0.35, 0.0, TABLE_HEIGHT + 0.002))
    base_xform.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))
    log("base_basic_pbr(2).usdz 加载到 /World/BasePlatform")


def setup_scene():
    stage = omni.usd.get_context().get_stage()
    log("=== 构建场景 ===")

    # 物理场景
    ps = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    ps.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    ps.CreateGravityMagnitudeAttr().Set(9.81)

    # 地面
    ground = UsdGeom.Plane.Define(stage, "/World/Ground")
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    m_ground = create_material(stage, "/World/Mat/Ground", Gf.Vec3f(0.22, 0.24, 0.27))
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(m_ground)

    # 桌子
    m_table = create_material(stage, "/World/Mat/Table", Gf.Vec3f(0.50, 0.42, 0.32))
    add_static_cuboid(stage, "/World/Table",
                      pos=(0.15, 0.0, TABLE_HEIGHT / 2),
                      size=(0.80, 0.55, TABLE_HEIGHT),
                      mat=m_table)
    log("桌子创建完成")

    # 标志牌
    m_sign = create_material(stage, "/World/Mat/Sign", Gf.Vec3f(0.06, 0.50, 0.75))
    add_static_cuboid(stage, "/World/SignBoard",
                      pos=(0.32, 0.0, TABLE_HEIGHT + 0.10),
                      size=(0.015, 0.25, 0.20),
                      mat=m_sign)

    # 污渍标记
    m_dirt = create_material(stage, "/World/Mat/Dirt", Gf.Vec3f(0.55, 0.20, 0.08))
    for i, (dx, dy) in enumerate([(0.10, -0.06), (0.18, 0.04), (0.25, -0.02)]):
        add_static_cuboid(stage, f"/World/Dirt_{i}",
                          pos=(dx, dy, TABLE_HEIGHT + 0.002),
                          size=(0.05, 0.05, 0.003),
                          mat=m_dirt, collision=False)

    # 灯光
    UsdLux.DistantLight.Define(stage, "/World/DistantLight").CreateIntensityAttr().Set(600)
    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr().Set(400)

    # 导入 base_basic_pbr(2).usdz
    import_base_usdz(stage)

    # 导入 drone_with_arm URDF
    joints = import_drone_urdf(stage)

    log("=== 场景构建完成 ===")
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

    # 启动物理
    try:
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
    except Exception:
        omni.kit.commands.execute("Play")
    log("=== 物理仿真启动 ===")

    frame = 0
    t0 = time.time()
    target_dt = 1.0 / 20.0

    while simulation_app.is_running():
        loop_start = time.time()
        if joints:
            animate_joints(joints, time.time() - t0)
        simulation_app.update()
        frame += 1
        if frame % 150 == 0:
            elapsed = time.time() - t0
            fps = frame / elapsed if elapsed > 0 else 0
            log(f"frame {frame}  ({fps:.1f} fps)")
        spent = time.time() - loop_start
        if spent < target_dt:
            time.sleep(target_dt - spent)

except Exception as e:
    log(f"FATAL: {e}")
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
    log("=== 仿真关闭 ===")
