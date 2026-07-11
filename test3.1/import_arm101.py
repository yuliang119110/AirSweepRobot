"""
SO-ARM101 + 桌面导入 Isaac Sim 6.0.1
使用 isaacsim.asset.importer.urdf.URDFImporter 导入机械臂
"""
import os
import sys
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from pxr import UsdGeom, UsdPhysics, UsdLux, UsdShade, Sdf, Gf
import omni.usd
import omni.timeline


def log(msg):
    print(f"[import] {msg}", flush=True)


def setup_scene():
    stage = omni.usd.get_context().get_stage()

    log("创建物理场景")
    scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    ground = UsdGeom.Plane.Define(stage, "/World/Ground")
    ground.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.01))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(ground.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(ground.GetPrim())
    mass_api.CreateMassAttr().Set(1e9)
    ground_mat = UsdShade.Material.Define(stage, "/World/GroundMat")
    ground_shader = UsdShade.Shader.Define(stage, "/World/GroundMat/Shader")
    ground_shader.CreateIdAttr("UsdPreviewSurface")
    ground_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.2, 0.22, 0.25))
    ground_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    ground_mat.CreateSurfaceOutput().ConnectToSource(ground_shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(ground_mat)

    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr().Set(500)
    UsdLux.DistantLight.Define(stage, "/World/DistantLight").CreateIntensityAttr().Set(800)

    table_usdz = "/workspace/assets/base_basic_shaded.usdz"
    if os.path.exists(table_usdz):
        log(f"导入桌面: {table_usdz}")
        table_prim = stage.DefinePrim("/World/Table", "Xform")
        table_prim.GetReferences().AddReference(table_usdz)
        table_xform = UsdGeom.Xform(table_prim)
        table_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        table_xform.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))
    else:
        log(f"WARN: 未找到桌面文件: {table_usdz}")

    urdf_path = "/workspace/assets/SO-ARM101/urdf/so_arm101.urdf"
    if os.path.exists(urdf_path):
        log(f"导入 SO-ARM101: {urdf_path}")
        try:
            from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
            config = URDFImporterConfig()
            config.urdf_path = urdf_path
            config.usd_path = "/tmp/generated"
            config.fix_base = False
            config.merge_fixed_joints = False
            config.collision_from_visuals = False
            config.merge_mesh = False
            config.run_asset_transformer = False
            config.run_multi_physics_conversion = False
            importer = URDFImporter(config=config)
            generated_usd = importer.import_urdf()
            log(f"URDF 转换完成: {generated_usd}")
            if generated_usd and os.path.exists(generated_usd):
                robot_prim = stage.DefinePrim("/World/SO_ARM101", "Xform")
                robot_prim.GetReferences().AddReference(generated_usd)
                robot_xform = UsdGeom.Xform(robot_prim)
                robot_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.85))
                log("机械臂加载到场景中")
        except Exception as e:
            log(f"ERROR: URDF 导入失败: {e}")
            import traceback
            log(traceback.format_exc())
    else:
        log(f"WARN: 未找到 URDF 文件: {urdf_path}")

    log("场景搭建完成，启动物理仿真")


setup_scene()

timeline = omni.timeline.get_timeline_interface()
timeline.play()

frame = 0
try:
    while simulation_app.is_running():
        simulation_app.update()
        frame += 1
        if frame % 500 == 0:
            log(f"运行中 - 帧 {frame}")
except KeyboardInterrupt:
    pass
finally:
    log("场景关闭")
    simulation_app.close()
