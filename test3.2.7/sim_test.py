"""
Isaac Sim 6.0.1 GUI 仿真测试脚本
使用 SimulationApp 启动 GUI,通过纯 USD + UsdPhysics API 创建物理场景。
"""
import numpy as np
import carb
from isaacsim.simulation_app import SimulationApp

simulation_app = SimulationApp({"headless": False})

from pxr import UsdGeom, UsdPhysics, UsdLux, UsdShade, Sdf, Gf
import omni.usd
import omni.kit.commands
import omni.timeline


def create_material(stage, path, color):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def add_dynamic_cube(stage, path, pos, size, mat):
    xform = UsdGeom.Xform.Define(stage, path)
    cube = UsdGeom.Cube.Define(stage, path + "/geometry")
    cube.CreateSizeAttr().Set(size)
    xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    return xform


def add_dynamic_sphere(stage, path, pos, radius, mat):
    xform = UsdGeom.Xform.Define(stage, path)
    sphere = UsdGeom.Sphere.Define(stage, path + "/geometry")
    sphere.CreateRadiusAttr().Set(radius)
    xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(mat)
    return xform


def add_dynamic_cylinder(stage, path, pos, radius, height, mat):
    xform = UsdGeom.Xform.Define(stage, path)
    cyl = UsdGeom.Cylinder.Define(stage, path + "/geometry")
    cyl.CreateRadiusAttr().Set(radius)
    cyl.CreateHeightAttr().Set(height)
    xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(mat)
    return xform


def setup_scene():
    stage = omni.usd.get_context().get_stage()

    # 物理场景
    scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    # 地面
    ground = UsdGeom.Plane.Define(stage, "/World/Ground")
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    ground_mat = create_material(stage, "/World/GroundMaterial", Gf.Vec3f(0.25, 0.27, 0.3))
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(ground_mat)

    # 材质
    mat_orange = create_material(stage, "/World/MatOrange", Gf.Vec3f(0.8, 0.5, 0.1))
    mat_blue = create_material(stage, "/World/MatBlue", Gf.Vec3f(0.1, 0.3, 0.9))
    mat_purple = create_material(stage, "/World/MatPurple", Gf.Vec3f(0.6, 0.2, 0.8))
    mat_yellow = create_material(stage, "/World/MatYellow", Gf.Vec3f(0.95, 0.8, 0.1))

    # 堆叠塔
    for i in range(5):
        add_dynamic_cube(stage, f"/World/TowerCube_{i}", [0, 0, 0.05 + i * 0.105], 0.1, mat_orange)

    # 散布球体
    for i in range(6):
        angle = i * (2 * np.pi / 6)
        add_dynamic_sphere(stage, f"/World/Sphere_{i}", [0.6 * np.cos(angle), 0.6 * np.sin(angle), 0.5], 0.06, mat_blue)

    # 散布圆柱
    for i in range(4):
        angle = i * (np.pi / 2) + 0.4
        add_dynamic_cylinder(stage, f"/World/Cylinder_{i}", [np.cos(angle), np.sin(angle), 0.4], 0.05, 0.2, mat_purple)

    # 固定锥体(装饰,无刚体物理)
    cone = UsdGeom.Cone.Define(stage, "/World/Cone_0")
    cone.CreateRadiusAttr().Set(0.1)
    cone.CreateHeightAttr().Set(0.3)
    cone.AddTranslateOp().Set(Gf.Vec3d(1.5, 0, 0.15))
    UsdShade.MaterialBindingAPI.Apply(cone.GetPrim()).Bind(mat_yellow)

    # 光照
    UsdLux.DistantLight.Define(stage, "/World/DistantLight").CreateIntensityAttr().Set(800)
    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr().Set(400)

    carb.log_info("[sim_test] Scene setup complete. Starting physics simulation.")


setup_scene()

# 启动物理仿真 (使用 timeline 接口，兼容 5.1.0 和 6.0.1)
timeline = omni.timeline.get_timeline_interface()
timeline.play()

frame = 0
while simulation_app.is_running():
    simulation_app.update()
    frame += 1
    if frame % 1000 == 0:
        carb.log_info(f"[sim_test] Running frame {frame}")

simulation_app.close()
