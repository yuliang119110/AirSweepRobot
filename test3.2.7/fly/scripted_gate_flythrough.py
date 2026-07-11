"""Scripted Isaac Sim demo: drone + arm flies through a gate.

This is a presentation-first fallback. It does not train RL and does not depend
on Isaac Lab. The drone-arm model is imported from URDF, a simple door frame is
created, and the robot root is moved along a smooth trajectory through the gate.
"""

from __future__ import annotations

import argparse
import math
import os
import time
import traceback

from isaacsim import SimulationApp


parser = argparse.ArgumentParser()
parser.add_argument(
    "--urdf-path",
    default=os.environ.get("DRONE_ARM_URDF_PATH", "/assets/drone_with_arm.urdf"),
    help="Path to drone_with_arm.urdf.",
)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--loop", action="store_true", help="Loop the fly-through until the app closes.")
parser.add_argument("--max-frames", type=int, default=720)
parser.add_argument("--gate-x", type=float, default=-1.0, help="Gate center X position in meters.")
parser.add_argument("--gate-y", type=float, default=1.0, help="Gate center Y position in meters.")
parser.add_argument("--flight-z", type=float, default=1.05, help="Robot flight height and gate center Z in meters.")
parser.add_argument("--gate-width", type=float, default=1.2, help="Clear gate opening width in meters.")
parser.add_argument("--gate-height", type=float, default=1.0, help="Clear gate opening height in meters.")
parser.add_argument("--bar-size", type=float, default=0.08, help="Door-frame bar thickness in meters.")
parser.add_argument("--pass-distance", type=float, default=1.2, help="Distance past the gate center after crossing.")
parser.add_argument("--yaw-offset-deg", type=float, default=0.0, help="Extra yaw offset if the URDF forward axis is different.")
args_cli, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args_cli.headless})

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

import carb  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.usd  # noqa: E402


GATE_X = args_cli.gate_x
GATE_Y = args_cli.gate_y
GATE_Z = args_cli.flight_z
GATE_WIDTH = args_cli.gate_width
GATE_HEIGHT = args_cli.gate_height
BAR = args_cli.bar_size

START_POS = (0.0, 0.0, GATE_Z)


def create_material(stage, path, color):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def add_box(stage, path, pos, size, mat, collision=True):
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


def import_urdf_to_usd(urdf_path):
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"Drone-arm URDF not found: {urdf_path}")

    print("[gate-demo] enabling URDF importer", flush=True)
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    for ext_name in ("isaacsim.robot.schema", "isaacsim.asset.importer.urdf"):
        ext_manager.set_extension_enabled_immediate(ext_name, True)

    from isaacsim.asset.importer.urdf.impl import URDFImporter, URDFImporterConfig

    import_cfg = URDFImporterConfig()
    import_cfg.urdf_path = urdf_path
    import_cfg.usd_path = "/tmp/drone_arm_gate_demo_usd"
    import_cfg.merge_fixed_joints = False
    import_cfg.fix_base = True
    import_cfg.collision_from_visuals = False
    import_cfg.collision_type = "Convex Hull"
    import_cfg.joint_drive_type = "force"
    import_cfg.joint_target_type = "position"

    print(f"[gate-demo] importing URDF: {urdf_path}", flush=True)
    usd_path = URDFImporter(import_cfg).import_urdf()
    print(f"[gate-demo] generated USD: {usd_path}", flush=True)
    return usd_path


def make_look_at_matrix(eye, target, up):
    import numpy as np

    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    matrix = np.eye(4)
    matrix[0, :3] = right
    matrix[1, :3] = cam_up
    matrix[2, :3] = -fwd
    matrix[:3, 3] = eye
    gf_mat = Gf.Matrix4d()
    for i in range(4):
        for j in range(4):
            gf_mat[i, j] = matrix[i, j]
    return gf_mat


def set_viewport_camera():
    try:
        import omni.kit.viewport.utility as vp_util

        viewport = vp_util.get_active_viewport()
        if viewport:
            viewport.set_active_camera("/World/ViewCamera")
    except Exception as exc:
        print(f"[gate-demo] viewport camera warning: {exc}", flush=True)


def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def gate_basis():
    # Gate normal points from the origin toward the left-front gate center.
    length = math.sqrt(GATE_X * GATE_X + GATE_Y * GATE_Y)
    if length < 1.0e-6:
        raise ValueError("Gate center cannot be at the horizontal origin; set --gate-x/--gate-y away from 0.")
    normal = (GATE_X / length, GATE_Y / length, 0.0)
    lateral = (-normal[1], normal[0], 0.0)
    yaw_deg = math.degrees(math.atan2(normal[1], normal[0]))
    return normal, lateral, yaw_deg


def end_position():
    normal, _, _ = gate_basis()
    return (
        GATE_X + normal[0] * args_cli.pass_distance,
        GATE_Y + normal[1] * args_cli.pass_distance,
        GATE_Z,
    )


def offset_from_gate(forward_offset, lateral_offset=0.0, z_offset=0.0):
    normal, lateral, _ = gate_basis()
    return (
        GATE_X + normal[0] * forward_offset + lateral[0] * lateral_offset,
        GATE_Y + normal[1] * forward_offset + lateral[1] * lateral_offset,
        GATE_Z + z_offset,
    )


def trajectory(frame, max_frames):
    end_pos = end_position()
    phase = smoothstep(frame / max_frames)
    x = START_POS[0] + (end_pos[0] - START_POS[0]) * phase
    y = START_POS[1] + (end_pos[1] - START_POS[1]) * phase
    z = GATE_Z + 0.05 * math.sin(math.pi * phase)
    _, _, yaw_deg = gate_basis()
    yaw = yaw_deg + args_cli.yaw_offset_deg + 2.5 * math.sin(2.0 * math.pi * phase)
    return x, y, z, yaw


def setup_scene():
    stage = omni.usd.get_context().get_stage()
    carb.settings.get_settings().set_int("/rtx/resolution/width", 1280)
    carb.settings.get_settings().set_int("/rtx/resolution/height", 720)

    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

    ground_mat = create_material(stage, "/World/Materials/Ground", Gf.Vec3f(0.20, 0.23, 0.25))
    gate_mat = create_material(stage, "/World/Materials/Gate", Gf.Vec3f(0.05, 0.55, 0.85))
    path_mat = create_material(stage, "/World/Materials/Path", Gf.Vec3f(0.90, 0.70, 0.15))

    ground = UsdGeom.Plane.Define(stage, "/World/Ground")
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(ground_mat)

    normal, lateral, yaw_deg = gate_basis()
    left_pos = offset_from_gate(0.0, -GATE_WIDTH / 2.0)
    right_pos = offset_from_gate(0.0, GATE_WIDTH / 2.0)
    top_pos = offset_from_gate(0.0, 0.0, GATE_HEIGHT / 2.0)
    bottom_pos = offset_from_gate(0.0, 0.0, -GATE_HEIGHT / 2.0)

    add_box(stage, "/World/GateLeft", left_pos, (BAR, BAR, GATE_HEIGHT), gate_mat)
    add_box(stage, "/World/GateRight", right_pos, (BAR, BAR, GATE_HEIGHT), gate_mat)
    add_box(
        stage,
        "/World/GateTop",
        top_pos,
        (BAR, GATE_WIDTH + 2.0 * BAR, BAR),
        gate_mat,
    )
    add_box(
        stage,
        "/World/GateBottom",
        bottom_pos,
        (BAR, GATE_WIDTH + 2.0 * BAR, BAR),
        gate_mat,
    )
    end_pos = end_position()
    path_len = math.sqrt((end_pos[0] - START_POS[0]) ** 2 + (end_pos[1] - START_POS[1]) ** 2)
    path_center = ((START_POS[0] + end_pos[0]) / 2.0, (START_POS[1] + end_pos[1]) / 2.0, 0.01)
    path = add_box(stage, "/World/PathLine", path_center, (path_len, 0.025, 0.01), path_mat, collision=False)
    path.AddRotateXYZOp().Set(Gf.Vec3d(0.0, 0.0, yaw_deg))

    UsdLux.DistantLight.Define(stage, "/World/DistantLight").CreateIntensityAttr().Set(750)
    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr().Set(500)

    camera = UsdGeom.Camera.Define(stage, "/World/ViewCamera")
    camera.AddTransformOp().Set(make_look_at_matrix([1.2, -1.8, 2.1], [-0.8, 0.8, 1.0], [0.0, 0.0, 1.0]))
    camera.CreateFocusDistanceAttr().Set(4.0)
    camera.CreateFocalLengthAttr().Set(28.0)

    robot_xform = UsdGeom.Xform.Define(stage, "/World/DroneArm")
    translate_op = robot_xform.AddTranslateOp()
    rotate_op = robot_xform.AddRotateXYZOp()
    translate_op.Set(Gf.Vec3d(*START_POS))
    rotate_op.Set(Gf.Vec3f(0.0, 0.0, yaw_deg + args_cli.yaw_offset_deg))
    robot_xform.GetPrim().GetReferences().AddReference(import_urdf_to_usd(args_cli.urdf_path))

    for _ in range(10):
        simulation_app.update()

    set_viewport_camera()
    print(
        "[gate-demo] config "
        f"start={START_POS} gate=({GATE_X:.2f}, {GATE_Y:.2f}, {GATE_Z:.2f}) "
        f"end=({end_pos[0]:.2f}, {end_pos[1]:.2f}, {end_pos[2]:.2f}) "
        f"gate_size=({GATE_WIDTH:.2f} x {GATE_HEIGHT:.2f}) yaw={yaw_deg + args_cli.yaw_offset_deg:.1f}deg",
        flush=True,
    )
    print("[gate-demo] scene ready", flush=True)
    return translate_op, rotate_op


try:
    translate_op, rotate_op = setup_scene()
    omni.kit.commands.execute("Play")
    print("[gate-demo] scripted fly-through started", flush=True)

    frame = 0
    target_dt = 1.0 / 30.0
    while simulation_app.is_running():
        loop_start = time.time()
        x, y, z, yaw = trajectory(frame % args_cli.max_frames, args_cli.max_frames)
        translate_op.Set(Gf.Vec3d(x, y, z))
        rotate_op.Set(Gf.Vec3f(0.0, 0.0, yaw))
        simulation_app.update()
        if frame % 90 == 0:
            print(f"[gate-demo] frame={frame} pos=({x:.2f}, {y:.2f}, {z:.2f})", flush=True)
        frame += 1
        if not args_cli.loop and frame > args_cli.max_frames:
            break
        spent = time.time() - loop_start
        if spent < target_dt:
            time.sleep(target_dt - spent)

    print("[gate-demo] PASS: scripted gate fly-through completed", flush=True)
except Exception as exc:
    print(f"[gate-demo] FAIL: {exc}", flush=True)
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
