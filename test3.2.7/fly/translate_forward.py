#!/usr/bin/env python3
"""Isaac Sim demo: drone+arm translates forward 3m through a door frame.

Loads:
  - base_basic_pbr(2).usdz as the door frame / platform
  - drone_with_arm URDF as the robot

The drone starts at one end of the frame and translates forward 3 meters.
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
parser.add_argument(
    "--base-usdz-path",
    default=os.environ.get("BASE_USDZ_PATH", "/workspace/blander/base_basic_pbr(2).usdz"),
    help="Path to door frame / base USDZ.",
)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--loop", action="store_true")
parser.add_argument("--max-frames", type=int, default=300)

# Start position
parser.add_argument("--start-x", type=float, default=0.0)
parser.add_argument("--start-y", type=float, default=0.0)
parser.add_argument("--start-z", type=float, default=0.75)
parser.add_argument("--yaw-deg", type=float, default=0.0)

# Door frame position
parser.add_argument("--base-x", type=float, default=1.5)
parser.add_argument("--base-y", type=float, default=0.0)
parser.add_argument("--base-z", type=float, default=0.0)

# Translation
parser.add_argument("--forward-meters", type=float, default=3.0)
parser.add_argument("--yaw-offset-deg", type=float, default=0.0)
args_cli, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args_cli.headless})

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

import carb
import omni.kit.app
import omni.kit.commands
import omni.usd


# ── Helpers ───────────────────────────────────────────────────


def create_material(stage, path, color):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def add_marker_box(stage, path, pos, size, mat, collision=False):
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


# ── URDF import ──────────────────────────────────────────────


def import_urdf_to_usd(urdf_path):
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"Drone-arm URDF not found: {urdf_path}")

    print("[translate] enabling URDF importer", flush=True)
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    for ext_name in ("isaacsim.robot.schema", "isaacsim.asset.importer.urdf"):
        ext_manager.set_extension_enabled_immediate(ext_name, True)

    from isaacsim.asset.importer.urdf.impl import URDFImporter, URDFImporterConfig

    import_cfg = URDFImporterConfig()
    import_cfg.urdf_path = urdf_path
    import_cfg.usd_path = "/tmp/drone_arm_translate_usd"
    import_cfg.merge_fixed_joints = False
    import_cfg.fix_base = True
    import_cfg.collision_from_visuals = False
    import_cfg.collision_type = "Convex Hull"
    import_cfg.joint_drive_type = "force"
    import_cfg.joint_target_type = "position"

    print(f"[translate] importing URDF: {urdf_path}", flush=True)
    usd_path = URDFImporter(import_cfg).import_urdf()
    print(f"[translate] generated USD: {usd_path}", flush=True)
    return usd_path


# ── USDZ import (door frame / base) ──────────────────────────


def import_base_usdz(stage, usdz_path):
    """Import the base_basic_pbr usdz as a door frame / platform."""
    if not os.path.isfile(usdz_path):
        print(f"[translate] WARN: base USDZ not found: {usdz_path}", flush=True)
        return

    print(f"[translate] loading base USDZ: {usdz_path}", flush=True)
    base_prim = stage.DefinePrim("/World/DoorFrame", "Xform")
    base_prim.GetReferences().AddReference(usdz_path)
    base_xform = UsdGeom.Xform(base_prim)
    base_xform.AddTranslateOp().Set(Gf.Vec3d(args_cli.base_x, args_cli.base_y, args_cli.base_z))
    base_xform.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))
    print(f"[translate] door frame placed at ({args_cli.base_x:.2f}, {args_cli.base_y:.2f}, {args_cli.base_z:.2f})", flush=True)


# ── Camera ────────────────────────────────────────────────────


def make_look_at_matrix(eye, target, up):
    import numpy as np

    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)
    fwd = target - eye
    fwd_norm = np.linalg.norm(fwd)
    if fwd_norm < 1e-8:
        fwd = np.array([1.0, 0.0, 0.0])
    else:
        fwd /= fwd_norm
    right = np.cross(fwd, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        right = np.array([0.0, 1.0, 0.0])
    else:
        right /= right_norm
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
        print(f"[translate] viewport camera warning: {exc}", flush=True)


def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


# ── Translation config ───────────────────────────────────────

START = (args_cli.start_x, args_cli.start_y, args_cli.start_z)
YAW_DEG = args_cli.yaw_deg + args_cli.yaw_offset_deg
YAW_RAD = math.radians(YAW_DEG)
DIST = args_cli.forward_meters

FORWARD_DIR = (math.cos(YAW_RAD), math.sin(YAW_RAD), 0.0)

END_POS = (
    START[0] + FORWARD_DIR[0] * DIST,
    START[1] + FORWARD_DIR[1] * DIST,
    START[2],
)

# Door frame is placed between start and end (default at midpoint)
BASE_POS = (args_cli.base_x, args_cli.base_y, args_cli.base_z)


def trajectory(frame, max_frames):
    phase = smoothstep(frame / max_frames)
    x = START[0] + (END_POS[0] - START[0]) * phase
    y = START[1] + (END_POS[1] - START[1]) * phase
    z = START[2] + 0.05 * math.sin(math.pi * phase)
    return x, y, z, YAW_DEG


# ── Scene ─────────────────────────────────────────────────────


def setup_scene():
    stage = omni.usd.get_context().get_stage()
    carb.settings.get_settings().set_int("/rtx/resolution/width", 1280)
    carb.settings.get_settings().set_int("/rtx/resolution/height", 720)

    # Physics
    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

    # Materials
    ground_mat = create_material(stage, "/World/Materials/Ground", Gf.Vec3f(0.20, 0.23, 0.25))
    start_mat = create_material(stage, "/World/Materials/StartMarker", Gf.Vec3f(0.05, 0.85, 0.25))
    end_mat = create_material(stage, "/World/Materials/EndMarker", Gf.Vec3f(0.85, 0.15, 0.15))
    path_mat = create_material(stage, "/World/Materials/Path", Gf.Vec3f(0.90, 0.70, 0.15))

    # Ground
    ground = UsdGeom.Plane.Define(stage, "/World/Ground")
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(ground_mat)

    # Start marker (green)
    add_marker_box(stage, "/World/StartMarker", START, (0.10, 0.10, 0.04), start_mat, collision=False)

    # End marker (red)
    add_marker_box(stage, "/World/EndMarker", END_POS, (0.10, 0.10, 0.04), end_mat, collision=False)

    # Path line (gold)
    path_center = (
        (START[0] + END_POS[0]) / 2.0,
        (START[1] + END_POS[1]) / 2.0,
        0.01,
    )
    path_line = add_marker_box(stage, "/World/PathLine", path_center, (DIST, 0.025, 0.01), path_mat, collision=False)
    path_line.AddRotateXYZOp().Set(Gf.Vec3d(0.0, 0.0, YAW_DEG))

    # Lighting
    UsdLux.DistantLight.Define(stage, "/World/DistantLight").CreateIntensityAttr().Set(750)
    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr().Set(500)

    # Camera
    mid_x = (START[0] + END_POS[0]) / 2.0
    mid_y = (START[1] + END_POS[1]) / 2.0
    cam_lateral_x = -FORWARD_DIR[1]
    cam_lateral_y = FORWARD_DIR[0]
    cam_dist = max(DIST * 0.7 + 1.0, 2.5)
    cam_eye = (
        mid_x + cam_lateral_x * cam_dist,
        mid_y + cam_lateral_y * cam_dist,
        args_cli.start_z + 1.0,
    )
    cam_target = (mid_x, mid_y, args_cli.start_z)
    camera = UsdGeom.Camera.Define(stage, "/World/ViewCamera")
    camera.AddTransformOp().Set(make_look_at_matrix(cam_eye, cam_target, (0.0, 0.0, 1.0)))
    camera.CreateFocusDistanceAttr().Set(cam_dist * 2.0)
    camera.CreateFocalLengthAttr().Set(28.0)

    # Door frame (base USDZ)
    import_base_usdz(stage, args_cli.base_usdz_path)

    # Drone + arm
    robot_xform = UsdGeom.Xform.Define(stage, "/World/DroneArm")
    translate_op = robot_xform.AddTranslateOp()
    rotate_op = robot_xform.AddRotateXYZOp()
    translate_op.Set(Gf.Vec3d(*START))
    rotate_op.Set(Gf.Vec3f(0.0, 0.0, YAW_DEG))
    robot_xform.GetPrim().GetReferences().AddReference(import_urdf_to_usd(args_cli.urdf_path))

    for _ in range(10):
        simulation_app.update()

    set_viewport_camera()
    print(
        f"[translate] config start=({START[0]:.2f}, {START[1]:.2f}, {START[2]:.2f}) "
        f"end=({END_POS[0]:.2f}, {END_POS[1]:.2f}, {END_POS[2]:.2f}) "
        f"distance={DIST:.1f}m yaw={YAW_DEG:.1f}deg "
        f"door_frame=({args_cli.base_x:.2f}, {args_cli.base_y:.2f}, {args_cli.base_z:.2f})",
        flush=True,
    )
    print("[translate] scene ready", flush=True)
    return translate_op, rotate_op


# ── Main ──────────────────────────────────────────────────────

try:
    translate_op, rotate_op = setup_scene()
    omni.kit.commands.execute("Play")
    print("[translate] forward translation started", flush=True)

    frame = 0
    target_dt = 1.0 / 30.0
    while simulation_app.is_running():
        loop_start = time.time()
        x, y, z, yaw = trajectory(frame % args_cli.max_frames, args_cli.max_frames)
        translate_op.Set(Gf.Vec3d(x, y, z))
        rotate_op.Set(Gf.Vec3f(0.0, 0.0, yaw))
        simulation_app.update()
        if frame % 60 == 0:
            print(f"[translate] frame={frame} pos=({x:.2f}, {y:.2f}, {z:.2f})", flush=True)
        frame += 1
        if not args_cli.loop and frame > args_cli.max_frames:
            break
        spent = time.time() - loop_start
        if spent < target_dt:
            time.sleep(target_dt - spent)

    print("[translate] PASS: forward translation completed", flush=True)
except Exception as exc:
    print(f"[translate] FAIL: {exc}", flush=True)
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
