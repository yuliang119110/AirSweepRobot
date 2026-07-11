#!/usr/bin/env python3
"""
倾斜旋翼8旋翼飞行器 Sample-based MPC 控制系统 - 简化版
- 无UI界面,直接运行
- 无动画,只保存PNG图表
- 自动保存所有关键数据图表
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
import matplotlib

matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
import time

# 设置matplotlib支持中文(如果可用)
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass


# ============================================================================
# 配置参数
# ============================================================================

@dataclass
class OctorotorConfig:
    """8旋翼飞行器配置"""
    mass: float = 1.0
    gravity: float = 9.81

    rotor_positions: np.ndarray = None
    rotor_thrust_dirs: np.ndarray = None
    rotor_max_thrust: float = 9.81
    rotor_km: float = -0.05

    n_samples: int = 512  # 默认512样本(快速)
    horizon: int = 20
    dt: float = 0.05

    disturbance_force_max: float = 5.0
    disturbance_torque_max: float = 0.1
    disturbance_update_interval: float = 2.0

    sim_duration: float = 20.0  # 默认20秒

    def __post_init__(self):
        if self.rotor_positions is None:
            self.rotor_positions = np.array([
                [0.185, 0.080, 0.0],
                [-0.185, -0.080, 0.0],
                [0.080, 0.185, 0.0],
                [-0.185, 0.080, 0.0],
                [0.185, -0.080, 0.0],
                [-0.080, -0.185, 0.0],
                [0.080, -0.185, 0.0],
                [-0.080, 0.185, 0.0],
            ])

        if self.rotor_thrust_dirs is None:
            self.rotor_thrust_dirs = np.array([
                [0.353553, 0.353553, -0.866025],
                [-0.353553, -0.353553, -0.866025],
                [-0.353553, 0.353553, -0.866025],
                [-0.353553, 0.353553, -0.866025],
                [0.353553, -0.353553, -0.866025],
                [0.353553, -0.353553, -0.866025],
                [-0.353553, -0.353553, -0.866025],
                [0.353553, 0.353553, -0.866025],
            ])


# ============================================================================
# 混控器
# ============================================================================

class TiltedOctorotorMixer:
    """倾斜旋翼混控器"""

    def __init__(self, config: OctorotorConfig):
        self.config = config
        self.mixer_matrix = self._compute_mixer_matrix()

    def _compute_mixer_matrix(self) -> np.ndarray:
        mixer = np.zeros((6, 8))

        for i in range(8):
            pos = self.config.rotor_positions[i]
            thrust_dir = self.config.rotor_thrust_dirs[i]

            mixer[0:3, i] = thrust_dir
            torque = np.cross(pos, thrust_dir)
            mixer[3:6, i] = torque
            mixer[5, i] += self.config.rotor_km

        return mixer

    def compute_wrench(self, thrusts: np.ndarray) -> np.ndarray:
        return self.mixer_matrix @ thrusts


# ============================================================================
# 动力学
# ============================================================================

@jit
def dynamics_step(state: jnp.ndarray, wrench: jnp.ndarray,
                  disturbance: jnp.ndarray, mass: float,
                  gravity: float, dt: float) -> jnp.ndarray:
    """质点动力学步进"""
    pos = state[0:3]
    vel = state[3:6]

    force = wrench[0:3] + disturbance[0:3] + jnp.array([0.0, 0.0, mass * gravity])
    acc = force / mass

    new_vel = vel + acc * dt
    new_pos = pos + new_vel * dt

    return jnp.concatenate([new_pos, new_vel])


@jit
def rollout_trajectory(initial_state: jnp.ndarray,
                       thrust_sequence: jnp.ndarray,
                       mixer_matrix: jnp.ndarray,
                       disturbance: jnp.ndarray,
                       mass: float, gravity: float, dt: float) -> jnp.ndarray:
    """Rollout轨迹"""

    def step_fn(state, thrusts):
        wrench = mixer_matrix @ thrusts
        next_state = dynamics_step(state, wrench, disturbance, mass, gravity, dt)
        return next_state, next_state

    _, states = jax.lax.scan(step_fn, initial_state, thrust_sequence)
    return jnp.vstack([initial_state, states])


# ============================================================================
# MPC控制器
# ============================================================================

class SampleBasedMPC:
    """Sample-based MPC控制器"""

    def __init__(self, config: OctorotorConfig, mixer: TiltedOctorotorMixer):
        self.config = config
        self.mixer = mixer
        self.rng = jax.random.PRNGKey(0)
        self.target_state = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.prev_thrust_seq = None

    def compute_cost(self, states: jnp.ndarray, thrusts: jnp.ndarray) -> float:
        Q_pos = jnp.array([500.0, 500.0, 1000.0])
        Q_vel = jnp.array([50.0, 50.0, 100.0])
        R = 0.001

        pos_error = states[:, 0:3] - self.target_state[0:3]
        vel_error = states[:, 3:6] - self.target_state[3:6]

        state_cost = jnp.sum(pos_error ** 2 * Q_pos) + jnp.sum(vel_error ** 2 * Q_vel)

        hover_thrust = self.config.mass * self.config.gravity / 8.0
        thrust_error = thrusts - hover_thrust
        control_cost = R * jnp.sum(thrust_error ** 2)

        return state_cost + control_cost

    def sample_thrust_sequences(self, n_samples: int, horizon: int) -> jnp.ndarray:
        self.rng, subkey = jax.random.split(self.rng)

        hover_thrust = self.config.mass * self.config.gravity / 8.0

        if self.prev_thrust_seq is not None:
            noise = jax.random.normal(subkey, (n_samples, horizon, 8)) * 1.0
            mean_thrust = jnp.tile(self.prev_thrust_seq[None, :, :], (n_samples, 1, 1))
            samples = mean_thrust + noise
        else:
            noise = jax.random.normal(subkey, (n_samples, horizon, 8)) * 1.5
            samples = hover_thrust + noise

        samples = jnp.clip(samples, 0, self.config.rotor_max_thrust)
        return samples

    def optimize(self, current_state: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
        state_jax = jnp.array(current_state)
        dist_jax = jnp.array(disturbance)
        mixer_jax = jnp.array(self.mixer.mixer_matrix)

        thrust_samples = self.sample_thrust_sequences(
            self.config.n_samples, self.config.horizon
        )

        def rollout_fn(thrust_seq):
            return rollout_trajectory(
                initial_state=state_jax,
                thrust_sequence=thrust_seq,
                mixer_matrix=mixer_jax,
                disturbance=dist_jax,
                mass=self.config.mass,
                gravity=self.config.gravity,
                dt=self.config.dt
            )

        trajectories = vmap(rollout_fn)(thrust_samples)
        costs = vmap(self.compute_cost)(trajectories, thrust_samples)

        best_idx = jnp.argmin(costs)
        best_thrust_seq = thrust_samples[best_idx]

        hover_thrust = self.config.mass * self.config.gravity / 8.0
        self.prev_thrust_seq = jnp.vstack([
            best_thrust_seq[1:],
            jnp.ones(8) * hover_thrust
        ])

        return np.array(best_thrust_seq[0])


# ============================================================================
# 扰动生成器
# ============================================================================

class DisturbanceGenerator:
    """扰动生成器"""

    def __init__(self, config: OctorotorConfig):
        self.config = config
        self.current_disturbance = np.zeros(6)
        self.last_update_time = 0.0

    def update(self, current_time: float) -> np.ndarray:
        if current_time - self.last_update_time >= self.config.disturbance_update_interval:
            force = np.random.uniform(
                -self.config.disturbance_force_max,
                self.config.disturbance_force_max,
                size=3
            )
            torque = np.random.uniform(
                -self.config.disturbance_torque_max,
                self.config.disturbance_torque_max,
                size=3
            )
            self.current_disturbance = np.concatenate([force, torque])
            self.last_update_time = current_time

            print(f"[{current_time:.2f}s] Disturbance updated: "
                  f"F=[{force[0]:.2f}, {force[1]:.2f}, {force[2]:.2f}] N, "
                  f"M=[{torque[0]:.3f}, {torque[1]:.3f}, {torque[2]:.3f}] N·m")

        return self.current_disturbance


# ============================================================================
# 仿真器
# ============================================================================

class Simulator:
    """仿真器"""

    def __init__(self, config: OctorotorConfig):
        self.config = config
        self.mixer = TiltedOctorotorMixer(config)
        self.controller = SampleBasedMPC(config, self.mixer)
        self.disturbance_gen = DisturbanceGenerator(config)

        self.state = np.array([0.0, 0.0, 0.8, 0.0, 0.0, 0.0])

        self.time_history = []
        self.state_history = []
        self.thrust_history = []
        self.disturbance_history = []

    def run(self):
        print("=" * 70)
        print("Starting simulation...")
        print(f"Config: n_samples={self.config.n_samples}, "
              f"horizon={self.config.horizon}, dt={self.config.dt}s")
        print(f"Duration: {self.config.sim_duration}s")
        print("=" * 70)

        current_time = 0.0
        step_count = 0

        while current_time < self.config.sim_duration:
            step_start = time.time()

            disturbance = self.disturbance_gen.update(current_time)
            optimal_thrusts = self.controller.optimize(self.state, disturbance)
            wrench = self.mixer.compute_wrench(optimal_thrusts)

            # 防止发散
            if np.linalg.norm(self.state[0:3]) > 10.0 or np.linalg.norm(self.state[3:6]) > 20.0:
                print(f"\nWarning: State out of bounds, resetting controller")
                self.controller.prev_thrust_seq = None
                self.state = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

            self.state = np.array(dynamics_step(
                jnp.array(self.state),
                jnp.array(wrench),
                jnp.array(disturbance),
                self.config.mass,
                self.config.gravity,
                self.config.dt
            ))

            self.time_history.append(current_time)
            self.state_history.append(self.state.copy())
            self.thrust_history.append(optimal_thrusts.copy())
            self.disturbance_history.append(disturbance.copy())

            if step_count % 20 == 0:
                step_time = time.time() - step_start
                print(f"[{current_time:5.2f}s] "
                      f"Pos=[{self.state[0]:6.3f}, {self.state[1]:6.3f}, {self.state[2]:6.3f}] m, "
                      f"Vel=[{self.state[3]:6.3f}, {self.state[4]:6.3f}, {self.state[5]:6.3f}] m/s, "
                      f"Compute={step_time * 1000:.1f}ms")

            current_time += self.config.dt
            step_count += 1

        print("=" * 70)
        print("Simulation completed!")
        print("=" * 70)


# ============================================================================
# 数据可视化与保存
# ============================================================================

def save_all_plots(simulator: Simulator, output_dir: str = "."):
    """保存所有关键数据图表"""

    times = np.array(simulator.time_history)
    states = np.array(simulator.state_history)
    thrusts = np.array(simulator.thrust_history)
    disturbances = np.array(simulator.disturbance_history)
    config = simulator.config

    print("\nSaving plots...")

    # ========== 图1: 位置和速度 ==========
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    axes[0].plot(times, states[:, 0], 'b-', linewidth=2, label='X')
    axes[0].plot(times, states[:, 1], 'g-', linewidth=2, label='Y')
    axes[0].plot(times, states[:, 2], 'r-', linewidth=2, label='Z')
    axes[0].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Target Z=1m')
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('Position (m)', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Position vs Time', fontsize=14, fontweight='bold')

    axes[1].plot(times, states[:, 3], 'b-', linewidth=2, label='Vx')
    axes[1].plot(times, states[:, 4], 'g-', linewidth=2, label='Vy')
    axes[1].plot(times, states[:, 5], 'r-', linewidth=2, label='Vz')
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Velocity (m/s)', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Velocity vs Time', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_position_velocity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 01_position_velocity.png")

    # ========== 图2: 3D轨迹 ==========
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(states[:, 0], states[:, 1], states[:, 2],
            'b-', linewidth=2, alpha=0.7, label='Trajectory')
    ax.scatter(states[0, 0], states[0, 1], states[0, 2],
               c='green', s=200, marker='o', label='Start', edgecolors='black', linewidths=2)
    ax.scatter(states[-1, 0], states[-1, 1], states[-1, 2],
               c='red', s=200, marker='o', label='End', edgecolors='black', linewidths=2)
    ax.scatter(0, 0, 1, c='gold', s=300, marker='*',
               label='Target', edgecolors='black', linewidths=2)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.legend(fontsize=11)
    ax.set_title('3D Trajectory', fontsize=14, fontweight='bold')
    ax.view_init(elev=20, azim=45)

    plt.savefig(f'{output_dir}/02_trajectory_3d.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 02_trajectory_3d.png")

    # ========== 图3: 推力分配 ==========
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for i in range(8):
        axes[0].plot(times, thrusts[:, i], linewidth=1.5, alpha=0.7, label=f'Rotor {i}')
    axes[0].axhline(y=config.mass * config.gravity / 8.0,
                    color='k', linestyle='--', alpha=0.5, label='Hover thrust')
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('Thrust (N)', fontsize=12)
    axes[0].legend(fontsize=9, ncol=3)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Individual Rotor Thrusts', fontsize=14, fontweight='bold')

    total_thrust = np.sum(thrusts, axis=1)
    axes[1].plot(times, total_thrust, 'b-', linewidth=2, label='Total thrust')
    axes[1].axhline(y=config.mass * config.gravity,
                    color='r', linestyle='--', alpha=0.5, label='Weight')
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Total Thrust (N)', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Total Thrust vs Weight', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_thrust_allocation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 03_thrust_allocation.png")

    # ========== 图4: 扰动力 ==========
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    axes[0].plot(times, disturbances[:, 0], 'r-', linewidth=2, label='Fx')
    axes[0].plot(times, disturbances[:, 1], 'g-', linewidth=2, label='Fy')
    axes[0].plot(times, disturbances[:, 2], 'b-', linewidth=2, label='Fz')
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('Disturbance Force (N)', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Disturbance Forces', fontsize=14, fontweight='bold')

    axes[1].plot(times, disturbances[:, 3], 'r-', linewidth=2, label='Mx')
    axes[1].plot(times, disturbances[:, 4], 'g-', linewidth=2, label='My')
    axes[1].plot(times, disturbances[:, 5], 'b-', linewidth=2, label='Mz')
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Disturbance Torque (N·m)', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Disturbance Torques', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_disturbances.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 04_disturbances.png")

    # ========== 图5: 位置误差 ==========
    fig, ax = plt.subplots(figsize=(14, 8))

    target = np.array([0.0, 0.0, 1.0])
    pos_error = states[:, 0:3] - target
    error_norm = np.linalg.norm(pos_error, axis=1)

    ax.plot(times, error_norm, 'r-', linewidth=2.5, label='Position error')
    ax.fill_between(times, 0, error_norm, alpha=0.3, color='red')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Position Error (m)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title('Position Error Norm vs Time', fontsize=14, fontweight='bold')

    # 添加统计信息
    mean_error = np.mean(error_norm)
    max_error = np.max(error_norm)
    ax.axhline(y=mean_error, color='blue', linestyle='--', alpha=0.5,
               label=f'Mean error: {mean_error:.3f}m')
    ax.axhline(y=max_error, color='orange', linestyle='--', alpha=0.5,
               label=f'Max error: {max_error:.3f}m')
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_position_error.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 05_position_error.png")

    # ========== 图6: XY平面轨迹 ==========
    fig, ax = plt.subplots(figsize=(10, 10))

    scatter = ax.scatter(states[:, 0], states[:, 1], c=times,
                         cmap='viridis', s=30, alpha=0.6)
    ax.plot(states[:, 0], states[:, 1], 'b-', linewidth=1, alpha=0.3)
    ax.scatter(0, 0, c='red', s=300, marker='*',
               edgecolors='black', linewidths=2, label='Target', zorder=10)
    ax.scatter(states[0, 0], states[0, 1], c='green', s=150,
               marker='o', edgecolors='black', linewidths=2, label='Start')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time (s)', fontsize=12)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title('XY Plane Trajectory', fontsize=14, fontweight='bold')
    ax.axis('equal')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_trajectory_xy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 06_trajectory_xy.png")

    print(f"\nAll plots saved to: {output_dir}/")
    print("=" * 70)


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("Tilted Octorotor Sample-based MPC Control System")
    print("=" * 70)

    # 配置参数(可在此修改)
    config = OctorotorConfig(
        n_samples=512,  # 采样数
        sim_duration=20.0,  # 仿真时长(秒)
        disturbance_force_max=5.0,  # 扰动力(N)
        disturbance_torque_max=0.1  # 扰动力矩(N·m)
    )

    # 运行仿真
    simulator = Simulator(config)
    simulator.run()

    # 保存所有图表
    save_all_plots(simulator, output_dir=".")

    print("\n✓ Simulation and visualization completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

