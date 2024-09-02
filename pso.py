import numpy as np
from scipy.optimize import minimize
from robot import Robot
from tqdm import tqdm


# 目标函数：最大化运动概率和测量概率的乘积
def objective_function(pose, measurement_model, z_star, init_grid, config):
    """
    目标函数：计算给定粒子状态的目标函数值（运动概率和测量概率的乘积）。
    """
    x, y, theta = pose
    tmp_r = Robot(x, y, theta, np.copy(init_grid), config)
    # Calculate particle's weights depending on robot's measurement
    z, _, _ = tmp_r.sense()
    # 计算测量概率
    z, _, _ = tmp_r.sense()
    measurement_prob = measurement_model.measurement_model(z_star, z)

    # 返回目标函数值（负对数似然，PSO 是最小化问题，所以我们用负值）
    return - measurement_prob


def optimize_particles_with_pso(idx, particles, measurement_model, z_star,
                                init_grid, config, num_particles=30, num_iterations=10):
    """
    使用 PSO 优化粒子状态。
    """
    # 初始化粒子群位置和速度
    particles_prev_positions = np.array([[p.x, p.y, p.theta] for p in particles])
    particle_positions = np.array([[p.x, p.y, p.theta] for p in particles])
    particle_velocities = np.zeros_like(particle_positions)

    # 初始化个人最佳位置和全局最佳位置
    personal_best_positions = np.copy(particle_positions)
    personal_best_scores = np.array([objective_function(particle_positions[i], measurement_model,
                                                        z_star, init_grid, config) for i in
                                     range(len(particle_positions))])
    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)

    # PSO 参数
    inertia_weight = 0.5
    cognitive_constant = 1.5
    social_constant = 1.5

    for iteration in tqdm(range(num_iterations), desc='Optimization Progress'):
        # 更新进度条上的信息
        for i in range(num_particles):
            # 更新速度
            cognitive_velocity = cognitive_constant * np.random.rand() * (
                    personal_best_positions[i] - particle_positions[i])
            social_velocity = social_constant * np.random.rand() * (global_best_position - particle_positions[i])
            particle_velocities[i] = inertia_weight * particle_velocities[i] + cognitive_velocity + social_velocity

            # 更新位置
            particle_positions[i] += particle_velocities[i]

            # # 应用边界约束
            # particle_positions[i] = np.clip(particle_positions[i], [bound[0] for bound in bounds],
            #                                 [bound[1] for bound in bounds])

            # 计算新的适应度值
            score = objective_function(particle_positions[i], measurement_model,
                                                        z_star, init_grid, config)

            # 更新个人最佳位置
            if score < personal_best_scores[i]:
                personal_best_positions[i] = particle_positions[i]
                personal_best_scores[i] = score

            # 更新全局最佳位置
            if score < global_best_score:
                global_best_position = particle_positions[i]
                global_best_score = score
        # 更新tqdm的进度条
        # tqdm.write(f'idx: {idx}, iter: {iteration + 1}/{num_iterations}, bestfit: {global_best_score}')

    for i in range(len(particles)):
        elem = particles[i]
        x, y, theta = personal_best_positions[i]
        elem.set_states(x, y, theta)
        particles[i] = elem
    return personal_best_positions
