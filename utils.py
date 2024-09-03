import numpy as np
import matplotlib.pyplot as plt

# Bresenhams Line Generation Algorithm
# ref: https://www.geeksforgeeks.org/bresenhams-line-generation-algorithm/
def bresenham(x1, y1, x2, y2, w, h):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    steep = 0
    if dx <= dy:
        steep = 1
        x1, y1 = y1, x1
        x2, y2 = y2, x2
        dx, dy = dy, dx

    pk = 2 * dy - dx

    loc = []
    for _ in range(0, dx + 1):
        if (x1 < 0 or y1 < 0) or (steep == 0 and (x1 >= h or y1 >= w)) or (steep == 1 and (x1 >= w or y1 >= h)):
            break

        if steep == 0:
            loc.append([x1, y1])
        else:
            loc.append([y1, x1])

        if x1 < x2:
            x1 = x1 + 1
        else:
            x1 = x1 - 1

        if (pk < 0):
            if steep == 0:
                pk = pk + 2 * dy
            else:
                pk = pk + 2 * dy
        else:
            if y1 < y2:
                y1 = y1 + 1
            else:
                y1 = y1 - 1
    
            pk = pk + 2 * dy - 2 * dx

    return loc


def wrapAngle(radian):
    radian = radian - 2 * np.pi * np.floor((radian + np.pi) / (2 * np.pi))
    return radian


def degree2radian(degree):
    return degree / 180 * np.pi
    

def prob2logodds(prob):
    return np.log(prob / (1 - prob + 1e-15))


def logodds2prob(logodds):
    return 1 - 1 / (1 + np.exp(logodds) + 1e-15)


def normalDistribution(mean, variance):
    return np.exp(-(np.power(mean, 2) / variance / 2.0) / np.sqrt(2.0 * np.pi * variance))


def create_rotation_matrix(theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    R_inv = np.linalg.inv(R)

    return R, R_inv


def absolute2relative(position, states):
    x, y, theta = states
    pose = np.array([x, y])

    R, R_inv = create_rotation_matrix(theta)
    position = position - pose
    position = np.array(position) @ R_inv.T

    return position


def relative2absolute(position, states):
    x, y, theta = states
    pose = np.array([x, y])

    R, R_inv = create_rotation_matrix(theta)
    position = np.array(position) @ R.T
    position = position + pose

    return position


def visualize(robot, particles, best_particle, radar_list, step, title, output_path, visualize=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    ax1.clear()
    ax2.clear()
    fig.suptitle("{}\n\n number of particles:{}, step:{}".format(title, len(particles), step + 1))
    ax1.set_title("Estimated by Particles")
    ax2.set_title("Ground Truth")
    ax1.axis("off")
    ax2.axis("off")

    grid_size = best_particle.grid_size
    ax1.set_xlim(0, grid_size[1])
    ax1.set_ylim(0, grid_size[0])

    grid_size = robot.grid_size
    ax2.set_xlim(0, grid_size[1])
    ax2.set_ylim(0, grid_size[0])

    # draw map
    world_map = 1 - best_particle.grid
    ax1.imshow(world_map, cmap='gray')
    np.savetxt('world_map.txt', world_map, fmt='%.10f')
    world_map = 1 - robot.grid
    ax2.imshow(world_map, cmap='gray')

    # draw radar beams
    for (x, y) in radar_list:
        ax2.plot(x, y, "yo", markersize=1)

    # draw tragectory
    true_path = np.array(robot.trajectory)
    ax2.plot(true_path[:, 0], true_path[:, 1], "b")
    np.savetxt('true_path.txt', true_path, fmt='%.10f')
    estimated_path = np.array(best_particle.trajectory)
    ax1.plot(estimated_path[:, 0], estimated_path[:, 1], "g")
    np.savetxt('estimated_path.txt', estimated_path, fmt='%.10f')

    array = np.empty((0, 2), dtype=float)
    # draw particles position
    for p in particles:
        array = np.append(array, [np.array([p.x, p.y])], axis=0)
        ax1.plot(p.x, p.y, "go", markersize=1)
    np.savetxt('array.txt', array, fmt='%.10f')

    # draw robot position
    ax2.plot(robot.x, robot.y, "bo")

    if step % 10 == 0:
        plt.savefig('{}_{}.png'.format(output_path, step), bbox_inches='tight')

    if visualize:
        plt.draw()
        plt.pause(0.01)

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 设置图形大小和分辨率
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), dpi=100)  # 增加dpi参数提高分辨率
    ax1.clear()

    # 设置标题
    ax1.set_title("Estimated by Particles", fontsize=16)  # 增加字体大小

    # 禁用坐标轴
    ax1.axis("off")

    # 加载地图数据
    world_map = np.loadtxt('world_map.txt')
    grid_size = world_map.shape
    ax1.set_xlim(0, max(grid_size[1], grid_size[0]))
    ax1.set_ylim(0, max(grid_size[1], grid_size[0]))

    # 将地图变为白色
    world_map.fill(255)
    world_map[0][0] = 0  # 假设地图的左上角是黑色

    # 显示地图
    ax1.imshow(world_map, cmap='gray', interpolation='none')  # 添加插值参数以避免模糊

    # 添加网格线
    ax1.grid(True, color='black', linestyle='-', linewidth=0.5, which='both')  # 添加which参数

    # 加载轨迹数据
    true_path = np.loadtxt('true_path.txt')
    estimated_path = np.loadtxt('estimated_path.txt')
    estimated_path1 = np.loadtxt('estimated_path1.txt')
    # 计算平移
    dxy = estimated_path[0, :] - true_path[0, :]
    true_path += dxy


    def rotation_matrix(v1, v2):
        """计算两个二维向量之间的旋转矩阵"""
        v1 = np.asarray(v1)
        v2 = np.asarray(v2)
        a = v1 / np.linalg.norm(v1)
        b = v2 / np.linalg.norm(v2)
        c = np.dot(a, b)  # 计算两个向量的余弦值
        if np.isclose(c, 1.0):
            # 如果两个向量相同，则返回单位矩阵
            return np.eye(2)
        elif np.isclose(c, -1.0):
            # 如果两个向量相反，则返回180度旋转矩阵
            return np.array([[-1, 0], [0, -1]])
        # 计算旋转角度 theta
        theta = np.arccos(c)
        sin_theta = np.sin(theta)
        cos_theta = c
        # 构造二维旋转矩阵
        R = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        return R
    # 计算旋转矩阵
    end = 30
    R = rotation_matrix(estimated_path[end, :] - estimated_path[0, :], true_path[end, :] - true_path[0, :])
    R1 = rotation_matrix(estimated_path1[end, :] - estimated_path1[0, :], true_path[end, :] - true_path[0, :])
    # 对 estimated_path 进行操作
    for i in range(len(estimated_path)):
        # 将点平移，使 true_path[0, :] 成为原点
        estimated_path[i, :] -= true_path[0, :]
        # 应用旋转矩阵
        estimated_path[i, :] = R.dot(estimated_path[i, :])
        # 平移回原来的位置
        estimated_path[i, :] += true_path[0, :]

    # 对 estimated_path1 进行操作
    for i in range(len(estimated_path1)):
        # 将点平移，使 true_path[0, :] 成为原点
        estimated_path1[i, :] -= true_path[0, :]
        # 应用旋转矩阵
        estimated_path1[i, :] = R1.dot(estimated_path1[i, :])
        # 平移回原来的位置
        estimated_path1[i, :] += true_path[0, :]

    # 绘制轨迹
    ax1.plot(true_path[:, 0], true_path[:, 1], "r--", label='True Path')  # 红色虚线
    ax1.plot(estimated_path1[:,0], estimated_path1[:,1], "b-", label='fastSlam_pso')  # 蓝色线条，带圆圈标记
    ax1.plot(estimated_path[:,0], estimated_path[:,1], "g-", label='fastSlam')  # 绿色线条，带星号标记

    y1 = abs(estimated_path[:,0] - true_path[:, 0])
    y2 = abs(estimated_path[:,1] - true_path[:, 1])
    y3 = abs(estimated_path1[:,0] - true_path[:, 0])
    y4 = abs(estimated_path1[:,1] - true_path[:, 1])
    ax2.plot(y1, 'g', label='Difference of fastSlam')
    ax3.plot(y2, 'g', label='Difference of fastSlam')
    ax2.plot(y3, 'b', label='Difference of fastSlam_pso')
    ax3.plot(y4, 'b', label='Difference of fastSlam_pso')

    ax2.legend(loc='upper left')
    ax2.set_title("Path Differences X")
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Difference")

    ax3.legend(loc='upper left')
    ax3.set_title("Path Differences Y")
    ax3.set_xlabel("Index")
    ax3.set_ylabel("Difference")

    # # 加载粒子位置数据
    # array = np.loadtxt('array.txt')
    # ax1.plot(array[:,0], array[:,1], "go", markersize=1, label='Particles')  # 绿色圆点

    # 添加图例
    ax1.legend(loc='best', fontsize=12)  # 选择合适的位置并设置字体大小

    # 显示图形
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    fig.savefig('path_differences.png', dpi=300)  # 保存为PNG格式，DPI为300
    plt.show()


# if __name__ == '__main__':
#     fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
#     ax1.clear()
#     # ax2.clear()
#     ax1.set_title("Estimated by Particles")
#     # ax2.set_title("Ground Truth")
#     ax1.axis("off")
#     # ax2.axis("off")
#
#
#
#     # draw map
#     world_map = data = np.loadtxt('world_map.txt')
#
#     grid_size = world_map.shape
#     ax1.set_xlim(0, max(grid_size[0], grid_size[1]))
#     ax1.set_ylim(0, max(grid_size[0], grid_size[1]))
#     world_map.fill(255)
#     world_map[0][0] = 0
#     ax1.imshow(world_map, cmap='gray')
#     ax1.grid(True, color='black', linestyle='-', linewidth=0.5)
#
#     # draw tragectory
#     true_path = np.loadtxt('true_path.txt')
#     estimated_path = np.loadtxt('estimated_path.txt')
#     estimated_path1 = np.loadtxt('estimated_path1.txt')
#     dxy = estimated_path[0, :] - true_path[0, :]
#     ax1.plot(true_path[:, 0] + dxy[0], true_path[:, 1] + dxy[1], "b")
#     ax1.plot(estimated_path1[:, 0], estimated_path1[:, 1], "r")
#     ax1.plot(estimated_path[:, 0], estimated_path[:, 1], "g")
#
#     array = np.loadtxt('array.txt')
#     # draw particles position
#     # for p in particles:
#     #     array = np.append(array, [np.array([p.x, p.y])], axis=0)
#     ax1.plot(array[:, 0], array[:, 1], "go", markersize=1)
#     plt.show()