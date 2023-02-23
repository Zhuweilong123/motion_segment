"""
    这里的数据预处理是包含了两种格式处理，一种是将3x3的旋转矩阵转换为三维向量，即yaw，pitch和roll三个轴的角度；另一种是将3x3的矩阵转换为四元数也就是w p q r
    数据集也分为了两种形式，一种是kitti数据集是1x12的向量，一种是7scenes数据集是4x4的齐次矩阵
"""
from pyquaternion import Quaternion
import numpy as np
import math

rotate_matrix = [[-0.0174524064372832, -0.999847695156391, 0.0],
                 [0.308969929589947, -0.00539309018185907, -0.951056516295153],
                 [0.950911665781176, -0.0165982248672099, 0.309016994374948]]

RM = np.array(rotate_matrix)


# 旋转矩阵转换为四元数
def MatrixToQuaternion(rotateMatrix):
    q = Quaternion(matrix=rotateMatrix)
    print(q)  # 0.567 +0.412i -0.419j +0.577k
    print(f"x: {q.x}, y: {q.y}, z: {q.z}, w: {q.w}")
    # x: 0.41198412875061946, y: -0.41923809520381, z: 0.5770317346112972, w: 0.567047506333421
    return q


#if __name__ == '__main__':


# class MatrixToAngle:
# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)  # R的转置*R，因为旋转矩阵必须是正交矩阵，这里得到的乘积应该近似于单位矩阵
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)  # 默认求2范数
    return n < 1e-6


def R_to_angle(Rt):
    # Ground truth pose is present as [R | t]
    # R: Rotation Matrix, t: translation vector
    # transform matrix to angles，最后输出的是一个1x6的数组即角度加xyz
    Rt = np.reshape(np.array(Rt[:-1]), (3, 4))
    #Rt = np.reshape(np.array(Rt), (3, 4))
    #Rt = Rt - t  # 转换到cam2坐标系，因为x轴的正方向是cam0的位置，所以cam0→cam2需要减去这个平移量(已经做过转换)
    t = Rt[:, -1]  # 倒数第一列
    R = Rt[:, :3]  # 正数到第三列

    assert (isRotationMatrix(R))  # 断言

    x, y, z = euler_from_matrix(R)

    theta = [x, y, z]
    pose_6 = np.concatenate((t, theta))  # concatenate是numpy对array拼接的函数
    assert (pose_6.shape == (6,))
    return pose_6


def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])
    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])
    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def euler_from_matrix(matrix):
    # y-x-z Tait–Bryan angles intrincic
    # the method code is taken from https://github.com/awesomebytes/delta_robot/blob/master/src/transformations.py

    i = 2
    j = 0
    k = 1
    repetition = 0
    frame = 1
    parity = 0

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > _EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > _EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return np.array([ax, ay, az])


def normalize_angle_delta(angle):
    if (angle > np.pi):
        angle = angle - 2 * np.pi
    elif (angle < -np.pi):
        angle = 2 * np.pi + angle
    return angle

