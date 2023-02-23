# 根据深度信息、相机姿态（s→r）、相机内参、图像source和图像target
# sfmLearner
# batch_size会出现问题
from __future__ import division
import torch
import torch.nn.functional as F
from torch_sparse import coalesce
import numpy as np
from torch.autograd import Variable


pixel_coords = None


# 通过对source view到target view的转换矩阵求逆，得到target view到source view的转换矩阵
# 有一个问题是怎么判断该矩阵是否可逆呢

############Reference: https://github.com/JiawangBian/SC-SfMLearner-Release/blob/master/inverse_warp.py#################
def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


# 确定尺寸是否对应
def check_sizes(input, input_name, expected):
    #input总维度和expected的字数长度
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))  # 默认数字的对不对得上
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(
        input_name, 'x'.join(expected), list(input.size()))


def invert_intrinsics_matrix(intrinsics_mat):
    """Inverts an intrinsics matrix.
    Inverting matrices in not supported on TPU. The intrinsics matrix has however
    a closed form expression for its inverse, and this function invokes it.
    Args:
        intrinsics_mat: A tensor of shape [.... 3, 3], representing an intrinsics
        matrix `(in the last two dimensions).
    Returns:
        A tensor of the same shape containing the inverse of intrinsics_mat
    """
    intrinsics_mat = intrinsics_mat
    intrinsics_mat_cols = torch.unbind(intrinsics_mat, dim=-1)
    if len(intrinsics_mat_cols) != 3:
        raise ValueError('The last dimension of intrinsics_mat should be 3, not '
                    '%d.' % len(intrinsics_mat_cols))

    fx, _, _ = torch.unbind(intrinsics_mat_cols[0], dim=-1)
    _, fy, _ = torch.unbind(intrinsics_mat_cols[1], dim=-1)
    x0, y0, _ = torch.unbind(intrinsics_mat_cols[2], dim=-1)

    zeros = torch.zeros_like(fx)
    ones = torch.ones_like(fx)

    row1 = torch.stack([1.0 / fx, zeros, zeros], dim=-1)
    row2 = torch.stack([zeros, 1.0 / fy, zeros], dim=-1)
    row3 = torch.stack([-x0 / fx, -y0 / fy, ones], dim=-1)

    return torch.stack([row1, row2, row3], dim=-1)


#输入深度图和逆内参，得到相机坐标，即将图片坐标系的像素点投影到相机平面上
def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(
        b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]

    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    X_norm = 2*(X / Z)/(w-1) - 1
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b, h, w, 2)


def cam2homo(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode='zeros'):
    """
    Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.view(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot.bmm(cam_coords_flat)
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_homo = X / Z  # Homogeneous coords X
    Y_homo = Y / Z  # Homogeneous coords Y
    pixel_coords_homo = torch.stack([X_homo, Y_homo], dim=2)  # [B, H*W, 2]

    X_norm = 2 * (X / Z) / (w - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * (Y / Z) / (h - 1) - 1  # Idem [B, H*W]
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
        X_norm[X_mask] = 2  # make sure that no point in warped image is a combination of im and gray
        Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords_norm = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]

    valid_points = pixel_coords_norm.view(b, h, w, 2).abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(1).float()

    return pixel_coords_homo.view(b, h, w, 2), valid_mask


def mat2euler(R):
    """
    Convert rotation matrix to euler angles.
    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    Returns:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    """
    bs = R.size(0)

    sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
    singular = (sy < 1e-6).float()

    x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    y = torch.atan2(-R[:, 2, 0], sy)
    z = torch.atan2(R[:, 1, 0], R[:, 0, 0])

    xs = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    ys = torch.atan2(-R[:, 2, 0], sy)
    zs = R[:, 1, 0] * 0

    out_euler_x = x * (1 - singular) + xs * singular
    out_euler_y = y * (1 - singular) + ys * singular
    out_euler_z = z * (1 - singular) + zs * singular

    return torch.stack([out_euler_x, out_euler_y, out_euler_z], dim=-1)


def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:,
                                            1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat


def pose_mof2mat(mof, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:
        mof: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6, H, W]
    Returns:
        A transformation matrix -- [B, 3, 4, H, W]
    """
    bs, _, hh, ww = mof.size()
    translation = mof[:,:3].reshape(bs,3,1,hh,ww)    # [B, 3, 1, H, W]
    rot = mof[:,3:].mean(dim=[2,3]) # [B*1, 3]

    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)    # [B*1, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)     # [B*1, 3, 3]

    rot_mat = rot_mat.reshape(bs,3,3,1,1).repeat(1,1,1,hh,ww)   # [B, 3, 3, H, W]
    transform_mat = torch.cat([rot_mat, translation], dim=2)    # [B*N, 3, 4]
    # pdb.set_trace()
    return transform_mat


# 这里的inverse warp是backward mapping，即已得到想要的映射后的目标图像，反过来求源图像的坐标信息
def inverse_warp(img, depth, pose, intrinsics, rotation_mode, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    inverse warp:target → source(得到不是source image而是将target图像转换到source图像的坐标系中)
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]（这里的B是batch_size）:sample pixels即要采样利用双线性插值的周围四个像素
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]  这里的内参矩阵是？
    Returns:
        projected_img: Source image warped to the target image plane
        valid_points: Boolean array indicating point validity
    """

    check_sizes(img, 'img', 'BHW3')
    check_sizes(depth, 'depth', 'BHW')
    # 判断姿态是六自由度欧拉角的格式还是转换矩阵4x4的格式
    if rotation_mode == 'euler':
        check_sizes(pose, 'pose', 'B6')
        pose_mat = pose_vec2mat(pose, rotation_mode)
    elif rotation_mode == 'translation':
        check_sizes(pose, 'pose', 'B34')
        pose_mat = pose
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    # 将target frame 坐标系转换到相机坐标系得到了像素对应的三维点P
    cam_coords = pixel2cam(depth, intrinsics.inverse())  # [B,3,H,W]

    # Get projection matrix for tgt camera frame to source pixel frame:
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    # 乘以转换矩阵即得到相机移动后的三维点P', 再投影到source图像坐标系中
    src_pixel_coords = cam2pixel(
        cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    # 双线性插值（利用原source图像的周围四个点来获得像素值），把得到的连续值转换成离散值
    projected_img = F.grid_sample(
        img, src_pixel_coords, padding_mode=padding_mode)

    # 这里的有效点指的是什么？
    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

    return projected_img, valid_points


# 返回除了转换后的像素坐标系，还有深度信息（Z轴即前后方向）
def cam2pixel2(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    X_norm = 2*(X / Z)/(w-1) - 1
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1)+(X_norm < -1)).detach()
        # make sure that no point in warped image is a combinaison of im and gray
        X_norm[X_mask] = 2
        Y_mask = ((Y_norm > 1)+(Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b, h, w, 2), Z.reshape(b, 1, h, w)


# 对比inverse_warp函数，首先维度的顺序不相同一个是[H,W,C]一个是[C,H,W]，第二个区别是利用valid_point得到valid_mask,第三个区别是深度图也转换到了另外一个坐标系中
def inverse_warp2(img, depth, ref_depth, pose, intrinsics, rotation_mode, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, 1, H, W]  用于反投影到相机坐标系
        ref_depth: the source depth map (where to sample depth) -- [B, 1, H, W]  对比warp1多了源图的深度信息，用于inverse warp depth
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_mask: Float array indicating point validity
        projected_depth: sampled depth from source image
        computed_depth: computed depth of source image using the target depth
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'B1HW')
    check_sizes(ref_depth, 'ref_depth', 'B1HW')
    if rotation_mode == 'euler':
        check_sizes(pose, 'pose', 'B6')
        pose_mat = pose_vec2mat(pose, rotation_mode)
    elif rotation_mode == 'translation':
        check_sizes(pose, 'pose', 'B34')
        pose_mat = pose
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth.squeeze(1), torch.inverse(intrinsics))  # [B,3,H,W]

    #pose_mat = pose_vec2mat(pose)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]


    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    # cam2pixel2：返回两个张量，第一个是新图像坐标系的坐标，第二个是每个点的深度信息z
    src_pixel_coords, computed_depth = cam2pixel2(cam_coords, rot, tr, padding_mode)  # [B,H,W,2]

    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode, align_corners=False)

    #### 什么是valid point:像素移动距离归一到了(0,1)之间，即偏移不超过图像高宽的才为有效点
    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1  # max函数第一维是max的值，第二维是对应值的位置即索引
    valid_mask = valid_points.unsqueeze(1).float()

    # 利用inverse warp的方式得到扭曲后的深度图
    projected_depth = F.grid_sample(ref_depth, src_pixel_coords, padding_mode=padding_mode, align_corners=False)

    #return projected_img, valid_mask, projected_depth, computed_depth
    return projected_img, projected_depth

########################################################################################################################
#############################Reference:https://github.com/SeokjuLee/Insta-DM/blob/master/rigid_warp.py##################
########################################################################################################################

def transform_scale_consistent_depth(depth, pose, intrinsics, rotation_mode='euler', padding_mode='zeros'):
    """
    Transform scale of depth with given pose change.
    Args:
        depth: depth map of the target image -- [B, 1, H, W]                    // tgt_depth
        pose: 6DoF pose parameters from target to source -- [B, 6]              //
        intrinsics: camera intrinsic matrix -- [B, 3, 3]                        //
    Returns:
        scale_transformed_depth: Source depth scaled to the target depth
    """
    check_sizes(depth, 'depth', 'B1HW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    cam_coords = pixel2cam(depth.squeeze(1), intrinsics.inverse())  # D * K_inv * X, [B,3,H,W]
    pose_mat = pose_vec2mat(pose, rotation_mode)  # RT, [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    _, computed_depth, _ = cam2pixel2(cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    # pdb.set_trace()

    return computed_depth


def depth2flow(depth, pose, intrinsics, reverse_pose=False, rotation_mode='euler', padding_mode='zeros'):
    """
    Depth + Pose => Flow
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, 1, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    check_sizes(depth, 'depth', 'B1HW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, hh, ww = depth.size()
    cam_coords = pixel2cam(depth.squeeze(1), intrinsics.inverse())  # D * K_inv * X, [B,3,H,W]
    pose_mat = pose_vec2mat(pose, rotation_mode)  # RT, [B,3,4]

    if reverse_pose:
        aux_mat = torch.zeros([batch_size, 4]).cuda().unsqueeze(1)
        aux_mat[:, :, 3] = 1
        pose_mat = torch.cat([pose_mat, aux_mat], dim=1)  # [B, 4, 4]
        pose_mat = [t.inverse() for t in torch.functional.unbind(pose_mat)]
        pose_mat = torch.stack(pose_mat)  # [B, 4, 4]
        pose_mat = pose_mat[:, :3, :]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B,3,4]
    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    flow_grid, valid_mask = cam2homo(cam_coords, rot, tr, padding_mode)  # [B,H,W,2], [B,1,H,W]
    mgrid_np = np.expand_dims(np.mgrid[0:ww, 0:hh].transpose(2, 1, 0).astype(np.float32), 0).repeat(batch_size, axis=0)
    mgrid = torch.from_numpy(mgrid_np).cuda()  # [B,H,W,2]

    flow_rigid = flow_grid - mgrid
    flow_rigid = flow_rigid.permute(0, 3, 1, 2)

    return flow_rigid, valid_mask


def cam2pix_trans(cam_coords, pose_mat, intrinsics):
    b, _, h, w = cam_coords.size()
    rot, tr = pose_mat[:, :, :3], pose_mat[:, :, -1:]  # [B, 3, 3], [B, 3, 1]
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords_trans = rot @ cam_coords_flat  # [B, 3, H*W]
    cam_coords_trans = cam_coords_trans + tr  # [B, 3, H*W]

    X = cam_coords_trans[:, 0]  # [B, H*W]
    Y = cam_coords_trans[:, 1]  # [B, H*W]
    Z = cam_coords_trans[:, 2].clamp(min=1e-3)  # [B, H*W]

    # 归一化的方式和普通的cam2pix函数不一样，目的是：
    X_norm = (X / Z)  # [B, H*W]
    Y_norm = (Y / Z)  # [B, H*W]
    Z_norm = (Z / Z)  # [B, H*W]
    P_norm = torch.stack([X_norm, Y_norm, Z_norm], dim=1)  # [B, 3, H*W]
    pix_coords = (intrinsics @ P_norm).permute(0, 2, 1)[:, :, :2]  # [B, H*W, 2]

    return pix_coords.reshape(b, h, w, 2), Z.reshape(b, 1, h, w)


def forward_warp(img, depth, pose, intrinsics, rotation_mode, upscale=None, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, C, H, W]
        depth: depth map of the source image -- [B, 1, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_points: Boolean array indicating point validity

        plt.close('all')
        plt.figure(1); plt.imshow(img[0,0].detach().cpu()); plt.colorbar(); plt.ion(); plt.show()
        plt.figure(2); plt.imshow(img_w[0,0].detach().cpu()); plt.colorbar(); plt.ion(); plt.show()
        plt.figure(3); plt.imshow(depth[0,0].detach().cpu()); plt.colorbar(); plt.ion(); plt.show()
        plt.figure(4); plt.imshow(depth_w[0,0].detach().cpu()); plt.colorbar(); plt.ion(); plt.show()
        plt.figure(5); plt.imshow(fw_val[0,0].detach().cpu()); plt.colorbar(); plt.ion(); plt.show()
        plt.figure(6); plt.imshow(iw_val[0,0].detach().cpu()); plt.colorbar(); plt.ion(); plt.show()
        plt.figure(7); plt.imshow(valid[0,0].detach().cpu()); plt.colorbar(); plt.ion(); plt.show()
    """
    check_sizes(depth, 'depth', 'B1HW')

    check_sizes(intrinsics, 'intrinsics', 'B33')
    if rotation_mode == 'euler':
        check_sizes(pose, 'pose', 'B6')
        pose_mat = pose_vec2mat(pose, rotation_mode)
    elif rotation_mode == 'translation':
        check_sizes(pose, 'pose', 'B34')
        pose_mat = pose

    bs, _, hh, ww = depth.size()
    # 为了填充前向扭曲导致的空洞，对深度图进行上采样，内参也随着改变
    depth_u = F.interpolate(depth, scale_factor=upscale).squeeze(1)
    intrinsic_u = torch.cat((intrinsics[:, 0:2] * upscale, intrinsics[:, 2:]), dim=1)

    cam_coords = pixel2cam(depth_u, intrinsic_u.inverse())  # [B,3,uH,uW]

    pcoords, Z = cam2pix_trans(cam_coords, pose_mat, intrinsics)  # [B,uH,uW,2], [B,1,uH,uW]

    depth_w, fw_val = [], []
    # 对target图像的像素进行sparse tensor coding
    # zip为打包，将coo和z的每个对应位置的元素重新组合，生成一个个新的元组
    # 这里的目的是为了将坐标乘以转换矩阵后所有不规范的（小于0或大于高宽）的全部赋值为最大值，即把无效的坐标mask out
    for coo, z in zip(pcoords, Z):
        idx = coo.reshape(-1, 2).permute(1, 0).long()[[1, 0]]  # [2，B*H*W]，long将数据转为长整型
        val = z.reshape(-1)  #[B*H*W]
        idx[0][idx[0] < 0] = hh  # 转换后像素x坐标小于0的数字的都统一赋值为深度图的高
        idx[0][idx[0] > hh - 1] = hh
        idx[1][idx[1] < 0] = ww
        idx[1][idx[1] > ww - 1] = ww
        # 稀疏化
        _idx, _val = coalesce(idx, 1 / val, m=hh + 1, n=ww + 1,
                              op='max')  # Cast an index with maximum-inverse-depth: we do NOT interpolate points! >> errors near boundary
        # dense化
        depth_w.append(1 / torch.sparse.FloatTensor(_idx, _val, torch.Size([hh + 1, ww + 1])).to_dense()[:-1, :-1])
        fw_val.append(
            1 - (torch.sparse.FloatTensor(_idx, _val, torch.Size([hh + 1, ww + 1])).to_dense()[:-1, :-1] == 0).float())
        # pdb.set_trace()
    depth_w = torch.stack(depth_w, dim=0)
    fw_val = torch.stack(fw_val, dim=0)
    depth_w[fw_val == 0] = 0

    # size为([batch_size, 1, 4]),张量全为(0,0,0,1)
    aux_mat = torch.tensor([0, 0, 0, 1]).type_as(pose_mat).unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1)  # [B,1,1]

    # 转换为齐次矩阵([4,4])，求逆？这里是因为齐次矩阵求逆更容易？
    pose_mat_inv = torch.inverse(torch.cat([pose_mat, aux_mat], dim=1))  # [B,4,4]
    trans_vec = pose_mat_inv[:, :3, 3]
    euler_vec = mat2euler(pose_mat_inv[:, :3, :3])
    pose_inv = torch.cat([trans_vec, euler_vec], dim=1)

    # 这里使用的还是inverse warping？
    # 这里的深度是经过上采样和dense化后的值，姿态是原姿态的逆
    img_w, iw_val = inverse_warp(img, depth_w, pose_inv, intrinsics, rotation_mode='euler')
    iw_val = iw_val.float().unsqueeze(1)
    depth_w = depth_w.unsqueeze(1)
    valid = fw_val.unsqueeze(1) * iw_val
    # pdb.set_trace()

    return img_w * valid, depth_w * valid, valid


def cam2pixel_mof(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """
    Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]    // tgt_depth * K_inv
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 3, H, W]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1, H, W]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    bs, _, hh, ww = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(bs, 3, 1, -1)  # [B, 3, 1, H*W]
    c2p_rot_flat = proj_c2p_rot.reshape(bs, 3, 3, -1)  # [B, 3, 3, H*W]
    c2p_tr_flat = proj_c2p_tr.reshape(bs, 3, 1, -1)  # [B, 3, 1, H*W]

    ### Rotation ###
    if proj_c2p_rot is not None:
        pcoords = c2p_rot_flat.permute(0, 2, 1, 3) * cam_coords_flat  # [B, 3, 3, H*W] // (K * P) * (D_tgt * K_inv)
        pcoords = pcoords.sum(dim=1, keepdim=True).permute(0, 2, 1, 3)  # [B, 3, 1, H*W]
    else:
        pcoords = cam_coords_flat
    # pdb.set_trace()
    '''
        plt.close('all')
        bb = 0
        plt.figure(1); plt.imshow(pcoords[bb,0,0].reshape(hh, ww).detach().cpu()); plt.colorbar(); plt.ion(); plt.show();

    '''

    ### Translation ###
    if proj_c2p_tr is not None:
        pcoords = pcoords + c2p_tr_flat  # [B, 3, 1, H*W]

    pcoords = pcoords.reshape(bs, 3, -1)

    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2 * (X / Z) / (ww - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * (Y / Z) / (hh - 1) - 1  # Idem [B, H*W]
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
        X_norm[X_mask] = 2  # make sure that no point in warped image is a combination of im and gray
        Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    # pdb.set_trace()
    '''
        aaa = pixel_coords.reshape(bs, hh, ww, 2)[0,:,:,0]
        bbb = Z.reshape(bs, 1, hh, ww)[0,0]
        ccc = X_norm.reshape(bs, 1, hh, ww)[0,0]
        ddd = Y_norm.reshape(bs, 1, hh, ww)[0,0]
        plt.figure(1), plt.imshow(aaa.detach().cpu()), plt.colorbar(), plt.ion(), plt.show()
        plt.figure(2), plt.imshow(bbb.detach().cpu()), plt.colorbar(), plt.ion(), plt.show()
        plt.figure(3), plt.imshow(ccc.detach().cpu()), plt.colorbar(), plt.ion(), plt.show()
        plt.figure(4), plt.imshow(ddd.detach().cpu()), plt.colorbar(), plt.ion(), plt.show()
        plt.figure(9), plt.imshow(Z.reshape(bs, 1, hh, ww).detach().cpu().numpy()[0,0]), plt.colorbar(), plt.tight_layout(), plt.ion(), plt.show()
    '''

    X_z = X / Z
    Y_z = Y / Z
    pixel_coords2 = torch.stack([X_z, Y_z], dim=2)  # [B, H*W, 2]

    return pixel_coords.reshape(bs, hh, ww, 2), Z.reshape(bs, 1, hh, ww), pixel_coords2.reshape(bs, hh, ww, 2)


# 这里的motion field 指的是运动物体的pose变化
def inverse_warp_mof(img, depth, ref_depth, motion_field, intrinsics, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, 1, H, W]
        ref_depth: the source depth map (where to sample depth) -- [B, 1, H, W]
        motion_field: 6DoF pose parameters from target to source -- [B, 6, H, W]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_mask: Float array indicating point validity
        projected_depth: sampled depth from source image
        computed_depth: computed depth of source image using the target depth
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'B1HW')
    check_sizes(ref_depth, 'ref_depth', 'B1HW')
    check_sizes(motion_field, 'motion_field', 'B6HW')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    bs, _, hh, ww = img.size()

    cam_coords = pixel2cam(depth.squeeze(1), intrinsics.inverse())  # [B,3,H,W]

    transform_field = pose_mof2mat(motion_field)  # [B, 3, 4, 256, 832]
    transform_field = transform_field.permute(0, 3, 4, 1, 2).reshape(bs, -1, 3, 4)  # [B, N, 3, 4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics.reshape(bs, 1, 3, 3) @ transform_field  # [B, N, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :, :3], proj_cam_to_src_pixel[:, :, :, -1:]
    rot = rot.reshape(bs, hh, ww, 3, 3).permute(0, 3, 4, 1, 2)  # [8, 3, 3, 256, 832]
    tr = tr.reshape(bs, hh, ww, 3, 1).permute(0, 3, 4, 1, 2)  # [8, 3, 1, 256, 832]

    # pdb.set_trace()
    '''
        plt.close('all')
        plt.figure(1); plt.imshow(rot[0,0,0].detach().cpu()); plt.colorbar(); plt.ion(); plt.show();
        plt.figure(2); plt.imshow(tr[0,0,0].detach().cpu()); plt.colorbar(); plt.ion(); plt.show();
    '''
    src_pixel_coords, computed_depth, flow_grid = cam2pixel_mof(cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    if np.array(torch.__version__[:3]).astype(float) >= 1.3:
        projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode, align_corners=False)
    else:
        projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(1).float()

    if np.array(torch.__version__[:3]).astype(float) >= 1.3:
        projected_depth = F.grid_sample(ref_depth, src_pixel_coords, padding_mode=padding_mode, align_corners=False)
    else:
        projected_depth = F.grid_sample(ref_depth, src_pixel_coords, padding_mode=padding_mode)

    mgrid_np = np.expand_dims(np.mgrid[0:ww, 0:hh].transpose(2, 1, 0).astype(np.float32), 0).repeat(bs, axis=0)
    mgrid = torch.from_numpy(mgrid_np).cuda()  # [B,H,W,2]
    flow = (flow_grid - mgrid).permute(0, 3, 1, 2)

    return projected_img, valid_mask, projected_depth, computed_depth, flow


def flow_warp(img, flo):
    '''
    Simple flow-guided warping operation with grid sampling interpolation.
    Args:
        img: b x c x h x w
        flo: b x c x h x w
    Returns:
        img_w: b x c x h x w
        valid: b x 1 x h x w
    '''
    bs, ch, gh, gw = img.size()
    mgrid_np = np.expand_dims(np.mgrid[0:gw, 0:gh].transpose(0, 2, 1).astype(np.float32), 0).repeat(bs, axis=0)
    mgrid = torch.from_numpy(mgrid_np).type_as(flo)
    grid = mgrid.add(flo).permute(0, 2, 3, 1)  # b x 2 x gh x gw

    grid[:, :, :, 0] = grid[:, :, :, 0].sub(gw / 2).div(gw / 2)
    grid[:, :, :, 1] = grid[:, :, :, 1].sub(gh / 2).div(gh / 2)

    if np.array(torch.__version__[:3]).astype(float) >= 1.3:
        img_w = F.grid_sample(img, grid, align_corners=True)
    else:
        img_w = F.grid_sample(img, grid)

    valid = (grid.abs().max(dim=-1)[0] <= 1).unsqueeze(1).float()  # b x 1 x h x w
    img_w[(valid == 0).repeat(1, ch, 1, 1)] = 0  # b x c x h x w

    return img_w, valid


# 这里inverse 和 forward方向得到的坐标应该是相同的
def dp2flow(depth, pose, intrinsics, rotation_mode, padding_mode=None):
    """
    Converts depth and pose parameters to rigid optical flow
    """
    check_sizes(depth, 'depth', 'BHW')
    if rotation_mode == 'euler':
        check_sizes(pose, 'pose', 'B6')
        pose_mat = pose_vec2mat(pose, rotation_mode)
    elif rotation_mode == 'translation':
        check_sizes(pose, 'pose', 'B34')
        pose_mat = pose
    check_sizes(intrinsics, 'intrinsics', 'B33')

    bs, h, w = depth.size()

    grid_x = Variable(torch.arange(0, w).view(1, 1, w).expand(1, h, w), requires_grad=False).type_as(depth).expand_as(depth)  # [bs, H, W]
    grid_y = Variable(torch.arange(0, h).view(1, h, 1).expand(1, h, w), requires_grad=False).type_as(depth).expand_as(depth)  # [bs, H, W]

    cam_coords = pixel2cam(depth, intrinsics.inverse())  # [B,3,H,W]
    # pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)  # [B, 3, 4]
    src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel[:,:,:3], proj_cam_to_src_pixel[:,:,-1:], padding_mode)  # [B,H,W,2]

    X = (w-1)*(src_pixel_coords[:, :, :, 0]/2.0 + 0.5) - grid_x
    Y = (h-1)*(src_pixel_coords[:, :, :, 1]/2.0 + 0.5) - grid_y

    return torch.stack((X, Y), dim=1)
