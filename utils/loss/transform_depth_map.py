#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from utils.loss import transform_utils
from utils.loss.consistency_loss import multiply_no_nan


#如何在不规则网格上（摄像机移动带来的）上转换深度图:保证像素的偏移在0,H-W之间，并对合理的像素偏移设置mask
class TransformedDepthMap(object):
    """A collection of tensors that described a transformed depth map.
    This class describes the result of a spatial transformation applied on a depth
    map. The initial depthmap was defined on a regular pixel grid. Knowing the
    camera intrinsics, each pixel can be mapped to a point in space.
    However once the camera or the scene has moved, when the points are projected
    back onto the camera, they don't fall on a regular pixel grid anymore. To
    obtain a new depth map on a regular pixel grid, one needs to resample, taking
    into account occlusions, and leaving gaps at areas that were occluded before
    the movement.
    This class described the transformed depth map on an IRREGULAR grid, before
    any resampling. The attributes are 4 tensors of shape [B, H, W]
    (batch, height, width): pixel_x, pixel_y, depth and mask.
    The given a triplet of indices, (b, i, j), the depth at the pixel location
    (pixel_y[b, i, j], pixel_x[b, i, j]) on the depth image is depth[b, i, j].
    As explained above, (pixel_y[b, i, j], pixel_x[b, i, j]) are not regular with
    respect to i and j. They are floating point numbers that generally fall in
    between pixels and can fall out of image boundaries (0, 0), (H - 1, W - 1).
    For all indices b, i, j where 0 <= pixel_y[b, i, j] <= H - 1 and
    0 <= pixel_x[b, i, j] < W - 1, mask[b, i, j] is True, otherwise it's False.
    For convenience, after we mark mask[b, i, j] as false for
    (pixel_y[b, i, j], pixel_x[b, i, j]) that are our of bounds, we clamp
    (pixel_y[b, i, j], pixel_x[b, i, j]) to be within the bounds. So, you're not
    supposed to look (pixel_y[b, i, j], pixel_x[b, i, j], depth[b, i, j]) where
    mask[b, i, j] is False, but if you do, you'll find that there were clamped
    to be within the bounds. The motivation for this is that if we later use
    pixel_x and pixel_y for warping, this clamping will result in extrapolating
    from the boundary by replicating the boundary value, which is reasonable.
    """

    def __init__(self, pixel_x, pixel_y, depth, mask):
        """Initializes an instance. The arguments is explained above."""
        self._pixel_x = pixel_x
        self._pixel_y = pixel_y
        self._depth = depth
        self._mask = mask
        attrs = sorted(self.__dict__.keys())
        # Unlike equality, compatibility is not transitive, so we have to check all
        # pairs.
        for i in range(len(attrs)):
            for j in range(i):
                tensor_i = self.__dict__[attrs[i]]
                tensor_j = self.__dict__[attrs[j]]
                if not (tensor_i.shape == tensor_j.shape):
                    raise ValueError(
                        'All tensors in TransformedDepthMap\'s constructor must have '
                        'compatible shapes, however \'%s\' and \'%s\' have the '
                        'incompatible shapes %s and %s.' %
                        (attrs[i][1:], attrs[j][1:], tensor_i.shape, tensor_j.shape))
        self._pixel_xy = None

    @property
    def pixel_x(self):
        return self._pixel_x

    @property
    def pixel_y(self):
        return self._pixel_y

    @property
    def depth(self):
        return self._depth

    @property
    def mask(self):
        return self._mask

    @property
    def pixel_xy(self):
        if self._pixel_xy is None:
            #name = self._pixel_x.op.name.rsplit('/', 1)[0]
            self._pixel_xy = torch.stack([self._pixel_x, self._pixel_y],
                                         dim=3)
        return self._pixel_xy


def using_transform_matrix(depth, transform, intrinsic_mat, name=None):
    """Transforms a depth map using a transform matrix.
    Args:
      depth: A torch.Tensor representing a depth map. Shape is [B, H, W].
      transform: A torch.Tensor representing a batch of transform matrices. Shape is
        [B, 4, 4]. The last row of each 4x4 is assumed (but not verified) to be
        (0, 0, 0, 1).
      intrinsic_mat: A torch.Tensor representing a batch of camera intrinsic
        matrices. Shape is [B, 3, 3].
      name: A string or None, a name scope for the ops.
    Returns:
      A TransformedDepthMap object.
    """
    pixel_x, pixel_y, z = _using_transform_matrix(depth, transform,
                                                  intrinsic_mat)
    pixel_x, pixel_y, mask = _clamp_and_filter_result(pixel_x, pixel_y, z)

    return TransformedDepthMap(pixel_x, pixel_y, z, mask)


def using_motion_vector(depth,
                        translation,
                        rotation_angles,
                        rot_mode,
                        intrinsic_mat,
                        intrinsic_mat_inv=None,
                        distortion_coeff=None,
                        name=None):
    """Transforms a depth map using a motion vector, or a motion vector field.
    This function receives a translation vector and rotation angles vector. They
    can be the same for the entire image, or different for each pixel.
    Args:
      depth: A torch.Tensor of shape [B, H, W]
      translation: A torch.Tensor of shape [B, 3] or [B, H, W, 3] representing a
        translation vector for the entire image or for every pixel respectively.
      rotation_angles: A torch.Tensor of shape [B, 3] or [B, H, W, 3] representing a
        set of rotation angles for the entire image or for every pixel  这里的rot可以是rot_field.
        respectively. We conform to the same convention as in inverse_warp above,
        but may need to reconsider, depending on the conventions tf.graphics and
        other users will converge to.
      rot_mode: 判断当前旋转向量的表示方式是欧拉角还是轴角
      intrinsic_mat: A torch.Tensor of shape [B, 3, 3].
      intrinsic_mat_inv: A torch.Tensor of shape [B, 3, 3], containing the inverse
        of intrinsic_mat. If None, intrinsic_mat_inv will be calculated from
        intrinsic_mat. Providing intrinsic_mat_inv directly is useful for TPU,
        where matrix inversion is not supported.
      distortion_coeff: A scalar (python or torch.Tensor) of a floating point type,
        or None, the quadratic radial distortion coefficient. If 0.0 or None, a
        distortion-less implementation (which is simpler and maybe faster) will be
        used.
      name: A string or None, a name scope for the ops.
    Returns:
      A TransformedDepthMap object.
    """
    if distortion_coeff is not None and distortion_coeff != 0.0:
        pixel_x, pixel_y, z = _using_motion_vector_with_distortion(
            depth, translation, rotation_angles, rot_mode, intrinsic_mat, distortion_coeff)
    else:
        pixel_x, pixel_y, z = _using_motion_vector(
            depth, translation, rotation_angles, rot_mode, intrinsic_mat, intrinsic_mat_inv)
    pixel_x, pixel_y, mask = _clamp_and_filter_result(pixel_x, pixel_y, z)
    return TransformedDepthMap(pixel_x, pixel_y, z, mask)


# 三维空间坐标转换到图像平面坐标系下
def _using_transform_matrix(depth,
                            transform,
                            intrinsic_mat,
                            intrinsic_mat_inv=None):
    """A helper for using_transform_matrix. See docstring therein."""
    _, height, width = torch.unbind(depth.shape)
    grid = torch.squeeze(
        torch.stack(torch.meshgrid(torch.range(end=width), torch.range(end=height), (1,))), dim=3)
    grid = grid.type(torch.FloatTensor)
    if intrinsic_mat_inv is None:
        intrinsic_mat_inv = torch.inverse(intrinsic_mat)
    cam_coords = torch.einsum('bij,jhw,bhw->bihw', intrinsic_mat_inv, grid, depth)

    rotation = transform[:, :3, :3]
    translation = transform[:, :3, 3]

    xyz = (
            torch.einsum('bij,bjk,bkhw->bihw', intrinsic_mat, rotation, cam_coords) +
            _expand_last_dim_twice(
                torch.einsum('bij,bj->bi', intrinsic_mat, translation)))
    x, y, z = torch.unbind(xyz, dim=1)
    pixel_x = x / z
    pixel_y = y / z
    return pixel_x, pixel_y, z


def _using_motion_vector(depth, translation, rotation_angles, rot_mode, intrinsic_mat,
                         intrinsic_mat_inv=None):
    """A helper for using_motion_vector. See docstring therein."""

    if translation.dim() not in (2, 4):
        raise ValueError('\'translation\' should have rank 2 or 4, not %d' %
                         translation.dim())
    if translation.shape[-1] != 3:
        raise ValueError('translation\'s last dimension should be 3, not %d' %
                         translation.shape[1])
    if translation.dim() == 2:
        translation = torch.unsqueeze(torch.unsqueeze(translation, 1), 1)
    _, height, width = depth.shape
    grid = torch.squeeze(
        torch.stack(torch.meshgrid(torch.arange(0, end=height, dtype=torch.float),
                                   torch.arange(0, end=width, dtype=torch.float),
                                   torch.tensor([1.0, ]))), dim=3)
    grid = grid.type(torch.FloatTensor).to(device=depth.device)

    if intrinsic_mat_inv is None:
        intrinsic_mat_inv = torch.inverse(intrinsic_mat).to(device=intrinsic_mat.device)
    # Use the depth map and the inverse intrinsic matrix to generate a point
    # cloud xyz.
    xyz = torch.einsum('bij,jhw,bhw->bihw', intrinsic_mat_inv, grid, depth)

    # TPU pads aggressively tensors that have small dimensions. Therefore having
    # A rotation of the shape [....., 3, 3] would overflow the HBM memory. To
    # address this, we represnet the rotations is a 3x3 nested python tuple of
    # torch.Tensors (that is, we unroll the rotation matrix at the small dimensions).
    # The 3x3 matrix multiplication is now done in a python loop, and tensors with
    # small dimensions are avoided.
    unstacked_xyz = torch.unbind(xyz, dim=1)  # unstack是分解的意思，unbind函数可以将xzy的一维分解穿成多个张量
    #unstacked_rotation_matrix = transform_utils.unstacked_matrix_from_angles(*torch.unbind(rotation_angles, dim=-1), rot_mode)
    unstacked_rotation_matrix = transform_utils.unstacked_matrix_from_angles(rotation_angles, rot_mode)
    rank_diff = (
            unstacked_xyz[0].dim() -
            unstacked_rotation_matrix[0][0].dim())

    def expand_to_needed_rank(t):
        for _ in range(rank_diff):
            t = torch.unsqueeze(t, -1)
        return t

    unstacked_rotated_xyz = [0.0] * 3
    for i in range(3):
        for j in range(3):
            unstacked_rotated_xyz[i] += expand_to_needed_rank(
                unstacked_rotation_matrix[i][j]) * unstacked_xyz[j]
    rotated_xyz = torch.stack(unstacked_rotated_xyz, dim=1).to(device=intrinsic_mat.device)

    # Project the transformed point cloud back to the camera plane.
    pcoords = torch.einsum('bij,bjhw->bihw', intrinsic_mat, rotated_xyz)

    projected_translation = torch.einsum('bij,bhwj->bihw', intrinsic_mat,
                                         translation)

    pcoords = pcoords + projected_translation
    x, y, z = torch.unbind(pcoords, dim=1)
    return x / z, y / z, z


def _using_motion_vector_with_distortion(depth,
                                         translation,
                                         rotation_angles,
                                         rot_mode,  # 当前的旋转向量是轴角还是欧拉角
                                         intrinsic_mat,
                                         distortion_coeff=0.0):  # 畸变系数
    """A helper for using_motion_vector. See docstring therein."""

    if translation.shape.ndims not in (2, 4):
        raise ValueError('\'translation\' should have rank 2 or 4, not %d' %
                         translation.shape.ndims)
    if translation.shape[-1] != 3:
        raise ValueError('translation\'s last dimension should be 3, not %d' %
                         translation.shape[1])
    if translation.shape.ndims == 2:
        translation = torch.unsqueeze(torch.unsqueeze(translation, 1), 1)

    _, height, width = torch.unbind(depth.shape)
    grid = torch.squeeze(
        torch.stack(torch.meshgrid(torch.arange(0, end=height, dtype=torch.float),
                                   torch.arange(0, end=width, dtype=torch.float),
                                   torch.tensor([1.0, ]))), dim=3)
    grid = grid.type(torch.FloatTensor)
    intrinsic_mat_inv = torch.inverse(intrinsic_mat).to(device=intrinsic_mat.device)

    normalized_grid = torch.einsum('bij,jhw->bihw', intrinsic_mat_inv, grid)
    radii_squared = torch.sum(torch.square(normalized_grid[:, :2, :, :]), dim=1)

    undistortion_factor = quadratic_inverse_distortion_scale(
        distortion_coeff, radii_squared)
    undistortion_factor = torch.stack([
        undistortion_factor, undistortion_factor,
        torch.ones_like(undistortion_factor)
    ],
        dim=1)
    normalized_grid *= undistortion_factor

    rot_mat = transform_utils.matrix_from_angles(rotation_angles, rot_mode)
    # We have to treat separately the case of a per-image rotation vector and a
    # per-image rotation field, because the broadcasting capabilities of einsum
    # are limited.
    # 计算三维图像的点云坐标
    if rotation_angles.shape.ndims == 2:  # 这里angle为(b,3)
        # 使用PyTorch的einsum函数对三维张量进行矩阵乘法。结果的维度是（batch_size，input_height，input_width）
        pcoords = torch.einsum('bij,bjhw,bhw->bihw', rot_mat, normalized_grid, depth)
    elif rotation_angles.shape.ndims == 4:  # 这里的angle维度是转换成了有h和w的即四维，得到的rot_mat就是五维
        # （batch_size，input_height，input_width，3），就需要把H和W维度移到最后，并且把旋转矩阵元素转置，像上面一样
        rot_mat = torch.transpose(rot_mat, [0, 3, 4, 1, 2])
        pcoords = torch.einsum('bijhw,bjhw,bhw->bihw', rot_mat, normalized_grid, depth)

    pcoords += torch.transpose(translation, [0, 3, 1, 2])

    x, y, z = torch.unbind(pcoords, dim=1)
    x = x / z
    y = y / z
    scale = quadraric_distortion_scale(distortion_coeff,
                                       torch.square(x) + torch.square(y))
    x = x * scale
    y = y * scale

    pcoords = torch.einsum('bij,bjhw->bihw', intrinsic_mat,
                        torch.stack([x, y, torch.ones_like(x)], dim=1))
    x, y, _ = torch.unbind(pcoords, dim=1)

    return x, y, z


# 对比二维图像上是否出现过界的位置
def _clamp_and_filter_result(pixel_x, pixel_y, z):
    """Clamps and masks out out-of-bounds pixel coordinates.
    Args:
      pixel_x: a torch.Tensor containing x pixel coordinates in an image.
      pixel_y: a torch.Tensor containing y pixel coordinates in an image.
      z: a torch.Tensor containing the depth ar each (pixel_y, pixel_x)  All shapes
        are [B, H, W].
    Returns:
      pixel_x, pixel_y, mask, where pixel_x and pixel_y are the original ones,
      except:
      - Values that fall out of the image bounds, which are [0, W-1) in x and
        [0, H-1) in y, are clamped to the bounds
      - NaN values in pixel_x, pixel_y are replaced by zeros
      mask is False at allpoints where:
      - Clamping in pixel_x or pixel_y was performed
      - NaNs were replaced by zeros
      - z is non-positive,
      and True everywhere else, that is, where pixel_x, pixel_y are finite and
      fall within the frame.
    """
    _, height, width = pixel_x.shape

    def _tensor(x):
        return torch.FloatTensor(x)

    #将坐标x, y在x>=0、y>=0、x<(W-1)、y<(H-1)、z>0和x和y坐标不是NaN时都取True
    x_not_underflow = pixel_x >= 0.0
    y_not_underflow = pixel_y >= 0.0
    x_not_overflow = pixel_x < width - 1
    y_not_overflow = pixel_y < height - 1
    z_positive = z > 0.0
    x_not_nan = torch.logical_not(torch.isnan(pixel_x))
    y_not_nan = torch.logical_not(torch.isnan(pixel_y))
    not_nan = torch.logical_and(x_not_nan, y_not_nan)
    not_nan_mask = not_nan.type(torch.FloatTensor).to(device=pixel_x.device)
    #在这些True的像素点处保留原始坐标值，否则将这些像素点的坐标值变为0
    pixel_x = multiply_no_nan(pixel_x, not_nan_mask)  # TODO
    pixel_y = multiply_no_nan(pixel_y, not_nan_mask)  # TODO
    pixel_x = torch.clamp(pixel_x, 0.0, width - 1)
    pixel_y = torch.clamp(pixel_y, 0.0, height - 1)
    #返回一个掩膜，True表示当前像素点在图像中，False表示像素点超出了图像范围或者深度为0或者像素坐标是NaN
    mask_stack = torch.stack([
        x_not_underflow, y_not_underflow, x_not_overflow, y_not_overflow,
        z_positive, not_nan
    ], axis=0)
    mask = torch.all(mask_stack, dim=0)  # 这里得到的mask是转换后超出图像范围的位置
    return pixel_x, pixel_y, mask


def quadraric_distortion_scale(distortion_coefficient, r_squared):
    """Calculates a quadratic distortion factor given squared radii.
    The distortion factor is 1.0 + `distortion_coefficient` * `r_squared`. When
    `distortion_coefficient` is negative (barrel distortion), the distorted radius
    is only monotonically increasing only when
    `r_squared` < r_squared_max = -1 / (3 * distortion_coefficient).
    Args:
      distortion_coefficient: A torch.Tensor of a floating point type. The rank can
        be from zero (scalar) to r_squared's rank. The shape of
        distortion_coefficient will be appended by ones until the rank equals that
        of r_squared.
      r_squared: A torch.Tensor of a floating point type, containing
        (x/z)^2 + (y/z)^2. We use r_squared rather than r to avoid an unnecessary
        sqrt, which may introduce gradient singularities. The non-negativity of
        r_squared only enforced in debug mode.
    Returns:
      A torch.Tensor of r_squared's shape, the correction factor that should
      multiply the projective coordinates (x/z) and (y/z) to apply the
      distortion.
    """
    return 1 + distortion_coefficient * r_squared


def quadratic_inverse_distortion_scale(distortion_coefficient,
                                       distorted_r_squared,
                                       newton_iterations=4):
    """Calculates the inverse quadratic distortion function given squared radii.
    The distortion factor is 1.0 + `distortion_coefficient` * `r_squared`. When
    `distortion_coefficient` is negative (barrel distortion), the distorted radius
    is monotonically increasing only when
    r < r_max = sqrt(-1 / (3 * distortion_coefficient)).
    max_distorted_r_squared is obtained by calculating the distorted_r_squared
    corresponding to r = r_max, and the result is
    max_distorted_r_squared = - 4 / (27.0 * distortion_coefficient)
    Args:
    distortion_coefficient: A torch.Tensor of a floating point type. The rank can
      be from zero (scalar) to r_squared's rank. The shape of
      distortion_coefficient will be appended by ones until the rank equals that
      of r_squared.
    distorted_r_squared: A torch.Tensor of a floating point type, containing
      (x/z)^2 + (y/z)^2. We use distorted_r_squared rather than distorted_r to
      avoid an unnecessary sqrt, which may introduce gradient singularities.
      The non-negativity of distorted_r_squared is only enforced in debug mode.
    newton_iterations: Number of Newton-Raphson iterations to calculate the
      inverse distprtion function. Defaults to 5, which is on the high-accuracy
      side.
    Returns:
    A torch.Tensor of distorted_r_squared's shape, containing the correction
    factor that should multiply the distorted the projective coordinates (x/z)
    and (y/z) to obtain the undistorted ones.
    """
    c = 1.0  # c for Correction
    # Newton-Raphson iterations for solving the inverse function of the
    # distortion.
    for _ in range(newton_iterations):
        c = (1.0 -
             (2.0 / 3.0) * c) / (1.0 + 3 * distortion_coefficient *
                                 distorted_r_squared * c * c) + (2.0 / 3.0) * c
    return c


def _expand_last_dim_twice(x):
    return torch.unsqueeze(torch.unsqueeze(x, -1), -1)