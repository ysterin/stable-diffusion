import numpy as np
from math import atan2, asin, cos
import cv2


def P2sRt(P):
    """ decompositing camera matrix P.
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation.
    """
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d


def matrix2angle(R):
    """ compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    refined by: https://stackoverflow.com/questions/43364900/rotation-matrix-to-euler-angles-with-opencv
     Args:
         R: (3,3). rotation matrix
     Returns:
         x: yaw
         y: pitch
         z: roll
     """
    if R[2, 0] > 0.998:
        z = 0
        x = np.pi / 2
        y = z + atan2(-R[0, 1], -R[0, 2])
    elif R[2, 0] < -0.998:
        z = 0
        x = -np.pi / 2
        y = -z + atan2(R[0, 1], R[0, 2])
    else:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

    return x, y, z


def calc_pose(param):
    P = param[:12].reshape(3, -1)  # camera matrix
    s, R, t3d = P2sRt(P)
    P = np.concatenate((R, t3d.reshape(3, -1)), axis=1)  # without scale
    pose = matrix2angle(R)
    pose = np.array(pose)

    return R.reshape(-1), t3d.reshape(-1), pose, s


def parse_param(param):
    """matrix pose form
    param: shape=(trans_dim+shape_dim+exp_dim,), i.e., 62 = 12 + 40 + 10
    """

    # pre-defined templates for parameter
    n = param.shape[0]
    if n == 62:
        trans_dim, shape_dim, exp_dim = 12, 40, 10
    elif n == 72:
        trans_dim, shape_dim, exp_dim = 12, 40, 20
    elif n == 141:
        trans_dim, shape_dim, exp_dim = 12, 100, 29
    else:
        raise Exception(f'Undefined templated param parsing rule')

    R_ = param[:trans_dim].reshape(3, -1)
    R, offset, pose, scale = calc_pose(R_)
    # R = R_[:, :3]
    # offset = R_[:, -1].reshape(3, 1)
    alpha_shp = param[trans_dim:trans_dim + shape_dim].reshape(-1)
    alpha_exp = param[trans_dim + shape_dim:].reshape(-1)

    return R, offset, pose, scale, alpha_shp, alpha_exp


def transform_scale_from_roi_box(roi_box, size):
    sx, sy, ex, ey = roi_box
    scale_x = (ex - sx) / size
    scale_y = (ey - sy) / size
    scale_z = (scale_x + scale_y) / 2
    return scale_x, scale_y, scale_z


def center_and_scale_params_3d(landmark: np.ndarray,
                               target_scale: float,
                               target_center_x: float,
                               target_center_y: float,
                               right_boundary_face_index: int = 0,
                               left_boundary_face_index: int = 16):
    center = ((landmark.min(0) + landmark.max(0)) / 2)
    size = np.linalg.norm(landmark[right_boundary_face_index, :] - landmark[left_boundary_face_index, :])
    center[1] -= size // 6

    transform_scale = target_scale / size
    diff_x = target_center_x - center[0] * transform_scale
    diff_y = target_center_y - center[1] * transform_scale
    return np.float32(diff_x), np.float32(diff_y), np.float32(transform_scale)


def transform_points(orig_pnts_2d: np.ndarray,
                     tx: float,
                     ty: float,
                     scale_x: float,
                     scale_y: float):
    pnts_2d = orig_pnts_2d.copy()
    pnts_2d[:, 0] *= scale_x
    pnts_2d[:, 0] += tx

    pnts_2d[:, 1] *= scale_y
    pnts_2d[:, 1] += ty
    return pnts_2d


def preprocess_3dmm(face_kp, center_x, center_y, target_scale):
    # face_kp = extract_landmark(tddfa_model=tddfa_model,
    #                            param_lst=face_pca,
    #                            roi_box_lst=face_pca_roi_bbox)
    # R, offset, pose, scale, alpha_shp, alpha_exp = parse_param(face_pca)
    # scale = np.array([scale])
    # face_roi = np.array(face_pca_roi_bbox)
    # tdmm_to_img_tform_scale = np.array(transform_scale_from_roi_box(face_roi, size=target_scale))
    diff_x, diff_y, transform_scale = center_and_scale_params_3d(face_kp,
                                                                 target_scale=target_scale,
                                                                 target_center_x=center_x,
                                                                 target_center_y=center_y)
    # offset_transformed = offset * tdmm_to_img_tform_scale * transform_scale
    # alpha_shp = alpha_shp / 1000
    # tdmm_parsed = [R, offset, offset_transformed, pose, scale, alpha_shp, alpha_exp]

    def make_transform_vertices(diff_x, diff_y, transform_scale):
        def transform_vertices(ver):
            transformed_ver = transform_points(ver.T,
                                               tx=diff_x,
                                               ty=diff_y,
                                               scale_x=transform_scale,
                                               scale_y=transform_scale).T
            return transformed_ver

        return transform_vertices

    transformation_vertices_func = make_transform_vertices(diff_x, diff_y, transform_scale)
    # image_pncc = render_pncc(tddfa_model=tddfa_model,
    #                          param_lst=face_pca,
    #                          roi_box_lst=face_pca_roi_bbox,
    #                          target_width=dst_width,
    #                          target_height=dst_height,
    #                          transformation_func=transformation_vertices_func)
    transformed_landmark = transform_points(face_kp,
                                            tx=diff_x,
                                            ty=diff_y,
                                            scale_x=transform_scale,
                                            scale_y=transform_scale)
    return diff_x, diff_y, transform_scale, transformation_vertices_func, transformed_landmark


def tensorflow_transform(img, tx, ty, scale_x, scale_y, dst_width, dst_height, interpolation='bilinear'):
    """
    [scale_x,  0,   offset_x,
     0,       scale_y   offset_y,
     0, 0]
    [a0,      a1, a2,       b0, b1,       b2,       c0,   c1]
    """
    T_pos1000 = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]])
    # rotate - opposite angle

    T_scale = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]])

    T = T_pos1000 @ T_scale
    # T_inv = np.linalg.inv(T)
    # T_inv_8 = T_inv.flatten()[:8]

    img_trans = cv2.warpAffine(img, T[:2], (dst_width, dst_height))

    return img_trans, T


def get_3ddfa_face_crop(face_kp, img, target_scale=210, target_center_y=0.5, target_center_x=0.5, dst_width=512, dst_height=512):
    # target_scale = 60.0 * 3.5  # 2
    target_center_x *= dst_width
    target_center_y *= dst_height
    target_scale *= (dst_width / 512)
    # target_center_y = 127.5 * 2
    # dst_width = 256 * 2
    # dst_height = 256 * 2

    diff_x, diff_y, transform_scale, transformation_vertices_func, \
        transformed_landmark = preprocess_3dmm(
            face_kp, target_center_x, target_center_y, target_scale)

    transformed_crop, T = tensorflow_transform(img=img,
                                               tx=diff_x,
                                               ty=diff_y,
                                               scale_x=transform_scale,
                                               scale_y=transform_scale,
                                               dst_width=dst_width,
                                               dst_height=dst_height
                                               )

    return transformed_crop, T
