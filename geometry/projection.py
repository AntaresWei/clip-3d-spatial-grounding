import numpy as np


def backproject_pixel_to_camera(u, v, depth, intrinsic):
    """
    u, v: pixel coordinate
    depth: depth value in meters
    intrinsic: 4x4 or 3x3 intrinsic matrix
    """

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    return np.array([X, Y, Z, 1.0])


def camera_to_world(point_camera, pose):
    """
    pose: 4x4 camera-to-world matrix
    """
    return pose @ point_camera


def depth_to_world_points(depth, intrinsic, pose):
    """
    depth: HxW depth map in meters
    """
    H, W = depth.shape
    points_world = []

    for v in range(0, H, 4):      # 下采样加速
        for u in range(0, W, 4):

            z = depth[v, u]
            if z == 0:
                continue

            point_cam = backproject_pixel_to_camera(u, v, z, intrinsic)
            point_world = camera_to_world(point_cam, pose)

            points_world.append(point_world[:3])

    return np.array(points_world)