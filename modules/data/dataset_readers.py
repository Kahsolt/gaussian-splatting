#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import json
import glob
from pathlib import Path
from typing import List, Dict, NamedTuple, Callable, Any

import numpy as np
from numpy import ndarray
import pandas as pd
from PIL import Image
from plyfile import PlyData, PlyElement

from modules.data.colmap_loader import (
    read_extrinsics_text, 
    read_intrinsics_text, 
    read_extrinsics_binary, 
    read_intrinsics_binary, 
    read_points3D_binary, 
    read_points3D_text, 
    qvec2rotmat,
    Image as ImageInfo,
)
from modules.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from modules.utils.sh_utils import SH2RGB


class BasicPointCloud(NamedTuple):
    points: ndarray
    normals: ndarray
    colors: ndarray


class CameraInfo(NamedTuple):
    uid: int
    R: ndarray
    T: ndarray
    FovY: ndarray
    FovX: ndarray
    image: ndarray
    image_path: str
    image_name: str
    width: int
    height: int

    def to_json(self, id:int=None) -> dict:
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = self.R.transpose()
        Rt[:3, 3] = self.T
        Rt[3, 3] = 1.0
        W2C = np.linalg.inv(Rt)
        pos = W2C[:3, 3]
        rot = W2C[:3, :3]
        return {
            'id': id,
            'img_name': self.image_name,
            'width': self.width,
            'height': self.height,
            'position': pos.tolist(),
            'rotation': [x.tolist() for x in rot],
            'fy': fov2focal(self.FovY, self.height),
            'fx': fov2focal(self.FovX, self.width)
        }


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: List[CameraInfo]
    test_cameras: List[CameraInfo]
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info) -> Dict[str, Any]:
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {'translate': translate, 'radius': radius}


def fetchPly(path:str) -> BasicPointCloud:
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    return BasicPointCloud(positions, normals, colors)


def storePly(path:str, xyz:ndarray, rgb:ndarray):
    # Define the dtype for the structured array
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ]

    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    properties = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, properties))

    # Create the PlyData object and write to file
    elem = PlyElement.describe(elements, 'vertex')
    PlyData([elem]).write(path)


def readColmapCameras(cam_extrinsics:Dict[Any, ImageInfo], cam_intrinsics:Dict[Any, ImageInfo], images_folder:str) -> List[CameraInfo]:
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(f'Reading camera {idx+1}/{len(cam_extrinsics)}')
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=='SIMPLE_PINHOLE':
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=='PINHOLE':
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, 'Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!'

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split('.')[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid, R, T, FovY, FovX, image, image_path, image_name, width, height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapSceneInfo(path:str, images:str, eval:bool, llffhold:int=8) -> SceneInfo:
    path_base = os.path.join(path, 'sparse', '0')
    try:
        cameras_extrinsic_file = os.path.join(path_base, 'images.bin')
        cameras_intrinsic_file = os.path.join(path_base, 'cameras.bin')
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path_base, 'images.txt')
        cameras_intrinsic_file = os.path.join(path_base, 'cameras.txt')
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = 'images' if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos  = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos  = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path_base, 'points3D.ply')
    bin_path = os.path.join(path_base, 'points3D.bin')
    txt_path = os.path.join(path_base, 'points3D.txt')
    if not os.path.exists(ply_path):
        print('Converting point3d.bin to .ply, will happen only the first time you open the scene.')
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    return SceneInfo(pcd, train_cam_infos, test_cam_infos, nerf_normalization, ply_path)


def readColmapExtSceneInfo(path:str, images:str, eval:bool):
    path_base = os.path.join(path, 'dense', 'sparse')

    # Step 1. Load files in the tsv first (split to train and test later)
    tsv = glob.glob(os.path.join(path, '*.tsv'))[0]
    files = pd.read_csv(tsv, sep='\t')
    files = files[~files['id'].isnull()] # remove data without id
    files.reset_index(inplace=True, drop=True)

    # Step 2. load image paths
    # Attention! The 'id' column in the tsv is BROKEN, don't use it!!!!
    # Instead, read the id from images.bin using image file name!
    cam_extrinsics = read_extrinsics_binary(os.path.join(path_base, 'images.bin'))
    img_path_to_id = {}
    for v in cam_extrinsics.values():
        img_path_to_id[v.name] = v.id
    img_ids = []
    image_paths = []
    id_name_pairs = {}
    for filename in list(files['filename']):
        id_ = img_path_to_id[filename]
        image_paths += [filename]
        id_name_pairs[id_] = filename
        img_ids += [id_]

    # Step 3. Load camera intrinsics
    cam_intrinsics = read_intrinsics_binary(os.path.join(path_base, 'cameras.bin'))
    reading_dir = "images" if images == None else images
    cam_infos = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, 'dense', reading_dir))
    cam_infos = [c for c in cam_infos if str(c.image_name) + '.jpg' in image_paths]

    if eval:
        img_ids_train   = [id_ for i, id_ in enumerate(img_ids) if files.loc[i, 'split'] == 'train']
        img_ids_test    = [id_ for i, id_ in enumerate(img_ids) if files.loc[i, 'split'] == 'test']
        img_names_train = [name for id, name in id_name_pairs.items() if id in img_ids_train]
        img_names_test  = [name for id, name in id_name_pairs.items() if id in img_ids_test]  
        train_cam_infos = [c for c in cam_infos if str(c.image_name)+'.jpg' in img_names_train]
        test_cam_infos  = [c for c in cam_infos if str(c.image_name)+'.jpg' in img_names_test]
    else:
        train_cam_infos = cam_infos
        test_cam_infos  = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path_base, 'points3D.ply')
    bin_path = os.path.join(path_base, 'points3D.bin')
    txt_path = os.path.join(path_base, 'points3D.txt')
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    return SceneInfo(pcd, train_cam_infos, test_cam_infos, nerf_normalization, ply_path)


def readCamerasFromTransforms(path, transformsfile, white_background, extension='.png') -> List[CameraInfo]:
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents['camera_angle_x']

        frames = contents['frames']
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame['file_path'] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame['transform_matrix'])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert('RGBA'))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), mode='RGB')

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(idx, R, T, FovY, FovX, image, image_path, image_name, width=image.size[0], height=image.size[1]))
    return cam_infos


def readNerfSyntheticInfo(path:str, white_background:bool, eval:bool, extension:str='.png') -> SceneInfo:
    print('Reading Train Transforms')
    train_cam_infos = readCamerasFromTransforms(path, 'transforms_train.json', white_background, extension)
    print('Reading Test Transforms')
    test_cam_infos = readCamerasFromTransforms(path, 'transforms_test.json', white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, 'points3d.ply')
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f'Generating random point cloud ({num_pts})...')

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, normals=np.zeros((num_pts, 3)), colors=SH2RGB(shs))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    return SceneInfo(pcd, train_cam_infos, test_cam_infos, nerf_normalization, ply_path)


SCENE_DATA_LOADERS: Dict[str, Callable[..., SceneInfo]] = {
    'Colmap': readColmapSceneInfo,
    'ColmapExt': readColmapExtSceneInfo,
    'Blender': readNerfSyntheticInfo,
}
