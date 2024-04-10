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

import json
import socket
from typing import Callable
import traceback

import torch

from modules.camera import MiniCam
from modules.scene import Scene

host = '127.0.0.1'
port = 6009

conn = None
addr = None

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def init(wish_host, wish_port):
    global host, port, listener
    host = wish_host
    port = wish_port
    listener.bind((host, port))
    listener.listen()
    listener.settimeout(0)


def try_connect():
    global conn, addr, listener
    try:
        conn, addr = listener.accept()
        print(f'\nConnected by {addr}')
        conn.settimeout(None)
    except Exception as inst:
        pass


def read():
    global conn
    messageLength = conn.recv(4)
    messageLength = int.from_bytes(messageLength, 'little')
    message = conn.recv(messageLength)
    return json.loads(message.decode('utf-8'))


def send(message_bytes, verify):
    global conn
    if message_bytes != None:
        conn.sendall(message_bytes)
    conn.sendall(len(verify).to_bytes(4, 'little'))
    conn.sendall(bytes(verify, 'ascii'))


def receive():
    message = read()
    width = message['resolution_x']
    height = message['resolution_y']

    if width != 0 and height != 0:
        try:
            do_training = bool(message['train'])
            fovy = message['fov_y']
            fovx = message['fov_x']
            znear = message['z_near']
            zfar = message['z_far']
            do_shs_python = bool(message['shs_python'])
            do_rot_scale_python = bool(message['rot_scale_python'])
            keep_alive = bool(message['keep_alive'])
            scaling_modifier = message['scaling_modifier']
            world_view_transform = torch.reshape(torch.tensor(message['view_matrix']), (4, 4)).cuda()
            world_view_transform[:,1] = -world_view_transform[:,1]
            world_view_transform[:,2] = -world_view_transform[:,2]
            full_proj_transform = torch.reshape(torch.tensor(message['view_projection_matrix']), (4, 4)).cuda()
            full_proj_transform[:,1] = -full_proj_transform[:,1]
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
        except Exception as e:
            print('')
            traceback.print_exc()
            raise e
        return custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier
    else:
        return [None] * 6


def handle(render_func:Callable, scene:Scene, steps:int):
    global conn, addr, listener
    if conn == None:
        try_connect()
    while conn != None:
        try:
            hp = scene.hp
            net_image_bytes = None
            custom_cam, do_training, hp.convert_SHs_python, hp.compute_cov3D_python, keep_alive, scaling_modifer = receive()
            if custom_cam != None:
                net_image = render_func(scene.gaussians, custom_cam, scene.background, scaling_modifer)['render']
                net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            send(net_image_bytes, hp.source_path)
            if do_training and (steps < int(hp.iterations) or not keep_alive):
                break
        except Exception as e:
            from traceback import print_exc
            print_exc()
            conn = None
