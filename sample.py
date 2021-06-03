#!/usr/bin/python3

from pointsmap import VoxelGridMap, Points, invertTransform, depth2colormap
import numpy as np
import cv2
import time
import h5py
import glob

with h5py.File('./points_test.hdf5', 'r') as h5file:

    PCD_PATH = glob.glob('/workspace/*.pcd')
    K = np.array([[h5file['K/rgb/Fx'][()], 0., h5file['K/rgb/Cx'][()]], [0., h5file['K/rgb/Fy'][()], h5file['K/rgb/Cy'][()]], [0., 0., 1.]])

    vgm = VoxelGridMap()

    print(vgm.get_intrinsic())

    vgm.set_intrinsic(K)
    print(vgm.get_intrinsic())

    vgm.set_shape(h5file['data/0/rgb'].shape)
    print(vgm.get_shape())

    vgm.set_depth_range((0.0, 100.0))
    print(vgm.get_depth_range())

    time_start = time.time()
    # vgm.set_pointsmap(PCD_PATH)
    vgm.set_semanticmap(h5file['map/points'][()], h5file['map/semantic1d'][()])
    # vgm.set_pointsmap(h5file['map/points'][()])
    print('Set PointsMap : {0:.6F}'.format(time.time() - time_start) + ' [sec]')
    # print('x: min:', np.min(h5file['map/points'][()][:,0]), 'max:', np.max(h5file['map/points'][()][:,0]))
    # print('y: min:', np.min(h5file['map/points'][()][:,1]), 'max:', np.max(h5file['map/points'][()][:,1]))
    # print('z: min:', np.min(h5file['map/points'][()][:,2]), 'max:', np.max(h5file['map/points'][()][:,2]))

    print(vgm.get_semanticmap()[1].shape)

    #mat = np.array([[0., 1., 0., 1.],[-1., 0., 0., 0.5],[0., 0., 1., 0.],[0., 0., 0., 1.]], dtype=np.float64)
    #map_depth = vgm.create_depthmap(mat, transformMode.PARENT2CHILD)

    # print('PointsMap : Set')

    time_sum = 0.0

    for i in range(h5file['header/length'][()]):
        translation = h5file['data/' + str(i) + '/pose/rgb/translation'][()]
        quaternion = h5file['data/' + str(i) + '/pose/rgb/rotation'][()]
        time_start = time.time()
        map_depth = vgm.create_depthmap(translation=translation, quaternion=quaternion, filter_radius=0)
        time_create_depthmap = time.time() - time_start
        time_sum += time_create_depthmap
        print('\r{0:>4d} : Create DepthMap : {1:.6F}'.format(i, time_create_depthmap) + ' [sec]', end='')
        # print(map_depth)
        map_depth_nc = depth2colormap(map_depth, 0.0, 100.0, invert=True)

        cv2.imwrite('./test_%03d_depth_map.png'%(i), map_depth_nc)

        pts = Points()
        pts.set_semanticpoints(h5file['data/' + str(i) + '/semantic3d/points'][()], h5file['data/' + str(i) + '/semantic3d/semantic1d'][()])
        pts.set_intrinsic(K)
        pts.set_shape(h5file['data/0/rgb'].shape)
        pts.set_depth_range((0.0, 100.0))
        points_semantic2d = pts.create_semantic2d(translation=np.array([0.,0.,0.], dtype=np.float32), quaternion=np.array([0.5,-0.5,0.5,0.5], dtype=np.float32))
        points_semantic2d_c = np.zeros(h5file['data/0/rgb'].shape, dtype=np.uint8)
        for key, item in h5file['label/semantic3d'].items():
            points_semantic2d_c[np.where(points_semantic2d == int(key))] = item['color'][()]
        cv2.imwrite('./test_%03d_semantic2d_lidar.png'%(i), points_semantic2d_c)

        points_depth = pts.create_depthmap(translation=np.array([0.,0.,0.], dtype=np.float32), quaternion=np.array([0.5,-0.5,0.5,0.5], dtype=np.float32))
        points_depth_nc = depth2colormap(points_depth, 0.0, 100.0, invert=True)
        cv2.imwrite('./test_%03d_depth_points.png'%(i), points_depth_nc)

        map_semantic2d = vgm.create_semantic2d(translation=translation, quaternion=quaternion, filter_radius=0)
        map_semantic2d_c = np.zeros(h5file['data/0/rgb'].shape, dtype=np.uint8)
        for key, item in h5file['label/semantic2d'].items():
            map_semantic2d_c[np.where(map_semantic2d == int(key))] = item['color'][()]
        cv2.imwrite('./test_%03d_semantic2d_map.png'%(i), map_semantic2d_c)

        cv2.imwrite('./test_%03d_rgb.png'%(i), h5file['data/' + str(i) + '/rgb'][()])
        break

    print('\nAvg. : Create DepthMap : {0:.6F}'.format(time_sum / (i + 1)) + ' [sec]')