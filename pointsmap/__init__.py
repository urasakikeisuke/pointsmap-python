# -*- coding: utf-8 -*-

from typing import List, Tuple, Union
import numpy as np
from pointsmap.libpointsmap import invert_transform, matrix_to_quaternion, quaternion_to_matrix, depth_to_colormap, combine_transforms, voxelgridmap, points, depth


def invertTransform(translation: np.ndarray = None, quaternion: np.ndarray = None, matrix_4x4: np.ndarray = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
  """invertTransform

  Invert a translation vector and quaternion, or a transformation matrix.

  Args:
      translation (np.ndarray): translation vector [x y z]
      quaternion (np.ndarray): quaternion [x y z w]

      matrix_4x4 (np.ndarray): transformation matrix
                              [[r11 r12 r13 tx]
                               [r21 r22 r23 ty]
                               [r31 r32 r33 tz]
                               [  0   0   0  1]]

  Returns:
      Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: a tuple of inverse transformed translation vector and quaternion, or a transformation matrix.
  """
  if isinstance(translation, np.ndarray) and isinstance(quaternion, np.ndarray):
    return invert_transform(translation, quaternion)
  elif isinstance(matrix_4x4, np.ndarray):
    return invert_transform(matrix_4x4)
  else:
    raise AssertionError('Set the values for "translation" and "quaternion" or "matrix_4x4".')

def matrix2quaternion(matrix_4x4: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """matrix2quaternion

  Convert the transformation matrix to a translation vector and a quaternion.

  Args:
      matrix_4x4 (np.ndarray): transformation matrix
                              [[r11 r12 r13 tx]
                               [r21 r22 r23 ty]
                               [r31 r32 r33 tz]
                               [  0   0   0  1]]

  Returns:
      Tuple[np.ndarray, np.ndarray]: a tuple of inverse transformed translation vector and quaternion.
  """
  return matrix_to_quaternion(matrix_4x4)

def quaternion2matrix(translation: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
  """quaternion2matrix

  Convert a translation vector and a quaternion to a transformation matrix.

  Args:
      translation (np.ndarray): translation vector [x y z]
      quaternion (np.ndarray): quaternion [x y z w]

  Returns:
      np.ndarray: transformation matrix
  """
  return quaternion_to_matrix(translation, quaternion)

def depth2colormap(src: np.ndarray, min: float, max: float, type: int = 2, invert: bool = False) -> np.ndarray:
  """depth2colormap

  Generate a color map from a depth map.

  Args:
      src (np.ndarray): depth map.
      min (float): Display range of depth (min)
      max (float): Display range of depth (max)
      type (int, optional): cv2.ColormapTypes (Default: cv2.COLORMAP_JET)
      invert (bool, optional): Inverting a color map.

  Returns:
      np.ndarray: a color map.
  """
  return depth_to_colormap(src, min, max, type, invert)

def combineTransforms(translations: List[np.ndarray] = None, quaternions: List[np.ndarray] = None, matrixes: List[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
  """combineTransforms

  Combine multiple transformation matrices, or translation vectors and quaternions.

  Args:
      translations (np.ndarray): list of translation vectors [x y z]
      quaternions (np.ndarray): list of quaternions [x y z w]

      matrixs (np.ndarray): list of transformation matrixes.
                              [[r11 r12 r13 tx]
                               [r21 r22 r23 ty]
                               [r31 r32 r33 tz]
                               [  0   0   0  1]]

  Returns:
      Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: a tuple of combined translation vector and quaternion, or a transformation matrix.
  """
  if isinstance(translations, list) and isinstance(quaternions, list):
    return combine_transforms(translations, quaternions)
  elif isinstance(matrixes, list):
    return combine_transforms(matrixes)
  else:
    raise AssertionError('Set the values for "transforms" and "quaternions" or "matrixs".')

class Depth():
  """Depth

  Class for handling depth map.
  """
  def __init__(self) -> None:
    """__init__
    """
    self.instance = depth()

  def set_intrinsic(self, K: np.ndarray) -> None:
    """set_intrinsic

    Load the intrinsic parameters of the camera.

    Args:
        K (np.ndarray): Numpy(3, 3) matrix containing the camera intrinsic parameters.
    """
    self.instance.set_intrinsic(K)

  def get_intrinsic(self) -> np.ndarray:
    """get_intrinsic

    Get the intrinsic parameters of the camera.

    Returns:
        np.ndarray: Numpy(3, 3) matrix containing the camera intrinsic parameters.
    """
    return self.instance.get_intrinsic()

  def set_shape(self, shape: Tuple[int, ...]) -> None:
    """set_shape

    Set the output image size.

    Args:
        shape (tuple): image size (H, W)
    """
    self.instance.set_shape(shape)

  def get_shape(self) -> Tuple[int, int]:
    """get_shape

    Get the output image size.

    Returns:
        tuple: image size (H, W)
    """
    return self.instance.get_shape()

  def set_depth_range(self, depth_range: Tuple[float, float]) -> None:
    """set_depth_range

    Set the display range of the depth map.

    Args:
        depth_range (tuple): display range of the depth map (MIN, MAX)
    """
    self.instance.set_depth_range(depth_range)

  def get_depth_range(self) -> Tuple[float, float]:
    """get_depth_range

    Get the display range of the depth map.

    Returns:
        tuple: display range of the depth map (MIN, MAX)
    """
    return self.instance.get_depth_range()

  def set_base_line(self, base_line: float) -> None:
    """set_base_line

    Set the baseline of the stereo camera.

    Args:
        base_line (float): baseline
    """
    self.instance.set_base_line(base_line)

  def get_base_line(self) -> float:
    """get_base_line

    Get the baseline of the stereo camera.

    Returns:
        float: baseline
    """
    return self.instance.get_base_line()

  def set_depthmap(self, depthmap: np.ndarray) -> None:
    """set_depthmap

    Set a depth map.

    Args:
        depthmap (np.ndarray): Depth map.
    """
    self.instance.set_depthmap(depthmap)

  def set_disparity(self, disparity: np.ndarray) -> None:
    """set_disparity

    Set a disparity map.

    Args:
        disparity (np.ndarray): Disparity map.
    """
    self.instance.set_disparity(disparity)

  def get_depthmap(self) -> np.ndarray:
    """get_depthmap

    Get a depth map.

    Returns:
        np.ndarray: Depth map.
    """
    return self.instance.get_depthmap()

class Points():
  """Points

  Class for handling 3D point clouds. The VoxelGridMap class is faster than this class for large scale 3D points cloud maps.

  Args:
      quiet (bool): If `True`, do not display messages other than "ERROR" and "WARNING" on the console.
  """
  def __init__(self, quiet: bool = False) -> None:
    """__init__

    Args:
        quiet (bool, optional): If `True`, do not display messages other than "ERROR" and "WARNING" on the console. Defaults to False.
    """
    self.instance = points(quiet)

  def set_points(self, obj: Union[str, List[str], np.ndarray]) -> None:
    """set_points

    Load a 3D points cloud.

    Args:
        obj (str): path of the 3D points cloud file (.pcd).
            (List[str]): list of paths of 3D point cloud files (.pcd).
            (np.ndarray): a Numpy(N, 3) matrix containing the 3D points cloud.
    """
    self.instance.set_points(obj)

  def set_semanticpoints(self, points: np.ndarray, semantic1d: np.ndarray) -> None:
    """set_semanticpoints

    Loads a labeled 3D points cloud of type 'semantic3d'.

    Args:
        points (np.ndarray): a Numpy(N, 3) matrix containing the point clouds that make up the labeled 3D points cloud.
        semantic1d (np.ndarray): a Numpy(N,) matrix containing the labels that make up the labeled 3D points cloud.
    """
    self.instance.set_semanticpoints(points, semantic1d)

  def add_points(self, obj: Union[str, List[str], np.ndarray]) -> None:
    """add_points

    Add a 3D points cloud.

    Args:
        obj (str): path of the 3D points cloud file (.pcd).
            (List[str]): list of paths of 3D point cloud files (.pcd).
            (np.ndarray): a Numpy(N, 3) matrix containing the 3D points cloud.
    """
    self.instance.add_points(obj)

  def add_semanticpoints(self, points: np.ndarray, semantic1d: np.ndarray) -> None:
    """add_semanticpoints

    Add a labeled 3D points cloud of type 'semantic3d'.

    Args:
        points (np.ndarray): a Numpy(N, 3) matrix containing the point clouds that make up the labeled 3D points cloud.
        semantic1d (np.ndarray): a Numpy(N,) matrix containing the labels that make up the labeled 3D points cloud.
    """
    self.instance.add_semanticpoints(points, semantic1d)

  def get_points(self) -> np.ndarray:
    """get_points

    Get 3D points cloud.

    Returns:
        np.ndarray: Obtained 3D points cloud (Numpy(N, 3) matrix, 'points' type)
    """
    return self.instance.get_points()

  def get_semanticpoints(self) -> Tuple[np.ndarray, np.ndarray]:
    """get_semanticpoints

    Get a labeled 3D points cloud of type 'semantic3d'.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of points (Numpy(N, 3) matrix, 'points' type) and labels (Numpy(N,) matrix, 'semantic1d' type) constituting a labeled 3D points cloud.
    """
    return self.instance.get_semanticpoints()

  def save_pcd(self, path: str) -> None:
    """save_pcd

    Save points as PCD file.

    Args:
        path (str): Output path.
    """
    self.instance.save_pcd(path)

  def set_intrinsic(self, K: np.ndarray) -> None:
    """set_intrinsic

    Load the intrinsic parameters of the camera.

    Args:
        K (np.ndarray): Numpy(3, 3) matrix containing the camera intrinsic parameters.
    """
    self.instance.set_intrinsic(K)

  def get_intrinsic(self) -> np.ndarray:
    """get_intrinsic

    Get the intrinsic parameters of the camera.

    Returns:
        np.ndarray: Numpy(3, 3) matrix containing the camera intrinsic parameters.
    """
    return self.instance.get_intrinsic()

  def set_shape(self, shape: Tuple[int, ...]) -> None:
    """set_shape

    Set the output image size.

    Args:
        shape (tuple): image size (H, W)
    """
    self.instance.set_shape(shape)

  def get_shape(self) -> Tuple[int, int]:
    """get_shape

    Get the output image size.

    Returns:
        tuple: image size (H, W)
    """
    return self.instance.get_shape()

  def set_depth_range(self, depth_range: Tuple[float, float]) -> None:
    """set_depth_range

    Set the display range of the depth map.

    Args:
        depth_range (tuple): display range of the depth map (MIN, MAX)
    """
    self.instance.set_depth_range(depth_range)

  def get_depth_range(self) -> Tuple[float, float]:
    """get_depth_range

    Get the display range of the depth map.

    Returns:
        tuple: display range of the depth map (MIN, MAX)
    """
    return self.instance.get_depth_range()

  def set_depthmap(self, depthmap: np.ndarray,
                    translation: np.ndarray = np.array([0., 0., 0.], dtype=np.float32), quaternion: np.ndarray = np.array([0., 0., 0., 1.], dtype=np.float32),
                    matrix_4x4: np.ndarray = None) -> None:
    """set_depthmap

    The depth map is transformed into a point cloud, and the coordinates are transformed by a translation vector and a quaternion or a transformation matrix, and stored.

    Args:
        depthmap (np.ndarray): depth map[m]

        translation (np.ndarray, optional): a translation vector[m] [X, Y, Z]. Defaults to np.array([0., 0., 0.], dtype=np.float32).
        quaternion (np.ndarray, optional): a quaternion [X, Y, Z, W]. Defaults to np.array([0., 0., 0., 1.], dtype=np.float32).

        matrix_4x4 (np.ndarray, optional): a transformation matrix. Defaults to None.
    """
    if isinstance(matrix_4x4, np.ndarray):
      self.instance.set_depthmap(depthmap, matrix_4x4)
    elif isinstance(translation, np.ndarray) and isinstance(quaternion, np.ndarray):
      self.instance.set_depthmap(depthmap, translation, quaternion)
    else:
      raise AssertionError('Set the values for "translation" and "quaternion" or "matrix_4x4".')

  def set_depthmap_semantic2d(self, depthmap: np.ndarray, semantic2d: np.ndarray,
                    translation: np.ndarray = np.array([0., 0., 0.], dtype=np.float32), quaternion: np.ndarray = np.array([0., 0., 0., 1.], dtype=np.float32),
                    matrix_4x4: np.ndarray = None) -> None:
    """set_depthmap_semantic2d

    The depth map and the 'semantic2d' label are transformed into a point cloud, and the coordinates are transformed by a translation vector and a quaternion or a transformation matrix, and stored.

    Args:
        depthmap (np.ndarray): depth map[m]
        semantic2d (np.ndarray): 'semantic2d' label

        translation (np.ndarray, optional): a translation vector[m] [X, Y, Z]. Defaults to np.array([0., 0., 0.], dtype=np.float32).
        quaternion (np.ndarray, optional): a quaternion [X, Y, Z, W]. Defaults to np.array([0., 0., 0., 1.], dtype=np.float32).

        matrix_4x4 (np.ndarray, optional): a transformation matrix. Defaults to None.
    """
    if isinstance(matrix_4x4, np.ndarray):
      self.instance.set_depthmap_semantic2d(depthmap, semantic2d, matrix_4x4)
    elif isinstance(translation, np.ndarray) and isinstance(quaternion, np.ndarray):
      self.instance.set_depthmap_semantic2d(depthmap, semantic2d, translation, quaternion)
    else:
      raise AssertionError('Set the values for "translation" and "quaternion" or "matrix_4x4".')

  def transform(self, translation: np.ndarray = None, quaternion: np.ndarray = None, matrix_4x4: np.ndarray = None) -> None:
    """transform

    Converts the coordinates of the stored point cloud.

    Args:
        translation (np.ndarray, optional): a translation vector[m] [X, Y, Z]. Defaults to None.
        quaternion (np.ndarray, optional): a quaternion [X, Y, Z, W]. Defaults to None.

        matrix_4x4 (np.ndarray, optional): a transformation matrix. Defaults to None.
    """
    if isinstance(translation, np.ndarray) and isinstance(quaternion, np.ndarray):
      self.instance.transform(translation, quaternion)
    elif isinstance(matrix_4x4, np.ndarray):
      self.instance.transform(matrix_4x4)
    else:
      raise AssertionError('Set the values for "translation" and "quaternion" or "matrix_4x4".')

  def downsampling(self, leaf_size:float) -> None:
    """downsampling

    Downsampling the stored point cloud using VoxelGridFilter.

    Args:
        leaf_size (float): Leaf Size (>0)
    """
    if leaf_size <= 0.0:
      raise ValueError('"leaf_size" must be greater than 0.')
    self.instance.downsampling(leaf_size)

  def create_depthmap(self, translation: np.ndarray = None, quaternion: np.ndarray = None, matrix_4x4: np.ndarray = None,
                      filter_radius: int = 0, filter_threshold: float = 3.0) -> np.ndarray:
    """create_depthmap

    Generate a depth map from the stored 3D points cloud using a translation vector and a quaternion or a transformation matrix.

    Args:
        translation (np.ndarray): translation vector [x y z]
        quaternion (np.ndarray): quaternion [x y z w]

        matrix_4x4 (np.ndarray): transformation matrix
                              [[r11 r12 r13 tx]
                               [r21 r22 r23 ty]
                               [r31 r32 r33 tz]
                               [  0   0   0  1]]

        filter_radius (int, optional): Kernel radius of Visibility Filter. When 0, no filter is applied. Defaults to 0.
        filter_threshold (float, optional): Threshold of Visibility Filter. Defaults to 3.0.

    Returns:
        np.ndarray: a depth map
    """
    if isinstance(translation, np.ndarray) and isinstance(quaternion, np.ndarray):
      return self.instance.create_depthmap(translation, quaternion, filter_radius, filter_threshold)
    elif isinstance(matrix_4x4, np.ndarray):
      return self.instance.create_depthmap(matrix_4x4, filter_radius, filter_threshold)
    else:
      raise AssertionError('Set the values for "translation" and "quaternion" or "matrix_4x4".')

  def create_semantic2d(self, translation: np.ndarray = None, quaternion: np.ndarray = None, matrix_4x4: np.ndarray = None,
                        filter_radius: int = 0, filter_threshold: float = 3.0) -> np.ndarray:
    """create_semantic2d

    Generate a 'semantic2d' label from the stored labeled 3D points cloud using a translation vector and a quarterion, or a transformation matrix.

    Args:
        translation (np.ndarray): translation vector [x y z]
        quaternion (np.ndarray): quaternion [x y z w]

        matrix_4x4 (np.ndarray): transformation matrix
                                [[r11 r12 r13 tx]
                                 [r21 r22 r23 ty]
                                 [r31 r32 r33 tz]
                                 [  0   0   0  1]]

        filter_radius (int, optional): Kernel radius of Visibility Filter. When 0, no filter is applied. Defaults to 0.
        filter_threshold (float, optional): Threshold of Visibility Filter. Defaults to 3.0.

    Returns:
        np.ndarray: a 'semantic2d' label
    """
    if isinstance(translation, np.ndarray) and isinstance(quaternion, np.ndarray):
      return self.instance.create_semantic2d(translation, quaternion, filter_radius, filter_threshold)
    elif isinstance(matrix_4x4, np.ndarray):
      return self.instance.create_semantic2d(matrix_4x4, filter_radius, filter_threshold)
    else:
      raise AssertionError('Set the values for "translation" and "quaternion" or "matrix_4x4".')

class VoxelGridMap():
  """VoxelGridMap

  Class for handling 3D points cloud maps. For small-scale 3D points clouds, 'Points' class is recommended.

  Args:
      quiet (bool): If `True`, do not display messages other than "ERROR" and "WARNING" on the console.
  """

  def __init__(self, quiet: bool = False) -> None:
    """__init__

    Args:
        quiet (bool, optional): If `True`, do not display messages other than "ERROR" and "WARNING" on the console. Defaults to False.
    """
    self.instance = voxelgridmap(quiet)

  def set_pointsmap(self, obj: Union[str, List[str], np.ndarray], voxel_size: float = 10.0) -> None:
    """set_pointsmap

    Load a 3D points cloud map.

    Args:
        obj (str): path of the 3D points cloud file (.pcd).
            (List[str]): list of paths of 3D point cloud files (.pcd).
            (np.ndarray): a Numpy(N, 3) matrix containing the 3D points cloud.
        voxel_size (float, optional): size of voxels [m] (default: 10.0)
    """
    self.instance.set_pointsmap(obj, voxel_size)

  def set_semanticmap(self, points: np.ndarray, semantic1d: np.ndarray, voxel_size: float = 10.0) -> None:
    """set_semanticmap

    Loads a labeled 3D points cloud map of type 'semantic3d'.

    Args:
        points (np.ndarray): a Numpy(N, 3) matrix containing the point clouds that make up the labeled 3D points cloud.
        semantic1d (np.ndarray): a Numpy(N,) matrix containing the labels that make up the labeled 3D points cloud.
        voxel_size (float, optional): size of voxels [m] (default: 10.0)
    """
    self.instance.set_semanticmap(points, semantic1d, voxel_size)

  def set_voxelgridmap(self, vgm:np.ndarray, voxel_size:float, voxels_min:Tuple[float, float, float], voxels_max:Tuple[float, float, float], voxels_center:Tuple[float, float, float], voxels_origin:Tuple[int, int, int]) -> None:
    """set_voxelgridmap

    Store a VoxelGridMap.

    Args:
        vgm (np.ndarray): VoxelGridMap (Numpy(Z, Y, X) matrix containing compound(N,)['x', 'y', 'z', 'label'])
        voxel_size (float, optional): size of voxels [m]
        voxels_min (Tuple[float, float, float]): Minimum values of range for VoxelGridMap(z_min, y_min, x_min)
        voxels_max (Tuple[float, float, float]): Maximum values of range for VoxelGridMap(z_max, y_max, x_max)
        voxels_center (Tuple[float, float, float]): Center coordinates of VoxelGridMap(z_center, y_center, x_center)
        voxels_origin (Tuple[int, int, int]): Indexes of the center Voxel in the VoxelGridMap(z_origin, y_origin, x_origin)
    """
    self.instance.set_voxelgridmap(vgm, voxel_size, voxels_min, voxels_max, voxels_center, voxels_origin)

  def set_empty_voxelgridmap(self, voxels_len:Tuple[int, int, int], voxel_size:float, voxels_min:Tuple[float, float, float], voxels_max:Tuple[float, float, float], voxels_center:Tuple[float, float, float], voxels_origin:Tuple[int, int, int]) -> None:
    """set_empty_voxelgridmap

    Store an empty VoxelGridMap.

    Args:
        voxels_len (Tuple[float, float, float]): Number of Grids in each axial direction (z_len, y_len, x_len)
        voxel_size (float, optional): size of voxels [m]
        voxels_min (Tuple[float, float, float]): Minimum values of range for VoxelGridMap(z_min, y_min, x_min)
        voxels_max (Tuple[float, float, float]): Maximum values of range for VoxelGridMap(z_max, y_max, x_max)
        voxels_center (Tuple[float, float, float]): Center coordinates of VoxelGridMap(z_center, y_center, x_center)
        voxels_origin (Tuple[int, int, int]): Indexes of the center voxel in the VoxelGridMap(z_origin, y_origin, x_origin)
    """
    self.instance.set_empty_voxelgridmap(voxels_len, voxel_size, voxels_min, voxels_max, voxels_center, voxels_origin)

  def get_pointsmap(self) -> np.ndarray:
    """get_pointsmap

    Get 3D points cloud map.

    Returns:
        np.ndarray: Obtained 3D points cloud map (Numpy(N, 3) matrix, 'points' type)
    """
    return self.instance.get_pointsmap()

  def get_semanticmap(self) -> Tuple[np.ndarray, np.ndarray]:
    """get_semanticmap

    Get a labeled 3D points cloud map of type 'semantic3d'.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of points (Numpy(N, 3) matrix, 'points' type) and labels (Numpy(N,) matrix, 'semantic1d' type) constituting a labeled 3D points cloud map.
    """
    return self.instance.get_semanticmap()

  def get_voxel_points(self) -> np.ndarray:
    """get_voxel_points

    Get a VoxelGridMap of type 'voxel-points'.

    Returns:
        np.ndarray: VoxelGridMap (Numpy(Z, Y, X) matrix containing compound(N,)['x', 'y', 'z'])
    """
    return self.instance.get_voxelgridmap(False)

  def get_voxel_semantic3d(self) -> np.ndarray:
    """get_voxel_semantic3d

    Get a VoxelGridMap of type 'voxel-semantic3d'.

    Returns:
        np.ndarray: VoxelGridMap (Numpy(Z, Y, X) matrix containing compound(N,)['x', 'y', 'z', 'label'])
    """
    return self.instance.get_voxelgridmap(True)

  def get_voxel_size(self) -> float:
    """get_voxel_size

    Get the size of the voxels.

    Returns:
        float: the size of the voxels [m].
    """
    return self.instance.get_voxel_size()

  def get_voxels_min(self) -> Tuple[float, float, float]:
    """get_voxels_min

    Get the minimum values of range for VoxelGridMap.

    Returns:
        Tuple[float, float, float]: the minimum values of range for VoxelGridMap (z_min, y_min, x_min).
    """
    return self.instance.get_voxels_min()

  def get_voxels_max(self) -> Tuple[float, float, float]:
    """get_voxels_max

    Get the maximum values of range for VoxelGridMap.

    Returns:
        Tuple[float, float, float]: the maximum values of range for VoxelGridMap (z_max, y_max, x_max)
    """
    return self.instance.get_voxels_max()

  def get_voxels_center(self) -> Tuple[float, float, float]:
    """get_voxels_center

    Get the center coordinates of VoxelGridMap.

    Returns:
        Tuple[float, float, float]: the center coordinates of VoxelGridMap (z_center, y_center, x_center).
    """
    return self.instance.get_voxels_center()

  def get_voxels_origin(self) -> Tuple[int, int, int]:
    """get_voxels_origin

    Get the indexes of the center voxel in the VoxelGridMap

    Returns:
        Tuple[int, int, int]: the indexes of the center voxel in the VoxelGridMap (z_origin, y_origin, x_origin)
    """
    return self.instance.get_voxels_origin()

  def get_voxels_include_frustum(self, translation: np.ndarray = None, quaternion: np.ndarray = None, matrix_4x4: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """get_voxels_include_frustum

    Get the indexes of the voxels containing the frustum.

    Args:
        translation (np.ndarray): translation vector [x y z]
        quaternion (np.ndarray): quaternion [x y z w]

        matrix_4x4 (np.ndarray): transformation matrix
                                [[r11 r12 r13 tx]
                                 [r21 r22 r23 ty]
                                 [r31 r32 r33 tz]
                                 [  0   0   0  1]]

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: the indexes of the voxels containing the frustum. (Same format as np.where())
    """
    if isinstance(translation, np.ndarray) and isinstance(quaternion, np.ndarray):
      return self.instance.get_voxels_include_frustum(translation, quaternion)
    elif isinstance(matrix_4x4, np.ndarray):
      return self.instance.get_voxels_include_frustum(matrix_4x4)
    else:
      raise AssertionError('Set the values for "translation" and "quaternion" or "matrix_4x4".')

  def set_intrinsic(self, K: np.ndarray) -> None:
    """set_intrinsic

    Load the intrinsic parameters of the camera.

    Args:
        K (np.ndarray): Numpy(3, 3) matrix containing the camera intrinsic parameters.
    """
    self.instance.set_intrinsic(K)

  def get_intrinsic(self) -> np.ndarray:
    """get_intrinsic

    Get the intrinsic parameters of the camera.

    Returns:
        np.ndarray: Numpy(3, 3) matrix containing the camera intrinsic parameters.
    """
    return self.instance.get_intrinsic()

  def set_shape(self, shape: Tuple[int, ...]) -> None:
    """set_shape

    Set the output image size.

    Args:
        shape (tuple): image size (H, W)
    """
    self.instance.set_shape(shape)

  def get_shape(self) -> Tuple[int, int]:
    """get_shape

    Get the output image size.

    Returns:
        tuple: image size (H, W)
    """
    return self.instance.get_shape()

  def set_depth_range(self, depth_range: Tuple[float, float]) -> None:
    """set_depth_range

    Set the display range of the depth map.

    Args:
        depth_range (tuple): display range of the depth map (MIN, MAX)
    """
    self.instance.set_depth_range(depth_range)

  def get_depth_range(self) -> Tuple[float, float]:
    """get_depth_range

    Get the display range of the depth map.

    Returns:
        tuple: display range of the depth map (MIN, MAX)
    """
    return self.instance.get_depth_range()

  def create_depthmap(self, translation: np.ndarray = None, quaternion: np.ndarray = None, matrix_4x4: np.ndarray = None,
                      filter_radius: int = 0, filter_threshold: float = 3.0) -> np.ndarray:
    """create_depthmap

    Generate a depth map from the stored 3D points cloud map using a translation vector and a quaternion or a transformation matrix.

    Args:
        translation (np.ndarray): translation vector [x y z]
        quaternion (np.ndarray): quaternion [x y z w]

        matrix_4x4 (np.ndarray): transformation matrix
                              [[r11 r12 r13 tx]
                               [r21 r22 r23 ty]
                               [r31 r32 r33 tz]
                               [  0   0   0  1]]

        filter_radius (int, optional): Kernel radius of Visibility Filter. When 0, no filter is applied. Defaults to 0.
        filter_threshold (float, optional): Threshold of Visibility Filter. Defaults to 3.0.

    Returns:
        np.ndarray: a depth map
    """
    if isinstance(translation, np.ndarray) and isinstance(quaternion, np.ndarray):
      return self.instance.create_depthmap(translation, quaternion, filter_radius, filter_threshold)
    elif isinstance(matrix_4x4, np.ndarray):
      return self.instance.create_depthmap(matrix_4x4, filter_radius, filter_threshold)
    else:
      raise AssertionError('Set the values for "translation" and "quaternion" or "matrix_4x4".')

  def create_semantic2d(self, translation: np.ndarray = None, quaternion: np.ndarray = None, matrix_4x4: np.ndarray = None,
                        filter_radius: int = 0, filter_threshold: float = 3.0) -> np.ndarray:
    """create_semantic2d

    Generate a 'semantic2d' label from the stored labeled 3D points cloud map using a translation vector and a quarterion, or a transformation matrix.

    Args:
        translation (np.ndarray): translation vector [x y z]
        quaternion (np.ndarray): quaternion [x y z w]

        matrix_4x4 (np.ndarray): transformation matrix
                                [[r11 r12 r13 tx]
                                 [r21 r22 r23 ty]
                                 [r31 r32 r33 tz]
                                 [  0   0   0  1]]

        filter_radius (int, optional): Kernel radius of Visibility Filter. When 0, no filter is applied. Defaults to 0.
        filter_threshold (float, optional): Threshold of Visibility Filter. Defaults to 3.0.

    Returns:
        np.ndarray: a 'semantic2d' label
    """
    if isinstance(translation, np.ndarray) and isinstance(quaternion, np.ndarray):
      return self.instance.create_semantic2d(translation, quaternion, filter_radius, filter_threshold)
    elif isinstance(matrix_4x4, np.ndarray):
      return self.instance.create_semantic2d(matrix_4x4, filter_radius, filter_threshold)
    else:
      raise AssertionError('Set the values for "translation" and "quaternion" or "matrix_4x4".')

  def set_voxels(self, indexs: Tuple[np.ndarray, np.ndarray, np.ndarray], voxels: np.ndarray) -> None:
    """set_voxels

    Store voxels at the specified indexes.

    Args:
        indexs (Tuple[np.ndarray, np.ndarray, np.ndarray]): the indexes of the voxels. (Same format as np.where())
        voxels (np.ndarray): voxels (Numpy(Z, Y, X) matrix containing compound(N,)['x', 'y', 'z', 'label'])
    """
    self.instance.set_voxels(indexs, voxels)
