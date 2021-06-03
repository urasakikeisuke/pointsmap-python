# -*- coding: utf-8 -*-

from enum import IntEnum
from typing import List, Tuple, Union
import numpy as np
from pointsmap.libpointsmap import invert_transform, matrix_to_quaternion, quaternion_to_matrix, depth_to_colormap, combine_transforms, voxelgridmap, points


def invertTransform(translation: np.ndarray = None, quaternion: np.ndarray = None, matrix_4x4: np.ndarray = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
  """invertTransform

  並進ベクトルとクォータニオン, または変換行列を逆変換します.

  Args:
      translation (np.ndarray): 並進ベクトル [x y z]
      quaternion (np.ndarray): クォータニオン [x y z w]

      matrix_4x4 (np.ndarray): 4x4の変換行列
                              [[r11 r12 r13 tx]
                               [r21 r22 r23 ty]
                               [r31 r32 r33 tz]
                               [  0   0   0  1]]

  Returns:
      Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 逆変換した並進ベクトルとクォータニオンのタプル, または変換行列
  """
  if isinstance(translation, np.ndarray) and isinstance(quaternion, np.ndarray):
    return invert_transform(translation, quaternion)
  elif isinstance(matrix_4x4, np.ndarray):
    return invert_transform(matrix_4x4)
  else:
    raise AssertionError('Set the values for "translation" and "quaternion" or "matrix_4x4".')

def matrix2quaternion(matrix_4x4: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """matrix2quaternion

  変換行列を並進ベクトルとクォータニオンに変換する

  Args:
      matrix_4x4 (np.ndarray): 4x4の変換行列
                              [[r11 r12 r13 tx]
                               [r21 r22 r23 ty]
                               [r31 r32 r33 tz]
                               [  0   0   0  1]]

  Returns:
      Tuple[np.ndarray, np.ndarray]: 並進ベクトルとクォータニオンのタプル
  """
  return matrix_to_quaternion(matrix_4x4)

def quaternion2matrix(translation: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
  """quaternion2matrix

  並進ベクトルとクォータニオンを変換行列に変換する

  Args:
      translation (np.ndarray): 並進ベクトル [x y z]
      quaternion (np.ndarray): クォータニオン [x y z w]

  Returns:
      np.ndarray: 4x4の変換行列
  """
  return quaternion_to_matrix(translation, quaternion)

def depth2colormap(src: np.ndarray, min: float, max: float, type: int = 2, invert: bool = False) -> np.ndarray:
  """depth2colormap

  深度マップからカラーマップを生成する

  Args:
      src (np.ndarray): 深度マップ
      min (float): 深度の表示範囲 (最小値)
      max (float): 深度の表示範囲 (最大値)
      type (int, optional): cv2.ColormapTypes (既定値：cv2.COLORMAP_JET)
      invert (bool, optional): カラーマップの反転

  Returns:
      np.ndarray: カラーマップ
  """
  return depth_to_colormap(src, min, max, type, invert)

def combineTransforms(translations: List[np.ndarray] = None, quaternions: List[np.ndarray] = None, matrixes: List[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
  """combineTransforms

  複数の変換行列, または並進ベクトル・クォータニオンを合成する

  Args:
      translations (np.ndarray): 並進ベクトルのリスト [x y z]
      quaternions (np.ndarray): クォータニオンのリスト [x y z w]

      matrixs (np.ndarray): 4x4の変換行列のリスト
                              [[r11 r12 r13 tx]
                               [r21 r22 r23 ty]
                               [r31 r32 r33 tz]
                               [  0   0   0  1]]

  Returns:
      Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 合成した並進ベクトルとクォータニオンのタプル, または変換行列
  """
  if isinstance(translations, list) and isinstance(quaternions, list):
    return combine_transforms(translations, quaternions)
  elif isinstance(matrixes, list):
    return combine_transforms(matrixes)
  else:
    raise AssertionError('Set the values for "transforms" and "quaternions" or "matrixs".')

class Points():
  """Points

  三次元点群を扱うクラス. 大規模な三次元地図を扱う場合は, VoxelGridMapクラスの方が高速.

  Args:
      quiet (bool): `True`のとき, "ERROR", "WARNING"以外のメッセージをコンソールに表示しない.
  """
  def __init__(self, quiet: bool = False) -> None:
    """__init__

    Args:
        quiet (bool, optional): `True`のとき, "ERROR", "WARNING"以外のメッセージをコンソールに表示しない. Defaults to False.
    """
    self.instance = points(quiet)
  
  def set_points(self, obj: Union[str, List[str], np.ndarray]) -> None:
    """set_points

    三次元点群を読み込む.

    Args:
        obj (str): 三次元点群ファイル(.pcd)のパス
            (List[str]): 三次元点群ファイル(.pcd)のパスのリスト
            (np.ndarray): 三次元点群を格納したNumpy(N, 3)行列
    """
    self.instance.set_points(obj)
  
  def set_semanticpoints(self, points: np.ndarray, semantic1d: np.ndarray) -> None:
    """set_semanticpoints

    semantic3d 型の ラベル付き三次元点群を読み込みます.

    Args:
        points (np.ndarray): ラベル付き三次元点群を構成する点群を格納したNumpy(N, 3)行列
        semantic1d (np.ndarray): ラベル付き三次元点群を構成するラベルを格納したNumpy(N,)行列
    """
    self.instance.set_semanticpoints(points, semantic1d)
  
  def add_points(self, obj: Union[str, List[str], np.ndarray]) -> None:
    """add_points

    三次元点群を追加する.

    Args:
        obj (str): 三次元点群ファイル(.pcd)のパス
            (List[str]): 三次元点群ファイル(.pcd)のパスのリスト
            (np.ndarray): 三次元点群を格納したNumpy(N, 3)行列
    """
    self.instance.add_points(obj)
  
  def add_semanticpoints(self, points: np.ndarray, semantic1d: np.ndarray) -> None:
    """add_semanticpoints

    semantic3d 型の ラベル付き三次元点群を追加する.

    Args:
        points (np.ndarray): ラベル付き三次元点群を構成する点群を格納したNumpy(N, 3)行列
        semantic1d (np.ndarray): ラベル付き三次元点群を構成するラベルを格納したNumpy(N,)行列
    """
    self.instance.add_semanticpoints(points, semantic1d)
  
  def get_points(self) -> np.ndarray:
    """get_points

    三次元点群のデータを取得します.

    Returns:
        np.ndarray: 取得した三次元点群 (Numpy(N, 3)行列, points型)
    """
    return self.instance.get_points()

  def get_semanticpoints(self) -> Tuple[np.ndarray, np.ndarray]:
    """get_semanticpoints

    semantic3d 型のラベル付き三次元点群のデータを取得します.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ラベル付き三次元点群を構成する点群(Numpy(N, 3)行列, points型), ラベル(Numpy(N,)行列)のTuple
    """
    return self.instance.get_semanticpoints()

  def set_intrinsic(self, K: np.ndarray) -> None:
    """set_intrinsic

    カメラ内部パラメータを読み込みます.

    Args:
        K (np.ndarray): カメラパラメータを格納したNumpy(3, 3)行列
    """
    self.instance.set_intrinsic(K)
  
  def get_intrinsic(self) -> np.ndarray:
    """get_intrinsic

    読み込まれたカメラ内部パラメータを取得します.

    Returns:
        np.ndarray: カメラ内部パラメータ
    """
    return self.instance.get_intrinsic()
  
  def set_shape(self, shape: Tuple[int, ...]) -> None:
    """set_shape

    出力する画像サイズを設定する.

    Args:
        shape (tuple): 画像サイズ (H, W)
    """
    self.instance.set_shape(shape)
  
  def get_shape(self) -> Tuple[int, int]:
    """get_shape

    設定した画像サイズを読み出す.

    Returns:
        tuple: 画像サイズ (H, W)
    """
    return self.instance.get_shape()
  
  def set_depth_range(self, depth_range: Tuple[float, float]) -> None:
    """set_depth_range

    Args:
        depth_range (tuple): Depthの表示範囲 (MIN, MAX)
    """
    self.instance.set_depth_range(depth_range)
  
  def get_depth_range(self) -> Tuple[float, float]:
    """get_depth_range

    Returns:
        tuple: Depthの表示範囲 (MIN, MAX)
    """
    return self.instance.get_depth_range()

  def set_depthmap(self, depthmap: np.ndarray,
                    translation: np.ndarray = np.array([0., 0., 0.], dtype=np.float32), quaternion: np.ndarray = np.array([0., 0., 0., 1.], dtype=np.float32),
                    matrix_4x4: np.ndarray = None) -> None:
    """set_depthmap

    深度マップを点群に変換し, 並進ベクトルとクォータニオン, または変換行列で座標変換をして格納する.

    Args:
        depthmap (np.ndarray): 深度マップ

        translation (np.ndarray, optional): 並進ベクトル[m] [X, Y, Z]. Defaults to np.array([0., 0., 0.], dtype=np.float32).
        quaternion (np.ndarray, optional): クォータニオン [X, Y, Z, W]. Defaults to np.array([0., 0., 0., 1.], dtype=np.float32).

        matrix_4x4 (np.ndarray, optional): 4x4変換行列. Defaults to None.
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

    深度マップとSemantic2Dラベルを点群に変換し, 並進ベクトルとクォータニオン, または変換行列で座標変換をして格納する.

    Args:
        depthmap (np.ndarray): 深度マップ
        semantic2d (np.ndarray): Semantic2Dラベル

        translation (np.ndarray, optional): 並進ベクトル[m] [X, Y, Z]. Defaults to np.array([0., 0., 0.], dtype=np.float32).
        quaternion (np.ndarray, optional): クォータニオン [X, Y, Z, W]. Defaults to np.array([0., 0., 0., 1.], dtype=np.float32).

        matrix_4x4 (np.ndarray, optional): 4x4変換行列. Defaults to None.
    """
    if isinstance(matrix_4x4, np.ndarray):
      self.instance.set_depthmap_semantic2d(depthmap, semantic2d, matrix_4x4)
    elif isinstance(translation, np.ndarray) and isinstance(quaternion, np.ndarray):
      self.instance.set_depthmap_semantic2d(depthmap, semantic2d, translation, quaternion)
    else:
      raise AssertionError('Set the values for "translation" and "quaternion" or "matrix_4x4".')

  def transform(self, translation: np.ndarray = None, quaternion: np.ndarray = None, matrix_4x4: np.ndarray = None) -> None:
    """transform

    格納されている点群を座標変換する

    Args:
        translation (np.ndarray): 並進ベクトル [x y z]
        quaternion (np.ndarray): クォータニオン [x y z w]

        matrix_4x4 (np.ndarray): 変換行列 [[r11 r12 r13 tx] [r21 r22 r23 ty] [r31 r32 r33 tz] [  0   0   0  1]]
    """
    if isinstance(translation, np.ndarray) and isinstance(quaternion, np.ndarray):
      self.instance.transform(translation, quaternion)
    elif isinstance(matrix_4x4, np.ndarray):
      self.instance.transform(matrix_4x4)
    else:
      raise AssertionError('Set the values for "translation" and "quaternion" or "matrix_4x4".')
  
  def downsampling(self, leaf_size:float) -> None:
    """downsampling

    格納されている点群をVoxelGridFilterを用いてダウンサンプリングする

    Args:
        leaf_size (float): Leaf Size (>0)
    """
    if leaf_size <= 0.0:
      raise ValueError('"leaf_size" must be greater than 0.')
    self.instance.downsampling(leaf_size)

  def create_depthmap(self, translation: np.ndarray = None, quaternion: np.ndarray = None, matrix_4x4: np.ndarray = None,
                      filter_radius: int = 0, filter_threshold: float = 3.0) -> np.ndarray:
    """create_depthmap

    並進ベクトルとクォータニオン, または変換行列を用いて三次元点群から深度マップを生成する.

    Args:
        translation (np.ndarray): 並進ベクトル [x y z]
        quaternion (np.ndarray): クォータニオン [x y z w]

        matrix_4x4 (np.ndarray): 変換行列 [[r11 r12 r13 tx] [r21 r22 r23 ty] [r31 r32 r33 tz] [  0   0   0  1]]

        filter_radius (int, optional): Visibility Filterのカーネル半径. 0 の場合, フィルタ処理を行わない. Defaults to 0.
        filter_threshold (float, optional): Visibility Filterの閾値. Defaults to 3.0.

    Returns:
        np.ndarray: 深度マップ
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

    並進ベクトルとクォータニオン, または変換行列を用いてラベル付き三次元点群からSemantic2dラベルを生成する.

    Args:
        translation (np.ndarray): 並進ベクトル [x y z]
        quaternion (np.ndarray): クォータニオン [x y z w]

        matrix_4x4 (np.ndarray): 変換行列 [[r11 r12 r13 tx] [r21 r22 r23 ty] [r31 r32 r33 tz] [  0   0   0  1]]

        filter_radius (int, optional): Visibility Filterのカーネル半径. 0 の場合, フィルタ処理を行わない. Defaults to 0.
        filter_threshold (float, optional): Visibility Filterの閾値. Defaults to 3.0.

    Returns:
        np.ndarray: Semantic2dラベル
    """
    if isinstance(translation, np.ndarray) and isinstance(quaternion, np.ndarray):
      return self.instance.create_semantic2d(translation, quaternion, filter_radius, filter_threshold)
    elif isinstance(matrix_4x4, np.ndarray):
      return self.instance.create_semantic2d(matrix_4x4, filter_radius, filter_threshold)
    else:
      raise AssertionError('Set the values for "translation" and "quaternion" or "matrix_4x4".')

class VoxelGridMap():
  """VoxelGridMap

  三次元地図を扱うクラス. 小規模な三次元点群を扱う場合は, Pointsクラスを推奨.

  Args:
      quiet (bool): `True`のとき, "ERROR", "WARNING"以外のメッセージをコンソールに表示しない.
  """

  def __init__(self, quiet: bool = False) -> None:
    """__init__

    Args:
        quiet (bool, optional): `True`のとき, "ERROR", "WARNING"以外のメッセージをコンソールに表示しない. Defaults to False.
    """
    self.instance = voxelgridmap(quiet)
  
  def set_pointsmap(self, obj: Union[str, List[str], np.ndarray], voxel_size: float = 10.0) -> None:
    """set_pointsmap

    三次元地図を読み込む.

    Args:
        obj (str): 三次元地図ファイル(.pcd)のパス
            (List[str]): 三次元地図ファイル(.pcd)のパスのリスト
            (np.ndarray): 三次元地図を格納したNumpy(N, 3)行列
        voxel_size (float, optional): ボクセルのサイズ (初期値: 10.0)
    """
    self.instance.set_pointsmap(obj, voxel_size)

  def set_semanticmap(self, points: np.ndarray, semantic1d: np.ndarray, voxel_size: float = 10.0) -> None:
    """set_semanticmap

    semantic3d 型の Semantic Map を読み込みます.

    Args:
        points (np.ndarray): Semantic Map を構成する点群を格納したNumpy(N, 3)行列
        semantic1d (np.ndarray): Semantic Map のラベルを格納したNumpy(N,)行列
        voxel_size (float, optional): ボクセルのサイズ (初期値: 10.0)
    """
    self.instance.set_semanticmap(points, semantic1d, voxel_size)
  
  def set_voxelgridmap(self, vgm:np.ndarray, voxel_size:float, voxels_min:Tuple[float, float, float], voxels_max:Tuple[float, float, float], voxels_center:Tuple[float, float, float], voxels_origin:Tuple[int, int, int]) -> None:
    """set_voxelgridmap

    VoxelGridMap のデータを格納します.

    Args:
        vgm (np.ndarray): VoxelGridMap (compound(N,)['x', 'y', 'z', 'label']を格納したNumpy(Z, Y, X)行列)
        voxel_size (float): Voxelのサイズ[m]
        voxels_min (Tuple[float, float, float]): VoxelGridMapの範囲の最小値(z_min, y_min, x_min)
        voxels_max (Tuple[float, float, float]): VoxelGridMapの範囲の最大値(z_max, y_max, x_max)
        voxels_center (Tuple[float, float, float]): VoxelGridMapの中心座標(z_center, y_center, x_center)
        voxels_origin (Tuple[int, int, int]): VoxelGridMapの中心のVoxelのインデックス(z_origin, y_origin, x_origin)
    """
    self.instance.set_voxelgridmap(vgm, voxel_size, voxels_min, voxels_max, voxels_center, voxels_origin)
  
  def set_empty_voxelgridmap(self, voxels_len:Tuple[int, int, int], voxel_size:float, voxels_min:Tuple[float, float, float], voxels_max:Tuple[float, float, float], voxels_center:Tuple[float, float, float], voxels_origin:Tuple[int, int, int]) -> None:
    """set_empty_voxelgridmap

    空の VoxelGridMap のデータを格納します.

    Args:
        voxels_len (Tuple[float, float, float]): 各軸方向のGridの数 (z_len, y_len, x_len)
        voxel_size (float): Voxelのサイズ[m]
        voxels_min (Tuple[float, float, float]): VoxelGridMapの範囲の最小値(z_min, y_min, x_min)
        voxels_max (Tuple[float, float, float]): VoxelGridMapの範囲の最大値(z_max, y_max, x_max)
        voxels_center (Tuple[float, float, float]): VoxelGridMapの中心座標(z_center, y_center, x_center)
        voxels_origin (Tuple[int, int, int]): VoxelGridMapの中心のVoxelのインデックス(z_origin, y_origin, x_origin)
    """
    self.instance.set_empty_voxelgridmap(voxels_len, voxel_size, voxels_min, voxels_max, voxels_center, voxels_origin)

  def get_pointsmap(self) -> np.ndarray:
    """get_pointsmap

    三次元地図のデータを取得します.

    Returns:
        np.ndarray: 取得した三次元地図 (Numpy(N, 3)行列, points型)
    """
    return self.instance.get_pointsmap()
  
  def get_semanticmap(self) -> Tuple[np.ndarray, np.ndarray]:
    """get_semanticmap

    semantic3d 型の Semantic Map のデータを取得します.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Semantic Map を構成する点群(Numpy(N, 3)行列, points型), ラベル(Numpy(N,)行列)のTuple
    """
    return self.instance.get_semanticmap()
  
  def get_voxel_points(self) -> np.ndarray:
    """get_voxel_points

    voxel-points 型の Voxel Grid Map のデータを取得します.

    Returns:
        np.ndarray: VoxelGridMap (compound(N,)['x', 'y', 'z']を格納したNumpy(Z, Y, X)行列)
    """
    return self.instance.get_voxelgridmap(False)

  def get_voxel_semantic3d(self) -> np.ndarray:
    """get_voxel_semantic3d

    voxel-semantic3d 型の Voxel Grid Map のデータを取得します.

    Returns:
        np.ndarray: VoxelGridMap (compound(N,)['x', 'y', 'z', 'label']を格納したNumpy(Z, Y, X)行列)
    """
    return self.instance.get_voxelgridmap(True)

  def get_voxel_size(self) -> float:
    """get_voxel_size

    Voxel のサイズを取得します.

    Returns:
        float: Voxelのサイズ[m]
    """
    return self.instance.get_voxel_size()
  
  def get_voxels_min(self) -> Tuple[float, float, float]:
    """get_voxels_min

    VoxelGridMapの範囲の最小値を取得します.

    Returns:
        Tuple[float, float, float]: VoxelGridMapの範囲の最小値(z_min, y_min, x_min)
    """
    return self.instance.get_voxels_min()
  
  def get_voxels_max(self) -> Tuple[float, float, float]:
    """get_voxels_max

    VoxelGridMapの範囲の最大値を取得します.

    Returns:
        Tuple[float, float, float]: VoxelGridMapの範囲の最大値(z_max, y_max, x_max)
    """
    return self.instance.get_voxels_max()

  def get_voxels_center(self) -> Tuple[float, float, float]:
    """get_voxels_center

    VoxelGridMapの中心座標を取得します.

    Returns:
        Tuple[float, float, float]: VoxelGridMapの中心座標(z_center, y_center, x_center)
    """
    return self.instance.get_voxels_center()
  
  def get_voxels_origin(self) -> Tuple[int, int, int]:
    """get_voxels_origin

    VoxelGridMapの中心のVoxelのインデックスを取得します.

    Returns:
        Tuple[int, int, int]: VoxelGridMapの中心のVoxelのインデックス(z_origin, y_origin, x_origin)
    """
    return self.instance.get_voxels_origin()
  
  def get_voxels_include_frustum(self, translation: np.ndarray = None, quaternion: np.ndarray = None, matrix_4x4: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """get_voxels_include_frustum

    視錘台を含むVoxelのインデックスを取得します.

    Args:
        translation (np.ndarray): 並進ベクトル [x y z]
        quaternion (np.ndarray): クォータニオン [x y z w]

        matrix_4x4 (np.ndarray): 変換行列 [[r11 r12 r13 tx] [r21 r22 r23 ty] [r31 r32 r33 tz] [  0   0   0  1]]

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 視錘台を含むVoxelのインデックス (np.where() と同様の形式)
    """
    if isinstance(translation, np.ndarray) and isinstance(quaternion, np.ndarray):
      return self.instance.get_voxels_include_frustum(translation, quaternion)
    elif isinstance(matrix_4x4, np.ndarray):
      return self.instance.get_voxels_include_frustum(matrix_4x4)
    else:
      raise AssertionError('Set the values for "translation" and "quaternion" or "matrix_4x4".')

  def set_intrinsic(self, K: np.ndarray) -> None:
    """set_intrinsic

    カメラ内部パラメータを読み込みます.

    Args:
        K (np.ndarray): カメラパラメータを格納したNumpy(3, 3)行列
    """
    self.instance.set_intrinsic(K)
  
  def get_intrinsic(self) -> np.ndarray:
    """get_intrinsic

    読み込まれたカメラ内部パラメータを取得します.

    Returns:
        np.ndarray: カメラ内部パラメータ
    """
    return self.instance.get_intrinsic()
  
  def set_shape(self, shape: Tuple[int, ...]) -> None:
    """set_shape

    出力する画像サイズを設定する.

    Args:
        shape (tuple): 画像サイズ (H, W)
    """
    self.instance.set_shape(shape)
  
  def get_shape(self) -> Tuple[int, int]:
    """get_shape

    設定した画像サイズを読み出す.

    Returns:
        tuple: 画像サイズ (H, W)
    """
    return self.instance.get_shape()
  
  def set_depth_range(self, depth_range: Tuple[float, float]) -> None:
    """set_depth_range

    Args:
        depth_range (tuple): Depthの表示範囲 (MIN, MAX)
    """
    self.instance.set_depth_range(depth_range)
  
  def get_depth_range(self) -> Tuple[float, float]:
    """get_depth_range

    Returns:
        tuple: Depthの表示範囲 (MIN, MAX)
    """
    return self.instance.get_depth_range()

  def create_depthmap(self, translation: np.ndarray = None, quaternion: np.ndarray = None, matrix_4x4: np.ndarray = None,
                      filter_radius: int = 0, filter_threshold: float = 3.0) -> np.ndarray:
    """create_depthmap

    並進ベクトルとクォータニオン, または変換行列を用いて三次元地図から深度マップを生成する.

    Args:
        translation (np.ndarray): 並進ベクトル [x y z]
        quaternion (np.ndarray): クォータニオン [x y z w]

        matrix_4x4 (np.ndarray): 変換行列 [[r11 r12 r13 tx] [r21 r22 r23 ty] [r31 r32 r33 tz] [  0   0   0  1]]

        filter_radius (int, optional): Visibility Filterのカーネル半径. 0 の場合, フィルタ処理を行わない. Defaults to 0.
        filter_threshold (float, optional): Visibility Filterの閾値. Defaults to 3.0.

    Returns:
        np.ndarray: 深度マップ
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

    並進ベクトルとクォータニオン, または変換行列を用いてSemantic MapからSemantic2dラベルを生成する.

    Args:
        translation (np.ndarray): 並進ベクトル [x y z]
        quaternion (np.ndarray): クォータニオン [x y z w]

        matrix_4x4 (np.ndarray): 変換行列 [[r11 r12 r13 tx] [r21 r22 r23 ty] [r31 r32 r33 tz] [  0   0   0  1]]

        filter_radius (int, optional): Visibility Filterのカーネル半径. 0 の場合, フィルタ処理を行わない. Defaults to 0.
        filter_threshold (float, optional): Visibility Filterの閾値. Defaults to 3.0.

    Returns:
        np.ndarray: Semantic2dラベル
    """
    if isinstance(translation, np.ndarray) and isinstance(quaternion, np.ndarray):
      return self.instance.create_semantic2d(translation, quaternion, filter_radius, filter_threshold)
    elif isinstance(matrix_4x4, np.ndarray):
      return self.instance.create_semantic2d(matrix_4x4, filter_radius, filter_threshold)
    else:
      raise AssertionError('Set the values for "translation" and "quaternion" or "matrix_4x4".')
  
  def set_voxels(self, indexs: Tuple[np.ndarray, np.ndarray, np.ndarray], voxels: np.ndarray) -> None:
    """set_voxels

    指定したインデックスにVoxelを格納する

    Args:
        indexs (Tuple[np.ndarray, np.ndarray, np.ndarray]): Voxelのインデックスのタプル (z, y, x) (np.where() と同様の形式)
        voxels (np.ndarray): Voxels (compound(N,)['x', 'y', 'z', 'label']を格納したNumpy(Z, Y, X)行列)
    """
    self.instance.set_voxels(indexs, voxels)
