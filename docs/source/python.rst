======
Python
======

定数
====

.. _transformmode:

TransformMode
-------------

* ``TransformMode.CHILD2PARENT = 0``
* ``TransformMode.PARENT2CHILD = 1``

関数
====

invertTransform
---------------

.. code-block:: python

  def invertTransform(
      translation: numpy.ndarray = None,
      quaternion: numpy.ndarray = None,
      matrix_4x4: numpy.ndarray = None
  ) -> tuple or numpy.ndarray:

同次変換行列, または並進ベクトル・クォータニオンを逆変換する.

* Args:

  * ``translation (numpy.ndarray)``: 並進ベクトル [x y z]
  * ``quaternion (numpy.ndarray)``: クォータニオン [x y z w]

    または

  * ``matrix_4x4 (numpy.ndarray)``: 4x4の同次変換行列

* Returns:

  * ``numpy.ndarray``: 逆変換行列

    または

  * ``tuple``: 逆変換した並進ベクトル・クォータニオン (translation, quaternion)

* 実装例:

  .. code-block:: python

    import numpy as np
    from pointsmap import *

    matrix = numpy.array([ [ 0., 1., 0., 1.],
                        [-1., 0., 0., 2.],
                        [ 0., 0., 1., 0.],
                        [ 0., 0., 0., 1.]])
    print(invertTransform(matrix_4x4=matrix))

    translation = numpy.array([74.63491058, 195.49958801, 2.00133348])
    quaternion = numpy.array([0.49215598, -0.50772443, 0.50772268, -0.49215452])
    print(invertTransform(translation=translation, quaternion=quaternion))

* 出力例:

  .. code-block::

    array([[ 0., -1.,  0.,  2.],
           [ 1.,  0.,  0., -1.],
           [ 0.,  0.,  1., -0.],
           [ 0.,  0.,  0.,  1.]])
    (array([ 197.72840881,    2.00151062,  -68.51225281]), array([-0.49215597,  0.5077244 , -0.50772268, -0.49215451]))

depth2colormap
--------------

.. code-block:: python

  def depth2colormap(
      src: numpy.ndarray,
      min: float, max: float,
      type: int = 2, invert: bool = False
  ) -> numpy.ndarray:

深度マップからカラーマップを生成する.

* Args:

  * ``src (numpy.ndarray)``: 深度マップ
  * ``min (float)``: 深度の表示範囲 (最小値)
  * ``max (float)``: 深度の表示範囲 (最大値)
  * ``type (int, optional)``: cv2.ColormapTypes (既定値: ``cv2.COLORMAP_JET``)
  * ``invert (bool, optional)``: カラーマップを反転する

* Returns:

  * ``numpy.ndarray``: カラーマップ

* 実装例:

  .. code-block:: python

    import h5py
    import cv2
    from pointsmap import *

    with h5py.File('sample.hdf5', 'r') as h5file:
      depth = h5file['data/0/depth'][()]
      color = depth2colormap(depth, 0.0, 100.0, type=cv2.COLORMAP_JET, invert=True)
      cv2.imwrite('sample.png', color)

combineTransforms
-----------------

.. code-block:: python

  def combineTransforms(
    translations: List[numpy.ndarray] = None,
    quaternions: List[numpy.ndarray] = None,
    matrixes: List[numpy.ndarray] = None
  ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:

複数の変換行列, または並進ベクトル・クォータニオンを合成する.

* Args:

  * ``translation (numpy.ndarray)``: 並進ベクトル [x y z] のリスト
  * ``quaternion (numpy.ndarray)``: クォータニオン [x y z w] のリスト

    または

  * ``matrix_4x4 (numpy.ndarray)``: 4x4の同次変換行列のリスト

* Returns:

  * ``numpy.ndarray``: 合成した変換行列

    または

  * ``tuple``: 合成した並進ベクトル・クォータニオン (translation, quaternion)

クラス
======

.. _points_class:

Points
------

.. code-block:: python

  from pointsmap import Points
  pts = Points(quiet: bool = False)

三次元点群を扱うクラス.
大規模な三次元点群地図を扱う場合は, :ref:`vgm_class` クラスの方が高速.

* Args:

  * ``quiet (bool, optional)``: ``True`` の場合, "ERROR", "WARNING"以外のメッセージをコンソールに表示しない. 初期値: ``False``

set_points
^^^^^^^^^^

.. code-block:: python

  def set_points(path: str) -> None:
  def set_points(paths: List[str]) -> None:
  def set_points(map: numpy.ndarray) -> None:

三次元点群を読み込む.
ファイル(.pcd)のパスを指定することで, 直接読み込むことが可能.
また, パスのリストを指定することで, 複数のファイルを一つの点群として読み込むことも可能.
さらに, NumPyの三次元点群データを指定して読み込むことも可能.
複数回実行した場合, それまで読み込まれていた点群は消去される.

* Args:

  * ``path (str)``: 三次元点群ファイル(.pcd)のパス
  * ``paths (List[str])``: 三次元点群ファイル(.pcd)のパスのリスト
  * ``map (numpy.ndarray)``: 三次元点群を格納したNumpy(N, 3)行列

* 実装例:

  .. code-block:: python

    from pointsmap import Points

    pcd_list = ['b.pcd', 'c.pcd', 'd.pcd']

    pts = Points()
    pts.set_points('a.pcd')
    pts.set_points(pcd_list)

  .. code-block:: python

    import h5py
    from pointsmap import Points

    pts = Points()

    with h5py.File('sample.hdf5', 'r') as h5file:
      pts.set_points(h5file['map/points'][()])

set_semanticpoints
^^^^^^^^^^^^^^^^^^

.. code-block:: python

  def set_semanticpoints(
    points: numpy.ndarray,
    semantic1d: numpy.ndarray
  ) -> None:

ラベル付き三次元点群を読み込む.
複数回実行した場合, それまで読み込まれていた点群は消去される.

* Args:

  * ``points (numpy.ndarray)``: ラベル付き三次元点群を構成する点群を格納したNumpy(N, 3)行列
  * ``semantic1d (numpy.ndarray)``: ラベル付き三次元点群のラベルを格納したNumpy(N,)行列

add_points
^^^^^^^^^^

.. code-block:: python

  def add_points(path: str) -> None:
  def add_points(paths: List[str]) -> None:
  def add_points(map: numpy.ndarray) -> None:

三次元点群を追加する.
ファイル(.pcd)のパスを指定することで, 直接追加することが可能.
また, パスのリストを指定することで, 複数のファイルを一つの点群として追加することも可能.
さらに, NumPyの三次元点群データを指定して追加することも可能.

* Args:

  * ``path (str)``: 三次元点群ファイル(.pcd)のパス
  * ``paths (List[str])``: 三次元点群ファイル(.pcd)のパスのリスト
  * ``map (numpy.ndarray)``: 三次元点群を格納したNumpy(N, 3)行列

add_semanticpoints
^^^^^^^^^^^^^^^^^^

.. code-block:: python

  def add_semanticpoints(
    points: numpy.ndarray,
    semantic1d: numpy.ndarray
  ) -> None:

ラベル付き三次元点群を追加する.

* Args:

  * ``points (numpy.ndarray)``: ラベル付き三次元点群を構成する点群を格納したNumpy(N, 3)行列
  * ``semantic1d (numpy.ndarray)``: ラベル付き三次元点群のラベルを格納したNumpy(N,)行列

get_points
^^^^^^^^^^

.. code-block:: python

  def get_points() -> numpy.ndarray:

三次元点群を取得する.

* Returns:

  * ``numpy.ndarray``: 三次元点群 (Numpy(N, 3)行列)

get_semanticpoints
^^^^^^^^^^^^^^^^^^

.. code-block:: python

  def get_semanticpoints() -> Tuple[numpy.ndarray, numpy.ndarray]:

ラベル付き三次元点群を取得する.

* Returns:

  * ``Tuple[numpy.ndarray, numpy.ndarray]``: 三次元点群 (Numpy(N, 3)行列)とラベルを格納した行列 (Numpy(N,)行列)

save_pcd
^^^^^^^^

.. code-block:: python

  def save_pcd(path: str) -> None:

三次元点群地図をPCDファイルに保存する.

* Args:

  * ``path (str)``: 保存するPCDファイルのパス

set_intrinsic
^^^^^^^^^^^^^

.. code-block:: python

  def set_intrinsic(K: numpy.ndarray) -> None:

3x3のカメラ内部パラメータを読み込む.

* Args:

  * ``K (numpy.ndarray)``: カメラ内部パラメータ

get_intrinsic
^^^^^^^^^^^^^

.. code-block:: python

  def get_intrinsic() -> numpy.ndarray:

設定した3x3のカメラ内部パラメータを取得する.

* Returns:

  * ``numpy.ndarray``: カメラ内部パラメータ

* 実装例:

  .. code-block:: python

    import numpy as np
    from pointsmap import Points

    pts = Points()

    K = numpy.array([
        [319.6,   0. , 384.],   # [Fx,  0, Cx]
        [  0. , 269.2, 192.],   # [ 0, Fy, Cy]
        [  0. ,   0. ,   1.]    # [ 0,  0,  1]
    ])

    pts.set_intrinsic(K)

    print(pts.get_intrinsic())

* 出力例:

  .. code-block::

    [[ 319.6    0.   384. ]
     [   0.   269.2  192. ]
     [   0.     0.     1. ]]

set_shape
^^^^^^^^^

.. code-block:: python

  def set_shape(
    shape: Tuple[int]
  ) -> None:

出力する画像のサイズを設定する.

* Args:

  * ``shape (Tuple[int])``: 画像サイズ (H, W)

get_shape
^^^^^^^^^

.. code-block:: python

  def get_shape() -> tuple:

設定した画像サイズを読み出す.

* Returns:

  * ``Tuple[int]``: 画像サイズ (H, W)

* 実装例:

  .. code-block:: python

    import numpy as np
    import cv2
    from pointsmap import Points

    pts = Points()

    img = cv2.imread("test.png")

    pts.set_shape(img.shape)

    print(pts.get_shape())

* 出力例:

  .. code-block::

    (256, 512)

set_depth_range
^^^^^^^^^^^^^^^

.. code-block:: python

  def set_depth_range(
    depth_range: Tuple[float]
  ) -> None:

深度マップに描画する深度の範囲を設定する.

* Args:

  * ``depth_range (Tuple[float])``: 深度の範囲 (MIN, MAX)

get_depth_range
^^^^^^^^^^^^^^^

.. code-block:: python

  def get_depth_range() -> None:

設定した深度の描画範囲を取得する.

* Returns:

  * ``tuple``: 深度の範囲 (MIN, MAX)

* 実装例:

  .. code-block:: python

    from pointsmap import Points

    pts = Points()

    print(pts.get_depth_range())

    pts.set_depth_range((1.0, 100.0))   # (MIN, MAX)
    print(pts.get_depth_range())

* 出力例:

  .. code-block::

    (0.0, inf)
    (1.0, 100.0)

set_depthmap
^^^^^^^^^^^^

.. code-block:: python

  def set_depthmap(
    depthmap: numpy.ndarray,
    translation: numpy.ndarray = numpy.array([0., 0., 0.], dtype=numpy.float32),
    quaternion: numpy.ndarray = numpy.array([0., 0., 0., 1.], dtype=numpy.float32),
    matrix_4x4: numpy.ndarray = None,
    transform_mode: int = TransformMode.CHILD2PARENT
  ) -> None:

深度マップを点群に変換し, 並進ベクトルとクォータニオン, または変換行列で座標変換をして格納する.

* Args:

  * ``depthmap (numpy.ndarray)``: 深度マップ
  * ``translation (numpy.ndarray)``: 並進ベクトル [x y z]
  * ``quaternion (numpy.ndarray)``: クォータニオン [x y z w]
  * ``matrix_4x4 (numpy.ndarray)``: 変換行列

    .. code-block::

      [[r11 r12 r13 tx]
       [r21 r22 r23 ty]
       [r31 r32 r33 tz]
       [  0   0   0  1]]

  * ``transform_mode (int, optional)``:

    * :ref:`transformmode`.CHILD2PARENT (0)
    * :ref:`transformmode`.PARENT2CHILD (1)

set_depthmap_semantic2d
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  def set_depthmap_semantic2d(
    depthmap: numpy.ndarray,
    semantic2d: numpy.ndarray,
    translation: numpy.ndarray = numpy.array([0., 0., 0.], dtype=numpy.float32),
    quaternion: numpy.ndarray = numpy.array([0., 0., 0., 1.], dtype=numpy.float32),
    matrix_4x4: numpy.ndarray = None,
    transform_mode: int = TransformMode.CHILD2PARENT
  ) -> None:

深度マップとSemanticマップを点群に変換し, 並進ベクトルとクォータニオン, または変換行列で座標変換をして格納する.

* Args:

  * ``depthmap (numpy.ndarray)``: 深度マップ
  * ``semantic2d (numpy.ndarray)``: Semantic マップ
  * ``translation (numpy.ndarray)``: 並進ベクトル [x y z]
  * ``quaternion (numpy.ndarray)``: クォータニオン [x y z w]
  * ``matrix_4x4 (numpy.ndarray)``: 変換行列

    .. code-block::

      [[r11 r12 r13 tx]
       [r21 r22 r23 ty]
       [r31 r32 r33 tz]
       [  0   0   0  1]]

  * ``transform_mode (int, optional)``:

    * :ref:`transformmode`.CHILD2PARENT (0)
    * :ref:`transformmode`.PARENT2CHILD (1)

transform
^^^^^^^^^

.. code-block:: python

  def transform(
    translation: numpy.ndarray = None,
    quaternion: numpy.ndarray = None,
    matrix_4x4: numpy.ndarray = None,
    transform_mode: int = TransformMode.CHILD2PARENT
  ) -> None:

格納されている点群を座標変換する.

* Args:

  * ``translation (numpy.ndarray)``: 並進ベクトル [x y z]
  * ``quaternion (numpy.ndarray)``: クォータニオン [x y z w]
  * ``matrix_4x4 (numpy.ndarray)``: 変換行列

    .. code-block::

      [[r11 r12 r13 tx]
       [r21 r22 r23 ty]
       [r31 r32 r33 tz]
       [  0   0   0  1]]

  * ``transform_mode (int, optional)``:

    * :ref:`transformmode`.CHILD2PARENT (0)
    * :ref:`transformmode`.PARENT2CHILD (1)

downsampling
^^^^^^^^^^^^

.. code-block:: python

  def downsampling(leaf_size:float) -> None:

格納されている点群をVoxel Grid Filterでダウンサンプリングする.

* Args:

  * ``leaf_size (float)``: Voxelの一辺の長さ (> 0)

create_depthmap
^^^^^^^^^^^^^^^

.. code-block:: python

  def create_depthmap(
    translation: numpy.ndarray = None,
    quaternion: numpy.ndarray = None,
    matrix_4x4: numpy.ndarray = None,
    transform_mode: int = TransformMode.CHILD2PARENT,
    filter_radius: int = 0,
    filter_threshold: float = 3.0
  ) -> numpy.ndarray:

並進ベクトルとクォータニオン, または変換行列を用いて三次元点群から深度マップを生成する.

* Args:

  * ``translation (numpy.ndarray)``: 並進ベクトル [x y z]
  * ``quaternion (numpy.ndarray)``: クォータニオン [x y z w]
  * ``matrix_4x4 (numpy.ndarray)``: 変換行列

    .. code-block::

      [[r11 r12 r13 tx]
       [r21 r22 r23 ty]
       [r31 r32 r33 tz]
       [  0   0   0  1]]

  * ``transform_mode (int, optional)``:

    * :ref:`transformmode`.CHILD2PARENT (0)
    * :ref:`transformmode`.PARENT2CHILD (1)

  * ``filter_radius (int, optional)``: Visibility Filterのカーネル半径. 0 の場合, フィルタ処理を行わない. (既定値: ``0``)
  * ``filter_threshold (float, optional)``: Visibility Filterの閾値. (既定値: ``3.0``)

* Returns:

  * ``numpy.ndarray``: 深度マップ

* 実装例:

  .. code-block:: python

    import numpy as np
    import h5py
    import cv2
    from pointsmap import *

    pts = Points()

    with h5py.File('sample.hdf5', 'r') as h5file:
      K = np.array([[h5file['K/rgb/Fx'][()], 0., h5file['K/rgb/Cx'][()]],
                    [0., h5file['K/rgb/Fy'][()], h5file['K/rgb/Cy'][()]],
                    [0., 0., 1.]])
      pts.set_intrinsic(K)

      pts.set_shape(h5file['data/0/rgb'].shape)

      pts.set_points(h5file['map/points'][()])

      translation = h5file['data/0/pose/rgb/translation'][()]
      quaternion = h5file['data/0/pose/rgb/rotation'][()]

      map_depth = pts.create_depthmap(
        translation=translation,
        quaternion=quaternion,
        transform_mode=TransformMode.PARENT2CHILD)

      map_depth_color = depth2colormap(map_depth, 0.0, 100.0)

      cv2.imwrite('sample.png', map_depth_color)

create_semantic2d
^^^^^^^^^^^^^^^^^

.. code-block:: python

  def create_semantic2d(
    translation: numpy.ndarray = None,
    quaternion: numpy.ndarray = None,
    matrix_4x4: numpy.ndarray = None,
    transform_mode: int = TransformMode.CHILD2PARENT,
    filter_radius: int = 0,
    filter_threshold: float = 3.0
  ) -> numpy.ndarray:

並進ベクトルとクォータニオン, または変換行列を用いて三次元点群のラベルからSemanticマップを生成する.

* Args:

  * ``translation (numpy.ndarray)``: 並進ベクトル [x y z]
  * ``quaternion (numpy.ndarray)``: クォータニオン [x y z w]
  * ``matrix_4x4 (numpy.ndarray)``: 変換行列

    .. code-block::

      [[r11 r12 r13 tx]
       [r21 r22 r23 ty]
       [r31 r32 r33 tz]
       [  0   0   0  1]]

  * ``transform_mode (int, optional)``:

    * :ref:`transformmode`.CHILD2PARENT (0)
    * :ref:`transformmode`.PARENT2CHILD (1)

  * ``filter_radius (int, optional)``: Visibility Filterのカーネル半径. 0 の場合, フィルタ処理を行わない. (既定値: ``0``)
  * ``filter_threshold (float, optional)``: Visibility Filterの閾値. (既定値: ``3.0``)

* Returns:

  * ``numpy.ndarray``: Semanticマップ

* 実装例:

  .. code-block:: python

    import numpy as np
    import h5py
    import cv2
    from pointsmap import *

    pts = Points()

    with h5py.File('sample.hdf5', 'r') as h5file:
      K = np.array([[h5file['K/rgb/Fx'][()], 0., h5file['K/rgb/Cx'][()]],
                    [0., h5file['K/rgb/Fy'][()], h5file['K/rgb/Cy'][()]],
                    [0., 0., 1.]])
      pts.set_intrinsic(K)

      pts.set_shape(h5file['data/0/rgb'].shape)

      pts.set_points(h5file['map/points'][()])

      translation = h5file['data/0/pose/rgb/translation'][()]
      quaternion = h5file['data/0/pose/rgb/rotation'][()]

      map_semantic2d = pts.create_semantic2d(
        translation=translation,
        quaternion=quaternion,
        transform_mode=TransformMode.PARENT2CHILD)

      map_semantic2d_color = np.zeros(h5file['data/0/rgb'].shape, dtype=np.uint8)
      for key, item in h5file['label/semantic2d'].items():
          map_semantic2d_c[np.where(map_semantic2d == int(key))] = item['color'][()]

      cv2.imwrite('sample.png', map_semantic2d_color)

.. _vgm_class:

VoxelGridMap
------------

.. code-block:: python

  from pointsmap import VoxelGridMap
  vgm = VoxelGridMap(quiet: bool = False)

三次元点群地図を扱うクラス.
小規模な三次元点群を扱う場合は, :ref:`points_class` クラスを推奨.

* Args:

  * ``quiet (bool, optional)``: ``True`` の場合, "ERROR", "WARNING"以外のメッセージをコンソールに表示しない. 初期値: ``False``

set_pointsmap
^^^^^^^^^^^^^

.. code-block:: python

  def set_pointsmap(path: str, voxel_size: float = 10.0) -> None:
  def set_pointsmap(paths: List[str], voxel_size: float = 10.0) -> None:
  def set_pointsmap(map: numpy.ndarray, voxel_size: float = 10.0) -> None:

三次元点群地図を読み込む.
ファイル(.pcd)のパスを指定することで, 直接読み込むことが可能.
また, パスのリストを指定することで, 複数のファイルを一つの地図として読み込むことも可能.
さらに, NumPyの三次元点群地図データを指定して読み込むことも可能.

* Args:

  * ``path (str)``: 三次元点群地図ファイル(.pcd)のパス
  * ``paths (List[str])``: 三次元点群地図ファイル(.pcd)のパスのリスト
  * ``map (numpy.ndarray)``: 三次元点群地図を格納したNumpy(N, 3)行列
  * ``voxel_size (float, optional)``: Voxelのサイズ (初期値: ``10.0``)

* 実装例:

  .. code-block:: python

    from pointsmap import VoxelGridMap

    pcd_list = ['b.pcd', 'c.pcd', 'd.pcd']

    vgm = VoxelGridMap()
    vgm.set_pointsmap('a.pcd')
    vgm.set_pointsmap(pcd_list)

  .. code-block:: python

    import h5py
    from pointsmap import VoxelGridMap

    vgm = VoxelGridMap()

    with h5py.File('sample.hdf5', 'r') as h5file:
      vgm.set_pointsmap(h5file['map/points'][()])

set_semanticmap
^^^^^^^^^^^^^^^

.. code-block:: python

  def set_semanticmap(
    points: numpy.ndarray,
    semantic1d: numpy.ndarray,
    voxel_size: float = 10.0
  ) -> None:

ラベル付き三次元点群地図を読み込む．

* Args:

  * ``points (numpy.ndarray)``: ラベル付き三次元点群地図を構成する点群を格納したNumpy(N, 3)行列
  * ``semantic1d (numpy.ndarray)``: 三次元点群地図のラベルを格納したNumpy(N,)行列
  * ``voxel_size (float, optional)``: Voxelのサイズ (初期値: ``10.0``)

set_voxelgridmap
^^^^^^^^^^^^^^^^

.. code-block:: python

  def set_voxelgridmap(
    vgm: numpy.ndarray,
    voxel_size: float,
    voxels_min: Tuple[float, float, float],
    voxels_max: Tuple[float, float, float],
    voxels_center: Tuple[float, float, float],
    voxels_origin: Tuple[int, int, int]
  ) -> None:

Voxel Gri Mapを読み込む.

* Args:

  * ``vgm (numpy.ndarray)``: Voxel Grid Map

    (Compound型(N,)['x','y','z','label']を格納したNumpy(Z, Y, X)行列)
  * ``voxel_size (float, optional)``: Voxelのサイズ
  * ``voxel_min (Tuple[float, float, float])``: Voxel Grid Mapの範囲の最小値

    (z_min, y_min, x_min)
  * ``voxel_max (Tuple[float, float, float])``: Voxel Grid Mapの範囲の最大値

    (z_max, y_max, x_max)
  * ``voxels_center (Tuple[float, float, float])``: Voxel Grid Mapの中心座標

    (z_center, y_center, x_center)
  * ``voxels_origin (Tuple[int, int, int])``: Voxel Grid Mapの中心座標が含まれるVoxelのインデックス

    (z_origin, y_origin, x_origin)

set_empty_voxelgridmap
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  def set_empty_voxelgridmap(
    voxels_len: Tuple[int, int, int],
    voxel_size: float,
    voxels_min: Tuple[float, float, float],
    voxels_max: Tuple[float, float, float],
    voxels_center: Tuple[float, float, float],
    voxels_origin: Tuple[int, int, int]
  ) -> None:

空のVoxel Grid Mapを格納する.

* Args:

  * ``voxels_len (numpy.ndarray)``: Voxelの数 (各軸方向)

    (z_len, y_len, x_len)
  * ``voxel_size (float, optional)``: Voxelのサイズ
  * ``voxel_min (Tuple[float, float, float])``: Voxel Grid Mapの範囲の最小値

    (z_min, y_min, x_min)
  * ``voxel_max (Tuple[float, float, float])``: Voxel Grid Mapの範囲の最大値

    (z_max, y_max, x_max)
  * ``voxels_center (Tuple[float, float, float])``: Voxel Grid Mapの中心座標

    (z_center, y_center, x_center)
  * ``voxels_origin (Tuple[int, int, int])``: Voxel Grid Mapの中心座標が含まれるVoxelのインデックス

    (z_origin, y_origin, x_origin)


get_pointsmap
^^^^^^^^^^^^^

.. code-block:: python

  def get_pointsmap() -> numpy.ndarray:

三次元点群地図を取得する. ラベルも出力する場合は :ref:`get_semanticmap` を使用する.

* Returns:

  * ``numpy.ndarray``: 三次元点群地図 (Numpy(N, 3)行列)

.. _get_semanticmap:

get_semanticmap
^^^^^^^^^^^^^^^

.. code-block:: python

  def get_semanticmap() -> Tuple[numpy.ndarray, numpy.ndarray]:

ラベル付き三次元点群地図を取得する.

* Returns:

  * ``Tuple[numpy.ndarray, numpy.ndarray]``: 三次元点群地図

    (Numpy(N, 3)行列)とラベル(Numpy(N,)行列) のTuple

get_voxel_points
^^^^^^^^^^^^^^^^

.. code-block:: python

  def get_voxel_points() -> numpy.ndarray:

Voxel Grid Mapを取得する. ラベルも出力する際は :ref:`get_voxel_semantic3d` を使用する.

* Returns:

  * ``numpy.ndarray``: Voxel Grid Map

    (Compound型(N,)['x','y','z']を格納したNumpy(Z, Y, X)行列)

.. _get_voxel_semantic3d:

get_voxel_semantic3d
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  def get_voxel_semantic3d() -> numpy.ndarray:

ラベル付きVoxel Grid Mapを取得する.

* Returns:

  * ``numpy.ndarray``: Voxel Grid Map

    (Compound型(N,)['x','y','z','label']を格納したNumpy(Z, Y, X)行列)

save_pcd
^^^^^^^^

.. code-block:: python

  def save_pcd(path: str) -> None:

三次元点群地図をPCDファイルに保存する.

* Args:

  * ``path (str)``: 保存するPCDファイルのパス

get_voxel_size
^^^^^^^^^^^^^^

.. code-block:: python

  def get_voxel_size() -> float:

Voxelのサイズを取得する.

* Returns:

  * ``float``: Voxelのサイズ

get_voxels_min
^^^^^^^^^^^^^^

.. code-block:: python

  def get_voxels_min() -> Tuple[float, float, float]:

Voxel Grid Mapの範囲の最小値を取得する.

* Returns:

  * ``Tuple[float, float, float]``: Voxel Grid Mapの範囲の最小値

    (z_min, y_min, x_min)

get_voxels_max
^^^^^^^^^^^^^^

.. code-block:: python

  def get_voxels_max() -> Tuple[float, float, float]:

Voxel Grid Mapの範囲の最大値を取得する.

* Returns:

  * ``Tuple[float, float, float]``: Voxel Grid Mapの範囲の最小値

    (z_max, y_max, x_max)

get_voxels_center
^^^^^^^^^^^^^^^^^

.. code-block:: python

  def get_voxels_center() -> Tuple[float, float, float]:

Voxel Grid Mapの中心座標を取得する.

* Returns:

  * ``Tuple[float, float, float]``: Voxel Grid Mapの中心座標

    (z_center, y_center, x_center)

get_voxels_origin
^^^^^^^^^^^^^^^^^

.. code-block:: python

  def get_voxels_origin() -> Tuple[int, int, int]:

Voxel Grid Mapの中心座標が含まれるVoxelのインデックスを取得する.

* Returns:

  * ``Tuple[int, int, int]``: Voxel Grid Mapの中心座標が含まれるVoxelのインデックス

    (z_origin, y_origin, x_origin)

get_voxels_include_frustum
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  def get_voxels_include_frustum(
    translation: np.ndarray = None,
    quaternion: np.ndarray = None,
    matrix_4x4: np.ndarray = None
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

同次変換行列, または並進ベクトルとクォータニオンを入力し, 画角内に含まれるVoxelのインデックスを取得する.

* Args:

  * ``translation (numpy.ndarray)``: 並進ベクトル [x y z]
  * ``quaternion (numpy.ndarray)``: クォータニオン [x y z w]
  * ``matrix_4x4 (numpy.ndarray)``: 変換行列

    .. code-block::

      [[r11 r12 r13 tx]
       [r21 r22 r23 ty]
       [r31 r32 r33 tz]
       [  0   0   0  1]]

* Returns:

  * ``Tuple[np.ndarray, np.ndarray, np.ndarray]``: 画角内に含まれるVoxelのインデックス. (``numpy.where()`` と同様の出力)

set_intrinsic
^^^^^^^^^^^^^

.. code-block:: python

  def set_intrinsic(K: numpy.ndarray) -> None:

3x3のカメラ内部パラメータを読み込む.

* Args:

  * ``K (numpy.ndarray)``: カメラ内部パラメータ

get_intrinsic
^^^^^^^^^^^^^

.. code-block:: python

  def get_intrinsic() -> numpy.ndarray:

設定した3x3のカメラ内部パラメータを取得する.

* Returns:

  * ``numpy.ndarray``: カメラ内部パラメータ

* 実装例:

  .. code-block:: python

    import numpy as np
    from pointsmap import VoxelGridMap

    vgm = VoxelGridMap()

    K = numpy.array([
        [319.6,   0. , 384.],   # [Fx,  0, Cx]
        [  0. , 269.2, 192.],   # [ 0, Fy, Cy]
        [  0. ,   0. ,   1.]    # [ 0,  0,  1]
    ])

    vgm.set_intrinsic(K)

    print(vgm.get_intrinsic())

* 出力例:

  .. code-block::

    [[ 319.6    0.   384. ]
     [   0.   269.2  192. ]
     [   0.     0.     1. ]]

set_shape
^^^^^^^^^

.. code-block:: python

  def set_shape(
    shape: Tuple[int]
  ) -> None:

出力する画像のサイズを設定する.

* Args:

  * ``shape (Tuple[int])``: 画像サイズ (H, W)

get_shape
^^^^^^^^^

.. code-block:: python

  def get_shape() -> tuple:

設定した画像サイズを読み出す.

* Returns:

  * ``Tuple[int]``: 画像サイズ (H, W)

* 実装例:

  .. code-block:: python

    import numpy as np
    import cv2
    from pointsmap import VoxelGridMap

    vgm = VoxelGridMap()

    img = cv2.imread("test.png")

    vgm.set_shape(img.shape)

    print(vgm.get_shape())

* 出力例:

  .. code-block::

    (256, 512)

set_depth_range
^^^^^^^^^^^^^^^

.. code-block:: python

  def set_depth_range(
    depth_range: Tuple[float]
  ) -> None:

深度マップに描画する深度の範囲を設定する.

* Args:

  * ``depth_range (Tuple[float])``: 深度の範囲 (MIN, MAX)

get_depth_range
^^^^^^^^^^^^^^^

.. code-block:: python

  def get_depth_range() -> None:

設定した深度の描画範囲を取得する.

* Returns:

  * ``tuple``: 深度の範囲 (MIN, MAX)

* 実装例:

  .. code-block:: python

    from pointsmap import VoxelGridMap

    vgm = VoxelGridMap()

    print(vgm.get_depth_range())

    vgm.set_depth_range((1.0, 100.0))   # (MIN, MAX)
    print(vgm.get_depth_range())

* 出力例:

  .. code-block::

    (0.0, inf)
    (1.0, 100.0)

create_depthmap
^^^^^^^^^^^^^^^

.. code-block:: python

  def create_depthmap(
    translation: numpy.ndarray = None,
    quaternion: numpy.ndarray = None,
    matrix_4x4: numpy.ndarray = None,
    transform_mode: int = TransformMode.CHILD2PARENT,
    filter_radius: int = 0,
    filter_threshold: float = 3.0
  ) -> numpy.ndarray:

並進ベクトルとクォータニオン, または変換行列を用いてVoxel Grid Mapから深度マップを生成する.

* Args:

  * ``translation (numpy.ndarray)``: 並進ベクトル [x y z]
  * ``quaternion (numpy.ndarray)``: クォータニオン [x y z w]
  * ``matrix_4x4 (numpy.ndarray)``: 変換行列

    .. code-block::

      [[r11 r12 r13 tx]
       [r21 r22 r23 ty]
       [r31 r32 r33 tz]
       [  0   0   0  1]]

  * ``transform_mode (int, optional)``:

    * :ref:`transformmode`.CHILD2PARENT (0)
    * :ref:`transformmode`.PARENT2CHILD (1)

  * ``filter_radius (int, optional)``: Visibility Filterのカーネル半径. 0 の場合, フィルタ処理を行わない. (既定値: ``0``)
  * ``filter_threshold (float, optional)``: Visibility Filterの閾値. (既定値: ``3.0``)

* Returns:

  * ``numpy.ndarray``: 深度マップ

* 実装例:

  .. code-block:: python

    import numpy as np
    import h5py
    import cv2
    from pointsmap import *

    vgm = VoxelGridMap()

    with h5py.File('sample.hdf5', 'r') as h5file:
      K = np.array([[h5file['K/rgb/Fx'][()], 0., h5file['K/rgb/Cx'][()]],
                    [0., h5file['K/rgb/Fy'][()], h5file['K/rgb/Cy'][()]],
                    [0., 0., 1.]])
      vgm.set_intrinsic(K)

      vgm.set_shape(h5file['data/0/rgb'].shape)

      vgm.set_pointsmap(h5file['map/points'][()])

      translation = h5file['data/0/pose/rgb/translation'][()]
      quaternion = h5file['data/0/pose/rgb/rotation'][()]

      map_depth = vgm.create_depthmap(
        translation=translation,
        quaternion=quaternion,
        transform_mode=TransformMode.PARENT2CHILD)

      map_depth_color = depth2colormap(map_depth, 0.0, 100.0)

      cv2.imwrite('sample.png', map_depth_color)

create_semantic2d
^^^^^^^^^^^^^^^^^

.. code-block:: python

  def create_semantic2d(
    translation: numpy.ndarray = None,
    quaternion: numpy.ndarray = None,
    matrix_4x4: numpy.ndarray = None,
    transform_mode: int = TransformMode.CHILD2PARENT,
    filter_radius: int = 0,
    filter_threshold: float = 3.0
  ) -> numpy.ndarray:

並進ベクトルとクォータニオン, または変換行列を用いてVoxel Grid MapのラベルからSemanticマップを生成する.

* Args:

  * ``translation (numpy.ndarray)``: 並進ベクトル [x y z]
  * ``quaternion (numpy.ndarray)``: クォータニオン [x y z w]
  * ``matrix_4x4 (numpy.ndarray)``: 変換行列

    .. code-block::

      [[r11 r12 r13 tx]
       [r21 r22 r23 ty]
       [r31 r32 r33 tz]
       [  0   0   0  1]]

  * ``transform_mode (int, optional)``:

    * :ref:`transformmode`.CHILD2PARENT (0)
    * :ref:`transformmode`.PARENT2CHILD (1)

  * ``filter_radius (int, optional)``: Visibility Filterのカーネル半径. 0 の場合, フィルタ処理を行わない. (既定値: ``0``)
  * ``filter_threshold (float, optional)``: Visibility Filterの閾値. (既定値: ``3.0``)

* Returns:

  * ``numpy.ndarray``: Semanticマップ

* 実装例:

.. code-block:: python

  import numpy as np
  import h5py
  import cv2
  from pointsmap import *

  vgm = VoxelGridMap()

  with h5py.File('sample.hdf5', 'r') as h5file:
    K = np.array([[h5file['K/rgb/Fx'][()], 0., h5file['K/rgb/Cx'][()]],
                  [0., h5file['K/rgb/Fy'][()], h5file['K/rgb/Cy'][()]],
                  [0., 0., 1.]])
    vgm.set_intrinsic(K)

    vgm.set_shape(h5file['data/0/rgb'].shape)

    vgm.set_pointsmap(h5file['map/points'][()])

    translation = h5file['data/0/pose/rgb/translation'][()]
    quaternion = h5file['data/0/pose/rgb/rotation'][()]

    map_semantic2d = vgm.create_semantic2d(
      translation=translation,
      quaternion=quaternion,
      transform_mode=TransformMode.PARENT2CHILD)

    map_semantic2d_color = np.zeros(h5file['data/0/rgb'].shape, dtype=np.uint8)
    for key, item in h5file['label/semantic2d'].items():
        map_semantic2d_c[np.where(map_semantic2d == int(key))] = item['color'][()]

    cv2.imwrite('sample.png', map_semantic2d_color)
