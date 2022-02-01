======
Python
======

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

同次変換行列，または並進ベクトル・クォータニオンを逆変換する．

* Args:

  * ``translation (numpy.ndarray)``: 並進ベクトル [x y z]
  * ``quaternion (numpy.ndarray)``: クォータニオン [x y z w]

    または

  * ``matrix_4x4 (numpy.ndarray)``: 4x4の同次変換行列

* Returns:

  * ``numpy.ndarray``: 逆変換行列

    または

  * ``tuple``: 逆変換した並進ベクトル・クォータニオン (translation, quaternion)

* 実装例

  .. code-block:: python

    import numpy as np
    from pointsmap import *

    matrix = np.array([ [ 0., 1., 0., 1.],
                        [-1., 0., 0., 2.],
                        [ 0., 0., 1., 0.],
                        [ 0., 0., 0., 1.]])
    print(invertTransform(matrix_4x4=matrix))

    translation = np.array([74.63491058, 195.49958801, 2.00133348])
    quaternion = np.array([0.49215598, -0.50772443, 0.50772268, -0.49215452])
    print(invertTransform(translation=translation, quaternion=quaternion))

* 出力例

  .. code-block:: text

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

深度マップからカラーマップを生成する．

* Args:

  * ``src (numpy.ndarray)``: 深度マップ
  * ``min (float)``: 深度の表示範囲 (最小値)
  * ``max (float)``: 深度の表示範囲 (最大値)
  * ``type (int, optional)``: cv2.ColormapTypes (既定値: ``cv2.COLORMAP_JET``)
  * ``invert (bool, optional)``: カラーマップを反転する

* Returns:

  * ``numpy.ndarray``: カラーマップ

* 実装例

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
    translations: List[np.ndarray] = None,
    quaternions: List[np.ndarray] = None,
    matrixes: List[np.ndarray] = None
  ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

複数の変換行列, または並進ベクトル・クォータニオンを合成する

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

Points
------

VoxelGridMap
------------
