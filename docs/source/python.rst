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

クラス
======

Points
------

VoxelGridMap
------------
