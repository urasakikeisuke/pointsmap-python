#include <string>

#include <boost/python/numpy.hpp>
#include <numpy/ndarrayobject.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace np = boost::python::numpy;
namespace py = boost::python;

namespace pointsmap {

#ifndef POINTSMAP_COMMON_HPP_INCLUDE

#define POINTSMAP_COMMON_HPP_INCLUDE

#define FILTER_RADIUS 0
#define FILTER_THRESHOLD 3.0f

enum axisXYZ {
    X = 0,
    Y = 1,
    Z = 2
};

enum frustumPointIndex {
    FrontBottomLeft = 0,
    FrontTopLeft = 1,
    FrontTopRight = 2,
    FrontBottomRight = 3,
    BackBottomLeft = 4,
    BackTopLeft = 5,
    BackTopRight = 6,
    BackBottomRight = 7,
    Origin = 8
};

enum frustumSurfaceIndex {
    Left = 0,
    Top = 1,
    Right = 2,
    Bottom = 3,
    Front = 4,
    Back = 5
};

enum frustumVectorIndex {
    Origin2BackBottomLeft = 0,
    Origin2BackTopLeft = 1,
    Origin2BackTopRight = 2,
    Origin2BackBottomRight = 3,
    FrontBottomRight2FrontBottomLeft = 4,
    FrontBottomRight2FrontTopRight = 5,
    BackTopLeft2BackBottomLeft = 6,
    BackTopLeft2BackTopRight = 7
};

template<typename T> 
struct point_xyz {
    T x;
    T y;
    T z;
};

//  Voxel
struct points_voxel {
    point_xyz<float_t> min;
    point_xyz<float_t> max;
    pcl::PointCloud<pcl::PointXYZL> points;
};

struct img_scan {
    int x_begin;
    int x_end;
    int y_begin;
    int y_end;
};

//  相対自己位置の逆変換
void invert_transform(const Eigen::Matrix4f &in_matrix, Eigen::Matrix4f &out_matrix);

//  相対自己位置の逆変換
void invert_transform(const Eigen::Vector3f &in_tr, const Eigen::Quaternionf &in_q, Eigen::Vector3f &out_tr, Eigen::Quaternionf &out_q);

//  変換行列を逆変換する (for Python)
np::ndarray invert_transform_matrix(const np::ndarray &matrix_4x4);

//  並進ベクトルとクォータニオンを逆変換する (for Python)
py::tuple invert_transform_quaternion(const np::ndarray &translation, const np::ndarray &quaternion);

//  変換行列を並進ベクトルとクォータニオンに変換する
void matrix2quaternion(const Eigen::Matrix4f &in_matrix, Eigen::Vector3f &out_translation, Eigen::Quaternionf &out_quaternion);

//  変換行列を並進ベクトルとクォータニオンに変換する (for Python)
py::tuple matrix2quaternion_python(const np::ndarray &matrix_4x4);

//  並進ベクトルとクォータニオンを変換行列に変換する
void quaternion2matrix(const Eigen::Vector3f &in_translation, const Eigen::Quaternionf &in_quaternion, Eigen::Matrix4f &out_matrix);

//  並進ベクトルとクォータニオンを変換行列に変換する (for Python)
np::ndarray quaternion2matrix_python(const np::ndarray &translation, const np::ndarray &quaternion);

//  cv::Matをnp::ndarrayへ変換する
np::ndarray cvmat2ndarray(const cv::Mat& src);

//  np::ndarrayをcv::Matへ変換する
void ndarray2cvmat(const np::ndarray &src, cv::Mat &dst);

//  深度を用いてフィルタリングを行う
void depth_filter(const pcl::PointCloud<pcl::PointXYZL> &src, pcl::PointCloud<pcl::PointXYZL> &dst, const float_t min, const float_t max);

//  並進ベクトルとクォータニオンで座標変換を行う
void transform_pointcloud(const pcl::PointCloud<pcl::PointXYZL> &src, pcl::PointCloud<pcl::PointXYZL> &dst, const Eigen::Vector3f &translation, const Eigen::Quaternionf &quaternion);

//  変換行列で座標変換を行う
void transform_pointcloud(const pcl::PointCloud<pcl::PointXYZL> &src, pcl::PointCloud<pcl::PointXYZL> &dst, const Eigen::Matrix4f &matrix);

//  深度マップをカラーマップへ変換
void depth2colormap(const cv::Mat &src, cv::Mat &dst, const float min, const float max, const int type, const bool invert);

//  深度マップをカラーマップへ変換
np::ndarray depth2colormap_python(const np::ndarray &src, const double_t min, const double_t max, const int type, const bool invert);

//  np::ndarrayをEigen::Vector3fへ変換
void ndarray2translation(const np::ndarray &src, Eigen::Vector3f &dst);

//  np::ndarrayをEigen::Quaternionfへ変換
void ndarray2quaternion(const np::ndarray &src, Eigen::Quaternionf &dst);

//  np::ndarrayをEigen::Matrix4fへ変換
void ndarray2matrix(const np::ndarray &src, Eigen::Matrix4f &dst);

//  Eigen::Vector3fをnp::ndarrayへ変換
np::ndarray translation2ndarray(const Eigen::Vector3f &src);

//  Eigen::Quaternionfをnp::ndarrayへ変換
np::ndarray quaternion2ndarray(const Eigen::Quaternionf &src);

//  Eigen::Matrix4fをnp::ndarrayへ変換
np::ndarray matrix2ndarray(const Eigen::Matrix4f &src);

//  pcl::PointCloudをpoints型のnp::ndarrayへ変換
np::ndarray pointcloud2nppoints(const pcl::PointCloud<pcl::PointXYZL> &src);

//  pcl::PointCloudをsemantic3d型のnp::ndarrayへ変換
py::tuple pointcloud2npsemantic3d(const pcl::PointCloud<pcl::PointXYZL> &src);

//  pcl::PointCloudをcompoundへ変換
np::ndarray pointcloud2compound(const pcl::PointCloud<pcl::PointXYZL> &src, const bool label);

//  points型のnp::ndarrayをpcl::PointCloudへ変換
void nppoints2pointcloud(const np::ndarray &points, pcl::PointCloud<pcl::PointXYZL> &dst);

//  semantic3d型のnp::ndarrayをpcl::PointCloudへ変換
void npsemantic3d2pointcloud(const np::ndarray &points, const np::ndarray &semantic1d, pcl::PointCloud<pcl::PointXYZL> &dst);

//  compoundをpcl::PointCloudへ変換
void compound2pointcloud(const np::ndarray &compound, pcl::PointCloud<pcl::PointXYZL> &dst);

//  [mm]単位の16bit unsigned intの画像を[m]単位の32bit floatの画像に変換
void depth_openni2canonical(const cv::Mat &src, cv::Mat &dst);

//  複数の変換行列を合成する
void combine_transforms(const std::vector<Eigen::Matrix4f> &src_matrixes, Eigen::Matrix4f &dst_matrix);

//  複数の並進ベクトルとクォータニオンを合成する
void combine_transforms(const std::vector<Eigen::Vector3f> &src_translations, const std::vector<Eigen::Quaternionf> &src_quaternions, Eigen::Vector3f &dst_translation, Eigen::Quaternionf &dst_quaternion);

//  複数の変換行列を合成する (for Python)
np::ndarray combine_transforms_matrix(const py::list &src_matrixes);

//  複数の並進ベクトルとクォータニオンを合成する (for Python)
py::tuple combine_transforms_quaternion(const py::list &src_translations, const py::list &src_quaternions);

//  Voxelのインデックスをndarrayのtupleへ変換
py::tuple voxelindexs2tuple(const std::vector<point_xyz<size_t> > &voxel_indexs);

#endif

}   //  namespace pointsmap
