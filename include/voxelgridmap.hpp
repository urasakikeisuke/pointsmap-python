#include "common.hpp"
#include "base/points_base.hpp"

namespace pointsmap {

#ifndef POINTSMAP_VOXELGRIDMAP_HPP_INCLUDE

#define POINTSMAP_VOXELGRIDMAP_HPP_INCLUDE

class VoxelGridMap : public PointsBase
{
    public:
        //  Constructor
        VoxelGridMap(bool quiet);

        //  Methods

        //  三次元点群地図をVoxelに格納する
        void set_points(const pcl::PointCloud<pcl::PointXYZL> &in, const double_t voxel_size);
        //  PCDファイルから点群を読み込み，Voxelに格納する
        void set_pointsfile(const std::string &path, const double_t voxel_size);
        //  PCDファイルから点群を読み込み，Voxelに格納する
        void set_pointsfile_python(const std::string &path, const double_t voxel_size);
        //  複数のPCDファイルから点群を読み込み，Voxelに格納する
        void set_pointsfiles(const std::vector<std::string> &paths, const double_t voxel_size);
        //  複数のPCDファイルから点群を読み込み，Voxelに格納する
        void set_pointsfiles_python(const py::list &paths, const double_t voxel_size);
        //  np::ndarrayから点群を読み込み，Voxelに格納する
        void set_points_from_numpy(const np::ndarray &points, const double_t voxel_size);
        //  semantic3dの点群データを読み込み，Voxelに格納する
        void set_semanticpoints_from_numpy(const np::ndarray &points, const np::ndarray &semantic1d, const double_t voxel_size);
        //  Voxelに格納された点群を取り出す
        void get_points(pcl::PointCloud<pcl::PointXYZL> &out) override;
        //  VoxelGridMapを格納する
        void set_voxelgridmap(const std::vector<std::vector<std::vector<points_voxel> > > &vgm, const float_t voxel_size, const point_xyz<float_t> &voxels_min, const point_xyz<float_t> &voxels_max, const point_xyz<float_t> &voxels_center, const point_xyz<size_t> &voxels_origin);
        //  VoxelGridMapを格納する (for Python)
        void set_voxelgridmap_python(const np::ndarray &vgm, const double_t voxel_size, const py::tuple &voxels_min, const py::tuple &voxels_max, const py::tuple &voxels_center, const py::tuple &voxels_origin);
        //  空のVoxelGridMapを作成する
        void set_empty_voxelgridmap(const point_xyz<size_t> &voxels_len, const float_t voxel_size, const point_xyz<float_t> &voxels_min, const point_xyz<float_t> &voxels_max, const point_xyz<float_t> &voxels_center, const point_xyz<size_t> &voxels_origin);
        //  空のVoxelGridMapを作成する
        void set_empty_voxelgridmap_python(const py::tuple &voxels_len, const double_t voxel_size, const py::tuple &voxels_min, const py::tuple &voxels_max, const py::tuple &voxels_center, const py::tuple &voxels_origin);
        //  VoxelGridMapを取得する
        void get_voxelgridmap(std::vector<std::vector<std::vector<points_voxel> > > &vgm);
        //  VoxelGridMapを取得する (for Python)
        np::ndarray get_voxelgridmap_python(const bool label);
        //  Voxelのサイズを取得する
        float_t get_voxel_size();
        //  Voxelのサイズを取得する (for Python)
        double_t get_voxel_size_python();
        //  VoxelGridMapの範囲の最小値を取得する
        void get_voxels_min(point_xyz<float_t> &voxels_min);
        //  VoxelGridMapの範囲の最小値を取得する (for Python)
        py::tuple get_voxels_min_python();
        //  VoxelGridMapの範囲の最大値を取得する
        void get_voxels_max(point_xyz<float_t> &voxels_max);
        //  VoxelGridMapの範囲の最大値を取得する (for Python)
        py::tuple get_voxels_max_python();
        //  VoxelGridMapの中心座標を取得する
        void get_voxels_center(point_xyz<float_t> &voxels_center);
        //  VoxelGridMapの中心座標を取得する (for Python)
        py::tuple get_voxels_center_python();
        //  VoxelGridMapの中心のVoxelのインデックスを取得する
        void get_voxels_origin(point_xyz<size_t> &voxels_origin);
        //  VoxelGridMapの中心のVoxelのインデックスを取得する (for Python)
        py::tuple get_voxels_origin_python();
        //  視錐台に含まれるVoxelを判定し，そのVoxelのインデックスを取得する
        void voxels_include_frustum(const Eigen::Vector3f &translation, const Eigen::Quaternionf &quaternion, std::vector<point_xyz<size_t> > &dst_voxel_indexs);
        //  視錐台に含まれるVoxelを判定し，そのVoxelのインデックスを取得する
        void voxels_include_frustum(const Eigen::Matrix4f &matrix, std::vector<point_xyz<size_t> > &dst_voxel_indexs);
        //  変換行列から視錐台に含まれるVoxelを判定し，そのVoxelのインデックスを取得する
        py::tuple voxels_include_frustum_from_matrix(const np::ndarray &matrix_4x4);
        //  並進ベクトルとクォータニオンから視錐台に含まれるVoxelを判定し，そのVoxelのインデックスを取得する
        py::tuple voxels_include_frustum_from_quaternion(const np::ndarray &translation, const np::ndarray &quaternion);
        //  複数のVoxelを格納する (for Python)
        void set_voxels_python(const py::tuple &indexs, const np::ndarray &voxels);
        //  変換行列から深度マップを生成する
        np::ndarray create_depthmap_from_matrix(const np::ndarray &matrix_4x4, const int filter_radius = FILTER_RADIUS, const float_t filter_threshold = FILTER_THRESHOLD);
        //  並進ベクトルとクォータニオンから深度マップを生成する
        np::ndarray create_depthmap_from_quaternion(const np::ndarray &translation, const np::ndarray &quaternion, const int filter_radius = FILTER_RADIUS, const float_t filter_threshold = FILTER_THRESHOLD);
        //  変換行列からsemantic2dを生成する
        np::ndarray create_semantic2d_from_matrix(const np::ndarray &matrix_4x4, const int filter_radius = FILTER_RADIUS, const float_t filter_threshold = FILTER_THRESHOLD);
        //  並進ベクトルとクォータニオンからsemantic2dを生成する
        np::ndarray create_semantic2d_from_quaternion(const np::ndarray &translation, const np::ndarray &quaternion, const int filter_radius = FILTER_RADIUS, const float_t filter_threshold = FILTER_THRESHOLD);
        //  ダウンサンプリングする
        void downsampling(float_t leaf_size);

    protected:
        //  Properties

        std::vector<std::vector<std::vector<points_voxel> > > _pointsmap_voxels;    //  三次元地図を格納したVoxelGrid
        point_xyz<float_t> _voxels_center = {0.0f, 0.0f, 0.0f};                     //  VoxelGridの中心座標
        point_xyz<size_t> _voxels_origin = {0ul, 0ul, 0ul};                         //  VoxelGridの中心のVoxelのインデックス
        point_xyz<float_t> _voxels_min = {0.0f, 0.0f, 0.0f};                        //  VoxelGridの座標の最小値
        point_xyz<float_t> _voxels_max = {0.0f, 0.0f, 0.0f};                        //  VoxelGridの座標の最大値
        point_xyz<size_t> _voxels_len = {1ul, 1ul, 1ul};                            //  VoxelGridの幅
        float_t _voxel_size;                                                        //  Voxelの一辺の長さ

        //  Methods

        //  プロパティの三次元点群地図をVoxelに移動する
        void set_voxelgridmap(const double_t voxel_size);
        //  視錐台(Frustum)を生成するための点群を生成する
        void create_frustum_points(pcl::PointCloud<pcl::PointXYZL> &dst);
        //  Voxelが平面の法線方向(右手系)に存在するか判定する
        bool voxel_frontside(const points_voxel &voxel, const Eigen::Vector3f &normal, const Eigen::Vector3f &point_on_plane);
        //  視錐台に含まれるVoxelを判定し，そのVoxelのインデックスを取得する
        void voxels_include_frustum(const pcl::PointCloud<pcl::PointXYZL> &frustum_points, std::vector<point_xyz<size_t> > &dst_voxel_indexs);
        //  自己位置を用いて視錐台に含まれるVoxelを判定し，そのVoxelから点群を取得する
        void points_include_frustum(const Eigen::Vector3f &translation, const Eigen::Quaternionf &quaternion, pcl::PointCloud<pcl::PointXYZL> &dst);
        //  自己位置を用いて視錐台に含まれるVoxelを判定し，そのVoxelから点群を取得する
        void points_include_frustum(const Eigen::Matrix4f &matrix, pcl::PointCloud<pcl::PointXYZL> &dst);
        //  点群から深度マップを生成する
        void create_depthmap(const std::vector<point_xyz<size_t> > &voxel_indexs, const Eigen::Vector3f &translation, const Eigen::Quaternionf &quaternion, cv::Mat &dst);
        //  点群から深度マップを生成する
        void create_depthmap(const std::vector<point_xyz<size_t> > &voxel_indexs, const Eigen::Matrix4f &matrix, cv::Mat &dst);
        //  点群からSemantic2dを生成する
        void create_semantic2d(const std::vector<point_xyz<size_t> > &voxel_indexs, const Eigen::Vector3f &translation, const Eigen::Quaternionf &quaternion, cv::Mat &dst, cv::Mat &depth);
        //  点群からSemantic2dを生成する
        void create_semantic2d(const std::vector<point_xyz<size_t> > &voxel_indexs, const Eigen::Matrix4f &matrix, cv::Mat &dst, cv::Mat &depth);
};

boost::shared_ptr<VoxelGridMap> VoxelGridMap_init(bool quite);

#endif

}   //  namespace pointsmap
