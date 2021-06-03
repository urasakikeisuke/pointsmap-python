#include "common.hpp"
#include "base/points_base.hpp"

namespace pointsmap {

#ifndef POINTSMAP_POINTSMAP_HPP_INCLUDE

#define POINTSMAP_POINTSMAP_HPP_INCLUDE

class Points : public PointsBase
{
    public:
        //  Constructor
        Points(bool quiet);

        //  Methods

        //  三次元点群地図を格納する
        void add_points(const pcl::PointCloud<pcl::PointXYZL> &in);
        //  PCDファイルから点群を読み込み，追加する
        void add_pointsfile(const std::string &path);
        //  PCDファイルから点群を読み込み，追加する
        void add_pointsfile_python(const std::string &path);
        //  複数のPCDファイルから点群を読み込み，追加する
        void add_pointsfiles(const std::vector<std::string> &paths);
        //  複数のPCDファイルから点群を読み込み，追加する
        void add_pointsfiles_python(const py::list &paths);
        //  np::ndarrayから点群を読み込み，追加する
        void add_points_from_numpy(const np::ndarray &points_map);
        //  semantic3dの点群データを読み込み，追加する
        void add_semanticpoints_from_numpy(const np::ndarray &points, const np::ndarray &semantic1d);
        //  変換行列から深度マップを生成する
        np::ndarray create_depthmap_from_matrix(const np::ndarray &matrix_4x4, const int filter_radius = FILTER_RADIUS, const float_t filter_threshold = FILTER_THRESHOLD);
        //  並進ベクトルとクォータニオンから深度マップを生成する
        np::ndarray create_depthmap_from_quaternion(const np::ndarray &translation, const np::ndarray &quaternion, const int filter_radius = FILTER_RADIUS, const float_t filter_threshold = FILTER_THRESHOLD);
        //  変換行列からsemantic2dを生成する
        np::ndarray create_semantic2d_from_matrix(const np::ndarray &matrix_4x4, const int filter_radius = FILTER_RADIUS, const float_t filter_threshold = FILTER_THRESHOLD);
        //  並進ベクトルとクォータニオンからsemantic2dを生成する
        np::ndarray create_semantic2d_from_quaternion(const np::ndarray &translation, const np::ndarray &quaternion, const int filter_radius = FILTER_RADIUS, const float_t filter_threshold = FILTER_THRESHOLD);
        //  深度マップを点群に変換して入力する
        void set_depthmap_matrix(const np::ndarray &depthmap, const np::ndarray &matrix_4x4);
        //  深度マップを点群に変換して入力する
        void set_depthmap_quaternion(const np::ndarray &depthmap, const np::ndarray &translation, const np::ndarray &quaternion);
        //  深度マップとSemantic2Dラベルを点群に変換して入力する
        void set_depthmap_semantic2d_matrix(const np::ndarray &depthmap, const np::ndarray &semantic2d, const np::ndarray &matrix_4x4);
        //  深度マップとSemantic2Dラベルを点群に変換して入力する
        void set_depthmap_semantic2d_quaternion(const np::ndarray &depthmap, const np::ndarray &semantic2d, const np::ndarray &translation, const np::ndarray &quaternion);
        //  格納されている点群を座標変換する
        void transform_matrix(const np::ndarray &matrix_4x4);
        //  格納されている点群を座標変換する
        void transform_quaternion(const np::ndarray &translation, const np::ndarray &quaternion);
        //  ダウンサンプリングする
        void downsampling(float_t leaf_size);
};

boost::shared_ptr<Points> Points_init(bool quite);

#endif

}   //  pointsmap