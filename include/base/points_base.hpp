#include "common.hpp"
#include "base/depth_base.hpp"

namespace pointsmap {

#ifndef POINTSMAP_BASE_POINTS_HPP_INCLUDE

#define POINTSMAP_BASE_POINTS_HPP_INCLUDE

class PointsBase : public DepthBase
{
    public:
        //  Constructor
        PointsBase(bool quiet);

        //  Methods

        //  三次元点群地図を格納する
        virtual void set_points(const pcl::PointCloud<pcl::PointXYZL> &in);
        //  PCDファイルから点群を読み込み，格納する
        virtual void set_pointsfile(const std::string &path);
        //  PCDファイルから点群を読み込み，格納する
        virtual void set_pointsfile_python(const std::string &path);
        //  複数のPCDファイルから点群を読み込み，格納する
        virtual void set_pointsfiles(const std::vector<std::string> &paths);
        //  複数のPCDファイルから点群を読み込み，格納する
        virtual void set_pointsfiles_python(const py::list &paths);
        //  np::ndarrayから点群を読み込み，格納する
        virtual void set_points_from_numpy(const np::ndarray &points_map);
        //  semantic3dの点群データを読み込み，格納する
        virtual void set_semanticpoints_from_numpy(const np::ndarray &points, const np::ndarray &semantic1d);
        //  格納された点群を取り出す
        virtual void get_points(pcl::PointCloud<pcl::PointXYZL> &out);
        //  格納された点群を取り出す
        virtual np::ndarray get_points_python();
        //  格納された点群を取り出す
        virtual py::tuple get_semanticpoints_python();

    protected:
        //  Properties

        bool _quiet = false;                        //  true: 標準出力へ出力しない
        pcl::PointCloud<pcl::PointXYZL> _points;    //  三次元点群

        //  Methods

        //  点群の座標変換を行い，距離によるフィルタ処理を行う
        void transformPointCloud(const pcl::PointCloud<pcl::PointXYZL> &src, pcl::PointCloud<pcl::PointXYZL> &dst, const Eigen::Vector3f &translation, const Eigen::Quaternionf &quaternion, const bool filter = false);
        //  点群の座標変換を行い，距離によるフィルタ処理を行う
        void transformPointCloud(const pcl::PointCloud<pcl::PointXYZL> &src, pcl::PointCloud<pcl::PointXYZL> &dst, const Eigen::Matrix4f &matrix, const bool filter = false);
        //  点群をダウンサンプリングする
        void voxelGridFilter(const pcl::PointCloud<pcl::PointXYZL> &src, pcl::PointCloud<pcl::PointXYZL> &dst, float_t leaf_size);

};

boost::shared_ptr<PointsBase> PointsBase_init(bool quite);

#endif

}   //  pointsmap
