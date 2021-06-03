#include "common.hpp"
#include "base/intrinsic_base.hpp"

namespace pointsmap {

#ifndef POINTSMAP_BASE_DEPTH_HPP_INCLUDE

#define POINTSMAP_BASE_DEPTH_HPP_INCLUDE

class DepthBase : public IntrinsicBase
{
    public:
        //  Constructor
        DepthBase();

        //  Methods

        //  出力する深度の範囲を設定する
        void set_depth_range(const float_t min, const float_t max);
        //  出力する深度の範囲を設定する
        void set_depth_range_python(const py::tuple &depth_range);
        //  設定した深度の範囲を取得する
        Eigen::Vector2f get_depth_range();
        //  設定した深度の範囲を取得する
        py::tuple get_depth_range_python();

    protected:
        //  Properties
        
        float_t _depth_max = INFINITY;  //  描画する深度の最大値
        float_t _depth_min = 0.0f;      //  描画する深度の最小値

        //  Methods

        //  点群から深度マップを生成する
        void create_depthmap(const pcl::PointCloud<pcl::PointXYZL> &src, cv::Mat &dst);
        //  点群からSemantic2Dを生成する
        void create_semantic2d(const pcl::PointCloud<pcl::PointXYZL> &src, cv::Mat &dst, cv::Mat &depth);
        //  深度マップから三次元点群を生成する
        void create_points_from_depthmap(const cv::Mat &depthmap, pcl::PointCloud<pcl::PointXYZL> &dst_points);
        //  深度マップとSemantic2Dからラベル付き三次元点群を生成する
        void create_points_from_depthmap_semantic2d(const cv::Mat &depthmap, const cv::Mat &semantic2d, pcl::PointCloud<pcl::PointXYZL> &dst_points);
        //  Semantic2Dを用いて三次元点群にラベルを付与する
        void set_semanticlabel_from_semantic2d(const cv::Mat &semantic2d, pcl::PointCloud<pcl::PointXYZL> &target_points);
        //  本来見えない位置にある点をDepthから除去する
        void depth_visibility_filter(const cv::Mat &src, cv::Mat &dst, const float_t threshold = 3.0f, const int radius = 5);
        //  本来見えない位置にある点をSemantic2Dから除去する
        void semantic2d_visibility_filter(const cv::Mat &src_depth, cv::Mat &target_semantic2d, const float_t threshold = 3.0f, const int radius = 5);

};

#endif

}