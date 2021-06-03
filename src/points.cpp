#include "points.hpp"

namespace pointsmap {

Points::Points(bool quiet)
:   PointsBase(quiet)
{
    //  pass
}

//  三次元点群地図を格納する
void Points::add_points(const pcl::PointCloud<pcl::PointXYZL> &in)
{
    this->_points += in;
    if (this->_quiet == false) {
        std::cout << "Add  : " << in.points.size() << " points" << std::endl;
    }
}

//  PCDファイルから点群を読み込み，追加する
void Points::add_pointsfile(const std::string &path)
{
    pcl::PointCloud<pcl::PointXYZL> points_map;
    if (pcl::io::loadPCDFile(path, points_map) == -1) {
        throw std::runtime_error("load failed \"" + path + "\"");
    }
    else {
        if (this->_quiet == false) std::cerr << "Load : \"" << path << "\"" << std::endl;
    }
    this->add_points(points_map);
}

//  PCDファイルから点群を読み込み，追加する
void Points::add_pointsfile_python(const std::string &path)
{
    this->add_pointsfile(path);
}

//  複数のPCDファイルから点群を読み込み，追加する
void Points::add_pointsfiles(const std::vector<std::string> &paths)
{
    size_t path_list_len = paths.size();

    pcl::PointCloud<pcl::PointXYZL> part;

    for (size_t i = 0ul; i < path_list_len; i++) {
        if (pcl::io::loadPCDFile(paths[i], part) == -1) {
            std::cerr << "load failed \"" << paths[i] << "\"" << std::endl;
        }
        else{
            if (this->_quiet == false) std::cerr << "load " << part.points.size() << " points. (\"" << paths[i] << "\")" << std::endl;
            this->add_points(part);
        }
    }
}

//  複数のPCDファイルから点群を読み込み，追加する
void Points::add_pointsfiles_python(const py::list &paths)
{
    size_t path_list_len = py::len(paths);
    bool warn_no_str = false;

    std::vector<std::string> paths_vector;

    for (size_t i = 0ul; i < path_list_len; i++) {
        py::object obj = paths[i];
        if (py::extract<std::string>(obj).check() == false){
            warn_no_str = true;
            continue;
        }
        paths_vector.push_back(py::extract<std::string>(obj));
    }

    if (warn_no_str == true){
        std::cerr << "WARNING: List elements must be of type \"str\"." << std::endl;
    }

    this->add_pointsfiles(paths_vector);
}

//  np::ndarrayから点群を読み込み，追加する
void Points::add_points_from_numpy(const np::ndarray &points_map)
{
    pcl::PointCloud<pcl::PointXYZL> points_map_pcl;
    nppoints2pointcloud(points_map, points_map_pcl);
    this->add_points(points_map_pcl);
}

//  semantic3dの点群データを読み込み，追加する
void Points::add_semanticpoints_from_numpy(const np::ndarray &points, const np::ndarray &semantic1d)
{
    pcl::PointCloud<pcl::PointXYZL> points_map_pcl;
    npsemantic3d2pointcloud(points, semantic1d, points_map_pcl);
    this->add_points(points_map_pcl);
}

//  変換行列から深度マップを生成する
np::ndarray Points::create_depthmap_from_matrix(const np::ndarray &matrix_4x4, const int filter_radius, const float_t filter_threshold)
{
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    ndarray2matrix(matrix_4x4, transform);

    invert_transform(transform, transform);

    pcl::PointCloud<pcl::PointXYZL> points_camera;
    this->transformPointCloud(this->_points, points_camera, transform, true);
    cv::Mat depth;
    this->create_depthmap(points_camera, depth);

    if (filter_radius > 0) {
        cv::Mat filterd_depth;
        this->depth_visibility_filter(depth, filterd_depth, filter_threshold, filter_radius);
        return cvmat2ndarray(filterd_depth);
    }
    else {
        return cvmat2ndarray(depth);
    }
}

//  並進ベクトルとクォータニオンから深度マップを生成する
np::ndarray Points::create_depthmap_from_quaternion(const np::ndarray &translation, const np::ndarray &quaternion, const int filter_radius, const float_t filter_threshold)
{
    Eigen::Vector3f translation_eigen;
    Eigen::Quaternionf quaternion_eigen;

    ndarray2translation(translation, translation_eigen);
    ndarray2quaternion(quaternion, quaternion_eigen);

    invert_transform(translation_eigen, quaternion_eigen, translation_eigen, quaternion_eigen);

    pcl::PointCloud<pcl::PointXYZL> points_camera;
    this->transformPointCloud(this->_points, points_camera, translation_eigen, quaternion_eigen, true);
    cv::Mat depth;
    this->create_depthmap(points_camera, depth);

    if (filter_radius > 0) {
        cv::Mat filterd_depth;
        this->depth_visibility_filter(depth, filterd_depth, filter_threshold, filter_radius);
        return cvmat2ndarray(filterd_depth);
    }
    else {
        return cvmat2ndarray(depth);
    }
}

//  変換行列からsemantic2dを生成する
np::ndarray Points::create_semantic2d_from_matrix(const np::ndarray &matrix_4x4, const int filter_radius, const float_t filter_threshold)
{
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    ndarray2matrix(matrix_4x4, transform);

    invert_transform(transform, transform);

    pcl::PointCloud<pcl::PointXYZL> points_camera;
    this->transformPointCloud(this->_points, points_camera, transform, true);
    cv::Mat semantic2d, depth;
    this->create_semantic2d(points_camera, semantic2d, depth);

    if (filter_radius > 0) this->semantic2d_visibility_filter(depth, semantic2d, filter_threshold, filter_radius);
    return cvmat2ndarray(semantic2d);
}

//  並進ベクトルとクォータニオンからsemantic2dを生成する
np::ndarray Points::create_semantic2d_from_quaternion(const np::ndarray &translation, const np::ndarray &quaternion, const int filter_radius, const float_t filter_threshold)
{
    Eigen::Vector3f translation_eigen;
    Eigen::Quaternionf quaternion_eigen;

    ndarray2translation(translation, translation_eigen);
    ndarray2quaternion(quaternion, quaternion_eigen);

    invert_transform(translation_eigen, quaternion_eigen, translation_eigen, quaternion_eigen);

    pcl::PointCloud<pcl::PointXYZL> points_camera;
    this->transformPointCloud(this->_points, points_camera, translation_eigen, quaternion_eigen, true);
    cv::Mat semantic2d, depth;
    this->create_semantic2d(points_camera, semantic2d, depth);

    if (filter_radius > 0) this->semantic2d_visibility_filter(depth, semantic2d, filter_threshold, filter_radius);
    return cvmat2ndarray(semantic2d);
}

//  深度マップを点群に変換して入力する
void Points::set_depthmap_matrix(const np::ndarray &depthmap, const np::ndarray &matrix_4x4)
{
    cv::Mat cv_depth;
    ndarray2cvmat(depthmap, cv_depth);

    pcl::PointCloud<pcl::PointXYZL> points;
    this->create_points_from_depthmap(cv_depth, points);

    Eigen::Matrix4f transform;
    ndarray2matrix(matrix_4x4, transform);

    transformPointCloud(points, this->_points, transform);
}

//  深度マップを点群に変換して入力する
void Points::set_depthmap_quaternion(const np::ndarray &depthmap, const np::ndarray &translation, const np::ndarray &quaternion)
{
    cv::Mat cv_depth;
    ndarray2cvmat(depthmap, cv_depth);

    pcl::PointCloud<pcl::PointXYZL> points;
    this->create_points_from_depthmap(cv_depth, points);

    Eigen::Vector3f translation_eigen;
    ndarray2translation(translation, translation_eigen);
    Eigen::Quaternionf quaternion_eigen;
    ndarray2quaternion(quaternion, quaternion_eigen);

    transformPointCloud(points, this->_points, translation_eigen, quaternion_eigen);
}

//  深度マップとSemantic2Dラベルを点群に変換して入力する
void Points::set_depthmap_semantic2d_matrix(const np::ndarray &depthmap, const np::ndarray &semantic2d, const np::ndarray &matrix_4x4)
{
    cv::Mat cv_depth, cv_semantic2d;
    ndarray2cvmat(depthmap, cv_depth);
    ndarray2cvmat(semantic2d, cv_semantic2d);

    pcl::PointCloud<pcl::PointXYZL> points;
    this->create_points_from_depthmap_semantic2d(cv_depth, cv_semantic2d, points);

    Eigen::Matrix4f transform;
    ndarray2matrix(matrix_4x4, transform);

    transformPointCloud(points, this->_points, transform);
}

//  深度マップとSemantic2Dラベルを点群に変換して入力する
void Points::set_depthmap_semantic2d_quaternion(const np::ndarray &depthmap, const np::ndarray &semantic2d, const np::ndarray &translation, const np::ndarray &quaternion)
{
    cv::Mat cv_depth, cv_semantic2d;
    ndarray2cvmat(depthmap, cv_depth);
    ndarray2cvmat(semantic2d, cv_semantic2d);

    pcl::PointCloud<pcl::PointXYZL> points;
    this->create_points_from_depthmap_semantic2d(cv_depth, cv_semantic2d, points);

    Eigen::Vector3f translation_eigen;
    ndarray2translation(translation, translation_eigen);
    Eigen::Quaternionf quaternion_eigen;
    ndarray2quaternion(quaternion, quaternion_eigen);

    transformPointCloud(points, this->_points, translation_eigen, quaternion_eigen);
}

//  格納されている点群を座標変換する
void Points::transform_matrix(const np::ndarray &matrix_4x4)
{
    Eigen::Matrix4f transform;
    ndarray2matrix(matrix_4x4, transform);

    this->transformPointCloud(this->_points, this->_points, transform, false);
}

//  格納されている点群を座標変換する
void Points::transform_quaternion(const np::ndarray &translation, const np::ndarray &quaternion)
{
    Eigen::Vector3f translation_eigen;
    ndarray2translation(translation, translation_eigen);
    Eigen::Quaternionf quaternion_eigen;
    ndarray2quaternion(quaternion, quaternion_eigen);

    this->transformPointCloud(this->_points, this->_points, translation_eigen, quaternion_eigen, false);
}

//  ダウンサンプリングする
void Points::downsampling(float_t leaf_size)
{
    pcl::PointCloud<pcl::PointXYZL> tmp_points;
    this->voxelGridFilter(this->_points, tmp_points, leaf_size);
    this->_points.swap(tmp_points);
}

boost::shared_ptr<Points> Points_init(bool quite)
{
    boost::shared_ptr<Points> pm(new Points(quite));
    return pm;
}

}   //  pointsmap