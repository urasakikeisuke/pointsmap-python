#include "base/points_base.hpp"

namespace pointsmap {

//  Constructor
PointsBase::PointsBase(bool quiet)
:   DepthBase()
{
    this->_quiet = quiet;
}

//  三次元点群地図を格納する
void PointsBase::set_points(const pcl::PointCloud<pcl::PointXYZL> &in)
{
    this->_points = in;
    if (this->_quiet == false) {
        std::cout << "Load : " << in.points.size() << " points" << std::endl;
    }
}

//  PCDファイルから点群を読み込み，格納する
void PointsBase::set_pointsfile(const std::string &path)
{
    pcl::PointCloud<pcl::PointXYZL> points_map;
    if (pcl::io::loadPCDFile(path, points_map) == -1) {
        throw std::runtime_error("load failed \"" + path + "\"");
    }
    else {
        if (this->_quiet == false) std::cerr << "Load : \"" << path << "\"" << std::endl;
    }
    this->set_points(points_map);
}

//  PCDファイルから点群を読み込み，格納する
void PointsBase::set_pointsfile_python(const std::string &path)
{
    this->set_pointsfile(path);
}

//  複数のPCDファイルから点群を読み込み，格納する
void PointsBase::set_pointsfiles(const std::vector<std::string> &paths)
{
    size_t path_list_len = paths.size();

    pcl::PointCloud<pcl::PointXYZL> points_map, part;

    for (size_t i = 0ul; i < path_list_len; i++) {
        if (pcl::io::loadPCDFile(paths[i], part) == -1) {
            std::cerr << "load failed \"" << paths[i] << "\"" << std::endl;
        }
        else{
            if (this->_quiet == false) std::cerr << "load " << part.points.size() << " points. (\"" << paths[i] << "\")" << std::endl;
            points_map += part;
        }
    }
    this->set_points(points_map);
}

//  複数のPCDファイルから点群を読み込み，格納する
void PointsBase::set_pointsfiles_python(const py::list &paths)
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

    this->set_pointsfiles(paths_vector);
}

//  np::ndarrayから点群を読み込み，格納する
void PointsBase::set_points_from_numpy(const np::ndarray &points_map)
{
    pcl::PointCloud<pcl::PointXYZL> points_map_pcl;
    nppoints2pointcloud(points_map, points_map_pcl);
    this->set_points(points_map_pcl);
}

//  semantic3dの点群データを読み込み，格納する
void PointsBase::set_semanticpoints_from_numpy(const np::ndarray &points, const np::ndarray &semantic1d)
{
    pcl::PointCloud<pcl::PointXYZL> points_map_pcl;
    npsemantic3d2pointcloud(points, semantic1d, points_map_pcl);
    this->set_points(points_map_pcl);
}

//  格納された点群を取り出す
void PointsBase::get_points(pcl::PointCloud<pcl::PointXYZL> &out)
{
    out = this->_points;
}

//  格納された点群を取り出す
np::ndarray PointsBase::get_points_python()
{
    pcl::PointCloud<pcl::PointXYZL> tmp;
    this->get_points(tmp);
    return pointcloud2nppoints(tmp);
}

//  格納された点群を取り出す
py::tuple PointsBase::get_semanticpoints_python()
{
    pcl::PointCloud<pcl::PointXYZL> tmp;
    this->get_points(tmp);
    return pointcloud2npsemantic3d(tmp);
}

//  点群をpclファイルに保存する
void PointsBase::save_pcd(const std::string &path)
{
    pcl::PointCloud<pcl::PointXYZL> tmp;
    this->get_points(tmp);
    try {
        pcl::io::savePCDFileBinary(path, tmp);
    }
    catch (pcl::IOException &ex) {
        std::cerr << "PCD write Error: " << ex.what() << std::endl;
    }
}

//  点群の座標変換を行い，距離によるフィルタ処理を行う
void PointsBase::transformPointCloud(const pcl::PointCloud<pcl::PointXYZL> &src, pcl::PointCloud<pcl::PointXYZL> &dst, const Eigen::Vector3f &translation, const Eigen::Quaternionf &quaternion, const bool filter)
{
    transform_pointcloud(src, dst, translation, quaternion);
    if (filter == true) depth_filter(dst, dst, this->_depth_min, this->_depth_max);
}

//  点群の座標変換を行い，距離によるフィルタ処理を行う
void PointsBase::transformPointCloud(const pcl::PointCloud<pcl::PointXYZL> &src, pcl::PointCloud<pcl::PointXYZL> &dst, const Eigen::Matrix4f &matrix, const bool filter)
{
    transform_pointcloud(src, dst, matrix);
    if (filter == true) depth_filter(dst, dst, this->_depth_min, this->_depth_max);
}

//  点群をダウンサンプリングする
void PointsBase::voxelGridFilter(const pcl::PointCloud<pcl::PointXYZL> &src, pcl::PointCloud<pcl::PointXYZL> &dst, float_t leaf_size)
{
    pcl::VoxelGrid<pcl::PointXYZL> vgf;
    vgf.setInputCloud(src.makeShared());
    vgf.setLeafSize(leaf_size, leaf_size, leaf_size);
    vgf.filter(dst);
}

boost::shared_ptr<PointsBase> PointsBase_init(bool quite)
{
    boost::shared_ptr<PointsBase> pb(new PointsBase(quite));
    return pb;
}

}   //  pointsmap
