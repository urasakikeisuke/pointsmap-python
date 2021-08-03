#include "voxelgridmap.hpp"

namespace pointsmap {

//  Constructor
VoxelGridMap::VoxelGridMap(bool quiet)
: PointsBase(quiet)
{
    this->_quiet = quiet;

    #ifdef _OPENMP
        if (this->_quiet == false) {
            std::cout << "OpenMP : ON" << std::endl;
        }
    #else
        if (this->_quiet == false) {
            std::cout << "OpenMP : OFF" << std::endl;
        }
    #endif
}

//  三次元点群地図をボクセルに格納する
void VoxelGridMap::set_points(const pcl::PointCloud<pcl::PointXYZL> &in, const double_t voxel_size)
{
    PointsBase::set_points(in);
    this->set_voxelgridmap(voxel_size);
}

//  PCDファイルから点群を読み込み，ボクセルに格納する
void VoxelGridMap::set_pointsfile(const std::string &path, const double_t voxel_size)
{
    PointsBase::set_pointsfile(path);
    this->set_voxelgridmap(voxel_size);
}

//  PCDファイルから点群を読み込み，ボクセルに格納する
void VoxelGridMap::set_pointsfile_python(const std::string &path, const double_t voxel_size)
{
    PointsBase::set_pointsfile_python(path);
    this->set_voxelgridmap(voxel_size);
}

//  複数のPCDファイルから点群を読み込み，ボクセルに格納する
void VoxelGridMap::set_pointsfiles(const std::vector<std::string> &paths, const double_t voxel_size)
{
    PointsBase::set_pointsfiles(paths);
    this->set_voxelgridmap(voxel_size);
}

//  複数のPCDファイルから点群を読み込み，ボクセルに格納する
void VoxelGridMap::set_pointsfiles_python(const py::list &paths, const double_t voxel_size)
{
    PointsBase::set_pointsfiles_python(paths);
    this->set_voxelgridmap(voxel_size);
}

//  np::ndarrayから点群を読み込み，ボクセルに格納する
void VoxelGridMap::set_points_from_numpy(const np::ndarray &points, const double_t voxel_size)
{
    PointsBase::set_points_from_numpy(points);
    this->set_voxelgridmap(voxel_size);
}

//  semantic3dの点群データを読み込み，ボクセルに格納する
void VoxelGridMap::set_semanticpoints_from_numpy(const np::ndarray &points, const np::ndarray &semantic1d, const double_t voxel_size)
{
    PointsBase::set_semanticpoints_from_numpy(points, semantic1d);
    this->set_voxelgridmap(voxel_size);
}

//  ボクセルに格納された点群を取り出す
void VoxelGridMap::get_points(pcl::PointCloud<pcl::PointXYZL> &out)
{
    out.clear();

    for (size_t z = 0ul; z < this->_voxels_len.z; z++) for (size_t y = 0ul; y < this->_voxels_len.y; y++) for (size_t x = 0ul; x < this->_voxels_len.x; x++) {
        out += this->_pointsmap_voxels[z][y][x].points;
    }
}

//  VoxelGridMapを格納する
void VoxelGridMap::set_voxelgridmap(const std::vector<std::vector<std::vector<points_voxel> > > &vgm, const float_t voxel_size, const point_xyz<float_t> &voxels_min, const point_xyz<float_t> &voxels_max, const point_xyz<float_t> &voxels_center, const point_xyz<size_t> &voxels_origin)
{
    this->_pointsmap_voxels = vgm;
    this->_voxels_len = {
        .x = vgm[0][0].size(),
        .y = vgm[0].size(),
        .z = vgm.size()
    };
    this->_voxel_size = voxel_size;
    this->_voxels_min = voxels_min;
    this->_voxels_max = voxels_max;
    this->_voxels_center = voxels_center;
    this->_voxels_origin = voxels_origin;
}

//  VoxelGridMapを格納する (for Python)
void VoxelGridMap::set_voxelgridmap_python(const np::ndarray &vgm, const double_t voxel_size, const py::tuple &voxels_min, const py::tuple &voxels_max, const py::tuple &voxels_center, const py::tuple &voxels_origin)
{
    if (py::len(voxels_min) != 3ul) {
        throw std::invalid_argument("\"len(voxels_min)\" must be 3.");
    }
    if (py::len(voxels_max) != 3ul) {
        throw std::invalid_argument("\"len(voxels_max)\" must be 3.");
    }
    if (py::len(voxels_center) != 3ul) {
        throw std::invalid_argument("\"len(voxels_center)\" must be 3.");
    }
    if (py::len(voxels_origin) != 3ul) {
        throw std::invalid_argument("\"len(voxels_origin)\" must be 3.");
    }

    point_xyz<float_t> voxels_min_p = {
        .x = py::extract<float_t>(voxels_min[2]),
        .y = py::extract<float_t>(voxels_min[1]),
        .z = py::extract<float_t>(voxels_min[0])
    };
    point_xyz<float_t> voxels_max_p = {
        .x = py::extract<float_t>(voxels_max[2]),
        .y = py::extract<float_t>(voxels_max[1]),
        .z = py::extract<float_t>(voxels_max[0])
    };
    point_xyz<float_t> voxels_center_p = {
        .x = py::extract<float_t>(voxels_center[2]),
        .y = py::extract<float_t>(voxels_center[1]),
        .z = py::extract<float_t>(voxels_center[0])
    };
    point_xyz<size_t> voxels_origin_p = {
        .x = py::extract<size_t>(voxels_origin[2]),
        .y = py::extract<size_t>(voxels_origin[1]),
        .z = py::extract<size_t>(voxels_origin[0])
    };

    auto vgm_shape = vgm.get_shape();
    auto vgm_dtype = vgm.get_dtype();

    point_xyz<size_t> voxels_len = {.x = vgm_shape[2], .y = vgm_shape[1], .z = vgm_shape[0]};
    std::vector<std::vector<std::vector<points_voxel> > > pointsmap_voxels(voxels_len.z, std::vector<std::vector<points_voxel> >(voxels_len.y, std::vector<points_voxel>(voxels_len.x)));

    for (size_t z = 0ul; z < voxels_len.z; z++) for (size_t y = 0ul; y < voxels_len.y; y++) for (size_t x = 0ul; x < voxels_len.x; x++) {
        points_voxel voxel;
        voxel.min.x = voxels_min_p.x + voxel_size * x;
        voxel.min.y = voxels_min_p.y + voxel_size * y;
        voxel.min.z = voxels_min_p.z + voxel_size * z;
        voxel.max.x = voxels_min_p.x + voxel_size * (x + 1ul);
        voxel.max.y = voxels_min_p.y + voxel_size * (y + 1ul);
        voxel.max.z = voxels_min_p.z + voxel_size * (z + 1ul);
        compound2pointcloud(np::from_object(vgm[z][y][x]), voxel.points);
        pointsmap_voxels[z][y][x] = voxel;
    }

    this->set_voxelgridmap(
        pointsmap_voxels, static_cast<float_t>(voxel_size),
        voxels_min_p, voxels_max_p,
        voxels_center_p, voxels_origin_p
    );
}

//  空のVoxelGridMapを作成する
void VoxelGridMap::set_empty_voxelgridmap(const point_xyz<size_t> &voxels_len, const float_t voxel_size, const point_xyz<float_t> &voxels_min, const point_xyz<float_t> &voxels_max, const point_xyz<float_t> &voxels_center, const point_xyz<size_t> &voxels_origin)
{
    std::vector<std::vector<std::vector<points_voxel> > > pointsmap_voxels(voxels_len.z, std::vector<std::vector<points_voxel> >(voxels_len.y, std::vector<points_voxel>(voxels_len.x)));

    for (size_t z = 0ul; z < voxels_len.z; z++) for (size_t y = 0ul; y < voxels_len.y; y++) for (size_t x = 0ul; x < voxels_len.x; x++) {
        points_voxel voxel;
        voxel.min.x = voxels_min.x + voxel_size * x;
        voxel.min.y = voxels_min.y + voxel_size * y;
        voxel.min.z = voxels_min.z + voxel_size * z;
        voxel.max.x = voxels_min.x + voxel_size * (x + 1ul);
        voxel.max.y = voxels_min.y + voxel_size * (y + 1ul);
        voxel.max.z = voxels_min.z + voxel_size * (z + 1ul);
        pointsmap_voxels[z][y][x] = voxel;
    }

    this->set_voxelgridmap(pointsmap_voxels, voxel_size, voxels_min, voxels_max, voxels_center, voxels_origin);
}

//  空のVoxelGridMapを作成する
void VoxelGridMap::set_empty_voxelgridmap_python(const py::tuple &voxels_len, const double_t voxel_size, const py::tuple &voxels_min, const py::tuple &voxels_max, const py::tuple &voxels_center, const py::tuple &voxels_origin)
{
    if (py::len(voxels_len) != 3ul) {
        throw std::invalid_argument("\"len(voxels_len)\" must be 3.");
    }
    if (py::len(voxels_min) != 3ul) {
        throw std::invalid_argument("\"len(voxels_min)\" must be 3.");
    }
    if (py::len(voxels_max) != 3ul) {
        throw std::invalid_argument("\"len(voxels_max)\" must be 3.");
    }
    if (py::len(voxels_center) != 3ul) {
        throw std::invalid_argument("\"len(voxels_center)\" must be 3.");
    }
    if (py::len(voxels_origin) != 3ul) {
        throw std::invalid_argument("\"len(voxels_origin)\" must be 3.");
    }

    point_xyz<size_t> voxels_len_p = {
        .x = py::extract<size_t>(voxels_len[2]),
        .y = py::extract<size_t>(voxels_len[1]),
        .z = py::extract<size_t>(voxels_len[0])
    };
    point_xyz<float_t> voxels_min_p = {
        .x = py::extract<float_t>(voxels_min[2]),
        .y = py::extract<float_t>(voxels_min[1]),
        .z = py::extract<float_t>(voxels_min[0])
    };
    point_xyz<float_t> voxels_max_p = {
        .x = py::extract<float_t>(voxels_max[2]),
        .y = py::extract<float_t>(voxels_max[1]),
        .z = py::extract<float_t>(voxels_max[0])
    };
    point_xyz<float_t> voxels_center_p = {
        .x = py::extract<float_t>(voxels_center[2]),
        .y = py::extract<float_t>(voxels_center[1]),
        .z = py::extract<float_t>(voxels_center[0])
    };
    point_xyz<size_t> voxels_origin_p = {
        .x = py::extract<size_t>(voxels_origin[2]),
        .y = py::extract<size_t>(voxels_origin[1]),
        .z = py::extract<size_t>(voxels_origin[0])
    };

    this->set_empty_voxelgridmap(voxels_len_p, static_cast<float_t>(voxel_size), voxels_min_p, voxels_max_p, voxels_center_p, voxels_origin_p);
}

//  VoxelGridMapを取得する
void VoxelGridMap::get_voxelgridmap(std::vector<std::vector<std::vector<points_voxel> > > &vgm)
{
    vgm = this->_pointsmap_voxels;
}

//  VoxelGridMapを取得する (for Python)
np::ndarray VoxelGridMap::get_voxelgridmap_python(const bool label)
{
    const Py_intptr_t vgm_shape[3] = {this->_voxels_len.z, this->_voxels_len.y, this->_voxels_len.x};
    const np::dtype vgm_dtype = np::dtype(py::object("O"));
    np::ndarray out = np::zeros(3, vgm_shape, vgm_dtype);

    for (size_t z = 0ul; z < this->_voxels_len.z; z++) for (size_t y = 0ul; y < this->_voxels_len.y; y++) for (size_t x = 0ul; x < this->_voxels_len.x; x++) {
        out[z][y][x] = pointcloud2compound(this->_pointsmap_voxels[z][y][x].points, label);
    }

    return out;
}

//  Voxelのサイズを取得する
float_t VoxelGridMap::get_voxel_size()
{
    return this->_voxel_size;
}

//  Voxelのサイズを取得する (for Python)
double_t VoxelGridMap::get_voxel_size_python()
{
    return static_cast<double_t>(this->get_voxel_size());
}

//  VoxelGridMapの範囲の最小値を取得する
void VoxelGridMap::get_voxels_min(point_xyz<float_t> &voxels_min)
{
    voxels_min = this->_voxels_min;
}

//  VoxelGridMapの範囲の最小値を取得する (for Python)
py::tuple VoxelGridMap::get_voxels_min_python()
{
    point_xyz<float_t> voxels_min;
    this->get_voxels_min(voxels_min);
    return py::make_tuple(voxels_min.z, voxels_min.y, voxels_min.x);
}

//  VoxelGridMapの範囲の最大値を取得する
void VoxelGridMap::get_voxels_max(point_xyz<float_t> &voxels_max)
{
    voxels_max = this->_voxels_max;
}

//  VoxelGridMapの範囲の最大値を取得する (for Python)
py::tuple VoxelGridMap::get_voxels_max_python()
{
    point_xyz<float_t> voxels_max;
    this->get_voxels_max(voxels_max);
    return py::make_tuple(voxels_max.z, voxels_max.y, voxels_max.x);
}

//  VoxelGridMapの中心座標を取得する
void VoxelGridMap::get_voxels_center(point_xyz<float_t> &voxels_center)
{
    voxels_center = this->_voxels_center;
}

//  VoxelGridMapの中心座標を取得する (for Python)
py::tuple VoxelGridMap::get_voxels_center_python()
{
    point_xyz<float_t> voxels_center;
    this->get_voxels_center(voxels_center);
    return py::make_tuple(voxels_center.z, voxels_center.y, voxels_center.x);
}

//  VoxelGridMapの中心のVoxelのインデックスを取得する
void VoxelGridMap::get_voxels_origin(point_xyz<size_t> &voxels_origin)
{
    voxels_origin = this->_voxels_origin;
}

//  VoxelGridMapの中心のVoxelのインデックスを取得する (for Python)
py::tuple VoxelGridMap::get_voxels_origin_python()
{
    point_xyz<size_t> voxels_origin;
    this->get_voxels_origin(voxels_origin);
    return py::make_tuple(voxels_origin.z, voxels_origin.y, voxels_origin.x);
}

//  複数のVoxelを格納する (for Python)
void VoxelGridMap::set_voxels_python(const py::tuple &indexs, const np::ndarray &voxels)
{
    np::ndarray indexs_x = np::from_object(indexs[2]);
    np::ndarray indexs_y = np::from_object(indexs[1]);
    np::ndarray indexs_z = np::from_object(indexs[0]);
    if (indexs_x.get_nd() != 1 || indexs_y.get_nd() != 1 || indexs_z.get_nd() != 1) {
        throw std::invalid_argument("\"indexs.ndim\" must be 1.");
    }
    auto strides = indexs_x.get_strides();

    if (voxels.get_nd() != 1) {
        throw std::invalid_argument("\"voxels.ndim\" must be 1.");
    }

    size_t len = voxels.shape(0);
    for (size_t i = 0ul; i < len; i++) {
        int64_t x = *reinterpret_cast<int64_t *>(indexs_x.get_data() + strides[0] * i);
        int64_t y = *reinterpret_cast<int64_t *>(indexs_y.get_data() + strides[0] * i);
        int64_t z = *reinterpret_cast<int64_t *>(indexs_z.get_data() + strides[0] * i);
        compound2pointcloud(np::from_object(voxels[i]), this->_pointsmap_voxels[z][y][x].points);
    }
}

//  変換行列から深度マップを生成する
np::ndarray VoxelGridMap::create_depthmap_from_matrix(const np::ndarray &matrix_4x4, const int filter_radius, const float_t filter_threshold)
{
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    ndarray2matrix(matrix_4x4, transform);

    invert_transform(transform, transform);

    std::vector<point_xyz<size_t> > voxel_indexs;
    this->voxels_include_frustum(transform, voxel_indexs);
    cv::Mat depth;
    this->create_depthmap(voxel_indexs, transform, depth);

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
np::ndarray VoxelGridMap::create_depthmap_from_quaternion(const np::ndarray &translation, const np::ndarray &quaternion, const int filter_radius, const float_t filter_threshold)
{
    Eigen::Vector3f translation_eigen;
    Eigen::Quaternionf quaternion_eigen;

    ndarray2translation(translation, translation_eigen);
    ndarray2quaternion(quaternion, quaternion_eigen);

    invert_transform(translation_eigen, quaternion_eigen, translation_eigen, quaternion_eigen);

    std::vector<point_xyz<size_t> > voxel_indexs;
    this->voxels_include_frustum(translation_eigen, quaternion_eigen, voxel_indexs);
    cv::Mat depth;
    this->create_depthmap(voxel_indexs, translation_eigen, quaternion_eigen, depth);

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
np::ndarray VoxelGridMap::create_semantic2d_from_matrix(const np::ndarray &matrix_4x4, const int filter_radius, const float_t filter_threshold)
{
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    ndarray2matrix(matrix_4x4, transform);

    invert_transform(transform, transform);

    std::vector<point_xyz<size_t> > voxel_indexs;
    this->voxels_include_frustum(transform, voxel_indexs);
    cv::Mat semantic2d, depth;
    this->create_semantic2d(voxel_indexs, transform, semantic2d, depth);

    if (filter_radius > 0) this->semantic2d_visibility_filter(depth, semantic2d, filter_threshold, filter_radius);
    return cvmat2ndarray(semantic2d);
}

//  並進ベクトルとクォータニオンからsemantic2dを生成する
np::ndarray VoxelGridMap::create_semantic2d_from_quaternion(const np::ndarray &translation, const np::ndarray &quaternion, const int filter_radius, const float_t filter_threshold)
{
    Eigen::Vector3f translation_eigen;
    Eigen::Quaternionf quaternion_eigen;

    ndarray2translation(translation, translation_eigen);
    ndarray2quaternion(quaternion, quaternion_eigen);

    invert_transform(translation_eigen, quaternion_eigen, translation_eigen, quaternion_eigen);

    std::vector<point_xyz<size_t> > voxel_indexs;
    this->voxels_include_frustum(translation_eigen, quaternion_eigen, voxel_indexs);
    cv::Mat semantic2d, depth;
    this->create_semantic2d(voxel_indexs, translation_eigen, quaternion_eigen, semantic2d, depth);

    if (filter_radius > 0) this->semantic2d_visibility_filter(depth, semantic2d, filter_threshold, filter_radius);
    return cvmat2ndarray(semantic2d);
}

//  プロパティの三次元点群地図をボクセルに移動する
void VoxelGridMap::set_voxelgridmap(const double_t voxel_size)
{
    this->_voxel_size = static_cast<float_t>(voxel_size);

    pcl::PointXYZL min_pt, max_pt;
    pcl::getMinMax3D(this->_points, min_pt, max_pt);

    this->_voxels_center.x = (min_pt.x + max_pt.x) * 0.5f;
    this->_voxels_center.y = (min_pt.y + max_pt.y) * 0.5f;
    this->_voxels_center.z = (min_pt.z + max_pt.z) * 0.5f;

    point_xyz<long> voxels_minus = {
        static_cast<long>(std::roundf((min_pt.x - this->_voxels_center.x) / this->_voxel_size - 0.5f)),
        static_cast<long>(std::roundf((min_pt.y - this->_voxels_center.y) / this->_voxel_size - 0.5f)),
        static_cast<long>(std::roundf((min_pt.z - this->_voxels_center.z) / this->_voxel_size - 0.5f))
    };

    point_xyz<long> voxels_plus = {
        static_cast<long>(std::roundf((max_pt.x - this->_voxels_center.x) / this->_voxel_size + 0.5f)),
        static_cast<long>(std::roundf((max_pt.y - this->_voxels_center.y) / this->_voxel_size + 0.5f)),
        static_cast<long>(std::roundf((max_pt.z - this->_voxels_center.z) / this->_voxel_size + 0.5f))
    };

    this->_voxels_origin.x = - static_cast<size_t>(voxels_minus.x);
    this->_voxels_origin.y = - static_cast<size_t>(voxels_minus.y);
    this->_voxels_origin.z = - static_cast<size_t>(voxels_minus.z);

    this->_voxels_len.x = static_cast<size_t>(voxels_plus.x - voxels_minus.x);
    this->_voxels_len.y = static_cast<size_t>(voxels_plus.y - voxels_minus.y);
    this->_voxels_len.z = static_cast<size_t>(voxels_plus.z - voxels_minus.z);

    this->_pointsmap_voxels.resize(this->_voxels_len.z, std::vector<std::vector<points_voxel> >(this->_voxels_len.y, std::vector<points_voxel>(this->_voxels_len.x)));

    #ifdef _OPENMP
        #pragma omp parallel
    #endif
    {
        for (size_t z = 0ul; z < this->_voxels_len.z; z++) {
            #ifdef _OPENMP
                #pragma omp for
            #endif
            for (size_t y = 0ul; y < this->_voxels_len.y; y++) for (size_t x = 0ul; x < this->_voxels_len.x; x++) {
                this->_pointsmap_voxels[z][y][x].min.x = static_cast<float_t>(voxels_minus.x + static_cast<long>(x)) * this->_voxel_size + this->_voxels_center.x;
                this->_pointsmap_voxels[z][y][x].max.x = static_cast<float_t>(voxels_minus.x + static_cast<long>(x) + 1l) * this->_voxel_size + this->_voxels_center.x;

                this->_pointsmap_voxels[z][y][x].min.y = static_cast<float_t>(voxels_minus.y + static_cast<long>(y)) * this->_voxel_size + this->_voxels_center.y;
                this->_pointsmap_voxels[z][y][x].max.y = static_cast<float_t>(voxels_minus.y + static_cast<long>(y) + 1l) * this->_voxel_size + this->_voxels_center.y;

                this->_pointsmap_voxels[z][y][x].min.z = static_cast<float_t>(voxels_minus.z + static_cast<long>(z)) * this->_voxel_size + this->_voxels_center.z;
                this->_pointsmap_voxels[z][y][x].max.z = static_cast<float_t>(voxels_minus.z + static_cast<long>(z) + 1l) * this->_voxel_size + this->_voxels_center.z;
            }
        }

        #ifdef _OPENMP
            #pragma omp for
        #endif
        for (size_t i = 0ul; i < this->_points.points.size(); i++) {
            point_xyz<size_t> v_idx = {
                static_cast<size_t>(std::floor((this->_points.points[i].x - this->_voxels_center.x + static_cast<float_t>(this->_voxels_origin.x) * this->_voxel_size) / this->_voxel_size)),
                static_cast<size_t>(std::floor((this->_points.points[i].y - this->_voxels_center.y + static_cast<float_t>(this->_voxels_origin.y) * this->_voxel_size) / this->_voxel_size)),
                static_cast<size_t>(std::floor((this->_points.points[i].z - this->_voxels_center.z + static_cast<float_t>(this->_voxels_origin.z) * this->_voxel_size) / this->_voxel_size))
            };

            #ifdef _OPENMP
                #pragma omp critical
            #endif
            {
                this->_pointsmap_voxels[v_idx.z][v_idx.y][v_idx.x].points.points.push_back(this->_points.points[i]);
            }
        }
    }

    this->_voxels_min = this->_pointsmap_voxels[0][0][0].min;
    this->_voxels_max = this->_pointsmap_voxels[this->_voxels_len.z - 1ul][this->_voxels_len.y - 1ul][this->_voxels_len.x - 1ul].max;

    if (this->_quiet == false) {
        std::cout << "     : " << this->_voxels_len.z * this->_voxels_len.y * this->_voxels_len.x << " voxels" << std::endl;
    }

    pcl::PointCloud<pcl::PointXYZL>().swap(this->_points);
}

//  視錐台(Frustum)を生成するための点群を生成する
void VoxelGridMap::create_frustum_points(pcl::PointCloud<pcl::PointXYZL> &dst)
{
    dst.points.resize(9);

    //  原点
    pcl::PointXYZL origin;
    origin.x = 0.0f; origin.y = 0.0f; origin.z = 0.0f;
    dst.points[frustumPointIndex::Origin] = origin;

    float_t depth_front, depth_back;

    if (std::isinf(this->_depth_max) == true) depth_back = this->_depth_min + 1.0f;
    else depth_back = this->_depth_max;

    if (this->_depth_min > 0.0f) depth_front = this->_depth_min;
    else depth_front = depth_back * 0.5f;

    //  前面左下
    pcl::PointXYZL front_bottom_left;
    front_bottom_left.x = this->_tanL * depth_front;
    front_bottom_left.y = this->_tanB * depth_front;
    front_bottom_left.z = depth_front;
    dst.points[frustumPointIndex::FrontBottomLeft] = front_bottom_left;

    //  前面左上
    pcl::PointXYZL front_top_left;
    front_top_left.x = this->_tanL * depth_front;
    front_top_left.y = this->_tanT * depth_front;
    front_top_left.z = depth_front;
    dst.points[frustumPointIndex::FrontTopLeft] = front_top_left;

    //  前面右上
    pcl::PointXYZL front_top_right;
    front_top_right.x = this->_tanR * depth_front;
    front_top_right.y = this->_tanT * depth_front;
    front_top_right.z = depth_front;
    dst.points[frustumPointIndex::FrontTopRight] = front_top_right;

    //  前面右下
    pcl::PointXYZL front_bottom_right;
    front_bottom_right.x = this->_tanR * depth_front;
    front_bottom_right.y = this->_tanB * depth_front;
    front_bottom_right.z = depth_front;
    dst.points[frustumPointIndex::FrontBottomRight] = front_bottom_right;

    //  後面左下
    pcl::PointXYZL back_bottom_left;
    back_bottom_left.x = this->_tanL * depth_back;
    back_bottom_left.y = this->_tanB * depth_back;
    back_bottom_left.z = depth_back;
    dst.points[frustumPointIndex::BackBottomLeft] = back_bottom_left;

    //  後面左上
    pcl::PointXYZL back_top_left;
    back_top_left.x = this->_tanL * depth_back;
    back_top_left.y = this->_tanT * depth_back;
    back_top_left.z = depth_back;
    dst.points[frustumPointIndex::BackTopLeft] = back_top_left;

    //  後面右上
    pcl::PointXYZL back_top_right;
    back_top_right.x = this->_tanR * depth_back;
    back_top_right.y = this->_tanT * depth_back;
    back_top_right.z = depth_back;
    dst.points[frustumPointIndex::BackTopRight] = back_top_right;

    //  後面右下
    pcl::PointXYZL back_bottom_right;
    back_bottom_right.x = this->_tanR * depth_back;
    back_bottom_right.y = this->_tanB * depth_back;
    back_bottom_right.z = depth_back;
    dst.points[frustumPointIndex::BackBottomRight] = back_bottom_right;
}

//  ボクセルが平面の法線方向(右手系)に存在するか判定する
bool VoxelGridMap::voxel_frontside(const points_voxel &voxel, const Eigen::Vector3f &normal, const Eigen::Vector3f &point_on_plane)
{
    //  平面から最も遠い点を導出
    Eigen::Vector3f voxel_apex_max(
        (normal[axisXYZ::X] > 0.0f)? voxel.max.x : voxel.min.x,
        (normal[axisXYZ::Y] > 0.0f)? voxel.max.y : voxel.min.y,
        (normal[axisXYZ::Z] > 0.0f)? voxel.max.z : voxel.min.z
    );
    Eigen::Vector3f voxel_apex_min(
        (normal[axisXYZ::X] < 0.0f)? voxel.max.x : voxel.min.x,
        (normal[axisXYZ::Y] < 0.0f)? voxel.max.y : voxel.min.y,
        (normal[axisXYZ::Z] < 0.0f)? voxel.max.z : voxel.min.z
    );

    //  0以下の場合，視錐台に含まれる
    return (point_on_plane - voxel_apex_max).dot(normal) <= 0.0f || (point_on_plane - voxel_apex_min).dot(normal) <= 0.0f;
}

//  視錐台に含まれるボクセルを判定し，そのボクセルのインデックスを取得する
void VoxelGridMap::voxels_include_frustum(const Eigen::Vector3f &translation, const Eigen::Quaternionf &quaternion, std::vector<point_xyz<size_t> > &dst_voxel_indexs)
{
    pcl::PointCloud<pcl::PointXYZL> frustum_points;
    this->create_frustum_points(frustum_points);

    Eigen::Vector3f i_translation;
    Eigen::Quaternionf i_quaternion;
    invert_transform(translation, quaternion, i_translation, i_quaternion);
    transform_pointcloud(frustum_points, frustum_points, i_translation, i_quaternion);

    this->voxels_include_frustum(frustum_points, dst_voxel_indexs);
}

//  視錐台に含まれるボクセルを判定し，そのボクセルのインデックスを取得する
void VoxelGridMap::voxels_include_frustum(const Eigen::Matrix4f &matrix, std::vector<point_xyz<size_t> > &dst_voxel_indexs)
{
    pcl::PointCloud<pcl::PointXYZL> frustum_points;
    this->create_frustum_points(frustum_points);

    Eigen::Matrix4f i_matrix;
    invert_transform(matrix, i_matrix);
    transform_pointcloud(frustum_points, frustum_points, i_matrix);

    this->voxels_include_frustum(frustum_points, dst_voxel_indexs);
}

//  変換行列から視錐台に含まれるボクセルを判定し，そのボクセルのインデックスを取得する
py::tuple VoxelGridMap::voxels_include_frustum_from_matrix(const np::ndarray &matrix_4x4)
{
    Eigen::Matrix4f matrix_eigen;
    ndarray2matrix(matrix_4x4, matrix_eigen);
    invert_transform(matrix_eigen, matrix_eigen);

    std::vector<point_xyz<size_t> > voxel_indexs;
    this->voxels_include_frustum(matrix_eigen, voxel_indexs);
    return voxelindexs2tuple(voxel_indexs);
}

//  並進ベクトルとクォータニオンから視錐台に含まれるボクセルを判定し，そのボクセルのインデックスを取得する
py::tuple VoxelGridMap::voxels_include_frustum_from_quaternion(const np::ndarray &translation, const np::ndarray &quaternion)
{
    Eigen::Vector3f translation_eigen;
    Eigen::Quaternionf quaternion_eigen;
    ndarray2translation(translation, translation_eigen);
    ndarray2quaternion(quaternion, quaternion_eigen);
    invert_transform(translation_eigen, quaternion_eigen, translation_eigen, quaternion_eigen);

    std::vector<point_xyz<size_t> > voxel_indexs;
    this->voxels_include_frustum(translation_eigen, quaternion_eigen, voxel_indexs);
    return voxelindexs2tuple(voxel_indexs);
}

//  視錐台に含まれるボクセルを判定し，そのボクセルのインデックスを取得する
void VoxelGridMap::voxels_include_frustum(const pcl::PointCloud<pcl::PointXYZL> &frustum_points, std::vector<point_xyz<size_t> > &dst_voxel_indexs)
{
    /*  0: FrontBottomLeft
        1: FrontTopLeft
        2: FrontTopRight
        3: FrontBottomRight
        4: BackBottomLeft
        5: BackTopLeft
        6: BackTopRight
        7: BackBottomRight
        8: Origin           */
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > points(9);
    for (size_t i = 0ul; i < 9ul; i++) {
        points[i][axisXYZ::X] = frustum_points.points[i].x;
        points[i][axisXYZ::Y] = frustum_points.points[i].y;
        points[i][axisXYZ::Z] = frustum_points.points[i].z;
    }

    /*  0: Origin -> BackBottomLeft
        1: Origin -> BackTopLeft
        2: Origin -> BackTopRight
        3: Origin -> BackBottomRight
        4: FrontBottomRight -> FrontBottomLeft
        5: FrontBottomRight -> FrontTopRight
        6: BackTopLeft -> BackBottomLeft
        7: BackTopLeft -> BackTopRight          */
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > vectors(8);
    for (size_t i = 0ul; i < 4ul; i++)
        vectors[i] = points[i + 4ul] - points[frustumPointIndex::Origin];
    for (size_t i = 0ul; i < 2ul; i++) {
        vectors[i + 4ul] = points[i * 2ul] - points[frustumPointIndex::FrontBottomRight];
        vectors[i + 6ul] = points[i * 2ul + 4ul] - points[frustumPointIndex::BackTopLeft];
    }

    /*  0: Left
        1: Top
        2: Right
        3: Bottom
        4: Front
        5: Back     */
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > normals(6);
    for (size_t i = 0ul; i < 4ul; i++) {
        size_t i_next = (i >= 3ul)? 0ul : i + 1ul;
        normals[i] = vectors[i].cross(vectors[i_next]);
    }
    for (size_t i = 0ul; i < 2ul; i++)
        normals[i + 4ul] = vectors[i * 2ul + 4ul].cross(vectors[i * 2ul + 5ul]);

    /*  0: FrontBottomLeft
        1: FrontTopLeft
        2: FrontTopRight
        3: FrontBottomRight
        4: BackBottomLeft
        5: BackTopLeft
        6: BackTopRight
        7: BackBottomRight  */
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > vertexes(8);

    if (this->_depth_min > 0.0f) for (size_t i = 0ul; i < 4ul; i++) vertexes[i] = points[i];
    else for (size_t i = 0ul; i < 4ul; i++) vertexes[i] = points[frustumPointIndex::Origin];

    if (std::isinf(this->_depth_max) == true)
        for (size_t i = 0ul; i < 4ul; i++) {
            Eigen::Vector3f voxels_lim(
                (vectors[i][axisXYZ::X] < 0.0f)? this->_voxels_min.x : this->_voxels_max.x,
                (vectors[i][axisXYZ::Y] < 0.0f)? this->_voxels_min.y : this->_voxels_max.y,
                (vectors[i][axisXYZ::Z] < 0.0f)? this->_voxels_min.z : this->_voxels_max.z
            );
            float_t t = std::min({
                (voxels_lim[axisXYZ::X] - points[frustumPointIndex::Origin][axisXYZ::X]) / vectors[i][axisXYZ::X],
                (voxels_lim[axisXYZ::Y] - points[frustumPointIndex::Origin][axisXYZ::Y]) / vectors[i][axisXYZ::Y],
                (voxels_lim[axisXYZ::Z] - points[frustumPointIndex::Origin][axisXYZ::Z]) / vectors[i][axisXYZ::Z]
            });
            vertexes[i + 4ul] = points[frustumPointIndex::Origin] + vectors[i] * t;
        }
    else
        for (size_t i = 4ul; i < 8ul; i++)
            vertexes[i] = points[i];

    Eigen::Vector3f point_min = vertexes[0];
    Eigen::Vector3f point_max = vertexes[0];
    for (size_t i = 1ul; i < 8ul; i++) {
        if (vertexes[i][axisXYZ::X] < point_min[axisXYZ::X]) point_min[axisXYZ::X] = vertexes[i][axisXYZ::X];
        else if (point_max[axisXYZ::X] < vertexes[i][axisXYZ::X]) point_max[axisXYZ::X] = vertexes[i][axisXYZ::X];

        if (vertexes[i][axisXYZ::Y] < point_min[axisXYZ::Y]) point_min[axisXYZ::Y] = vertexes[i][axisXYZ::Y];
        else if (point_max[axisXYZ::Y] < vertexes[i][axisXYZ::Y]) point_max[axisXYZ::Y] = vertexes[i][axisXYZ::Y];

        if (vertexes[i][axisXYZ::Z] < point_min[axisXYZ::Z]) point_min[axisXYZ::Z] = vertexes[i][axisXYZ::Z];
        else if (point_max[axisXYZ::Z] < vertexes[i][axisXYZ::Z]) point_max[axisXYZ::Z] = vertexes[i][axisXYZ::Z];
    }

    point_xyz<float_t> v_idx_min_f = {
        std::floor((point_min[axisXYZ::X] - this->_voxels_center.x + static_cast<float_t>(this->_voxels_origin.x) * this->_voxel_size) / this->_voxel_size),
        std::floor((point_min[axisXYZ::Y] - this->_voxels_center.y + static_cast<float_t>(this->_voxels_origin.y) * this->_voxel_size) / this->_voxel_size),
        std::floor((point_min[axisXYZ::Z] - this->_voxels_center.z + static_cast<float_t>(this->_voxels_origin.z) * this->_voxel_size) / this->_voxel_size)
    };
    point_xyz<float_t> v_idx_max_f = {
        std::ceil((point_max[axisXYZ::X] - this->_voxels_center.x + static_cast<float_t>(this->_voxels_origin.x) * this->_voxel_size) / this->_voxel_size),
        std::ceil((point_max[axisXYZ::Y] - this->_voxels_center.y + static_cast<float_t>(this->_voxels_origin.y) * this->_voxel_size) / this->_voxel_size),
        std::ceil((point_max[axisXYZ::Z] - this->_voxels_center.z + static_cast<float_t>(this->_voxels_origin.z) * this->_voxel_size) / this->_voxel_size)
    };

    point_xyz<size_t> v_idx_min = {
        (v_idx_min_f.x < 0.0f)? 0ul : static_cast<size_t>(v_idx_min_f.x),
        (v_idx_min_f.y < 0.0f)? 0ul : static_cast<size_t>(v_idx_min_f.y),
        (v_idx_min_f.z < 0.0f)? 0ul : static_cast<size_t>(v_idx_min_f.z)
    };
    point_xyz<size_t> v_idx_max = {
        (v_idx_max_f.x > static_cast<float_t>(this->_voxels_len.x))? this->_voxels_len.x : (v_idx_max_f.x < 0.0f)? 0ul : static_cast<size_t>(v_idx_max_f.x),
        (v_idx_max_f.y > static_cast<float_t>(this->_voxels_len.y))? this->_voxels_len.y : (v_idx_max_f.y < 0.0f)? 0ul : static_cast<size_t>(v_idx_max_f.y),
        (v_idx_max_f.z > static_cast<float_t>(this->_voxels_len.z))? this->_voxels_len.z : (v_idx_max_f.z < 0.0f)? 0ul : static_cast<size_t>(v_idx_max_f.z)
    };

    dst_voxel_indexs.clear();

    for (size_t z = v_idx_min.z; z < v_idx_max.z; z++) for (size_t y = v_idx_min.y; y < v_idx_max.y; y++) for (size_t x = v_idx_min.x; x < v_idx_max.x; x++) {
        bool in_frustum = true;

        for (size_t i = 0ul; i < 4ul; i++)
            in_frustum &= this->voxel_frontside(this->_pointsmap_voxels[z][y][x], normals[i], points[frustumPointIndex::Origin]);

        if (this->_depth_min > 0.0f)
            in_frustum &= this->voxel_frontside(this->_pointsmap_voxels[z][y][x], normals[frustumSurfaceIndex::Front], points[frustumPointIndex::FrontBottomRight]);

        if (std::isinf(this->_depth_max) == false)
            in_frustum &= this->voxel_frontside(this->_pointsmap_voxels[z][y][x], normals[frustumSurfaceIndex::Back], points[frustumPointIndex::BackTopLeft]);

        if (in_frustum == false) continue;

        point_xyz<size_t> voxel_index = {x, y, z};
        dst_voxel_indexs.push_back(voxel_index);
    }
}

//  自己位置を用いて視錐台に含まれるボクセルを判定し，そのボクセルから点群を取得する
void VoxelGridMap::points_include_frustum(const Eigen::Vector3f &translation, const Eigen::Quaternionf &quaternion, pcl::PointCloud<pcl::PointXYZL> &dst)
{
    std::vector<point_xyz<size_t> > voxel_indexs;
    this->voxels_include_frustum(translation, quaternion, voxel_indexs);

    size_t len = voxel_indexs.size();
    std::vector<size_t> voxels_size = {0ul};
    for (size_t i = 0ul; i < len; i++) {
        voxels_size.push_back(this->_pointsmap_voxels[voxel_indexs[i].z][voxel_indexs[i].y][voxel_indexs[i].x].points.points.size());
    }
    dst.resize(std::accumulate(voxels_size.begin(), voxels_size.end(), 0));
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    pcl::ExtractIndices<pcl::PointXYZL> extract;

    #ifdef _OPENMP
        #pragma omp parallel for
    #endif
    for (size_t i = 0ul; i < len; i++) {
        pcl::PointCloud<pcl::PointXYZL> points;
        this->transformPointCloud(this->_pointsmap_voxels[voxel_indexs[i].z][voxel_indexs[i].y][voxel_indexs[i].x].points, points, translation, quaternion, false);

        size_t *p_len = &voxels_size[i + 1];

        for (size_t p = 0; p < *p_len; p++) {
            pcl::PointXYZL *src_point = &(points.points[p]);

            float_t th_R = src_point->z * this->_tanR;
            float_t th_L = src_point->z * this->_tanL;
            float_t th_T = src_point->z * this->_tanT;
            float_t th_B = src_point->z * this->_tanB;

            if (th_L <= src_point->x && src_point->x <= th_R && th_T <= src_point->y && src_point->y <= th_B) {
                dst.points[voxels_size[i] + p] = *src_point;
            }
            else {
                #ifdef _OPENMP
                    #pragma omp critical
                #endif
                inliers->indices.push_back(voxels_size[i] + p);
            }
        }
    }

    extract.setInputCloud(dst.makeShared());
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(dst);
}

//  自己位置を用いて視錐台に含まれるボクセルを判定し，そのボクセルから点群を取得する
void VoxelGridMap::points_include_frustum(const Eigen::Matrix4f &matrix, pcl::PointCloud<pcl::PointXYZL> &dst)
{
    std::vector<point_xyz<size_t> > voxel_indexs;
    this->voxels_include_frustum(matrix, voxel_indexs);

    size_t len = voxel_indexs.size();
    std::vector<size_t> voxels_size = {0ul};
    for (size_t i = 0ul; i < len; i++) {
        voxels_size.push_back(this->_pointsmap_voxels[voxel_indexs[i].z][voxel_indexs[i].y][voxel_indexs[i].x].points.points.size());
    }
    dst.resize(std::accumulate(voxels_size.begin(), voxels_size.end(), 0));
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    pcl::ExtractIndices<pcl::PointXYZL> extract;

    #ifdef _OPENMP
        #pragma omp parallel for
    #endif
    for (size_t i = 0ul; i < len; i++) {
        pcl::PointCloud<pcl::PointXYZL> points;
        this->transformPointCloud(this->_pointsmap_voxels[voxel_indexs[i].z][voxel_indexs[i].y][voxel_indexs[i].x].points, points, matrix, false);

        size_t *p_len = &voxels_size[i + 1];

        for (size_t p = 0; p < *p_len; p++) {
            pcl::PointXYZL *src_point = &(points.points[p]);

            float_t th_R = src_point->z * this->_tanR;
            float_t th_L = src_point->z * this->_tanL;
            float_t th_T = src_point->z * this->_tanT;
            float_t th_B = src_point->z * this->_tanB;

            if (th_L <= src_point->x && src_point->x <= th_R && th_T <= src_point->y && src_point->y <= th_B) {
                dst.points[voxels_size[i] + p] = *src_point;
            }
            else {
                #ifdef _OPENMP
                    #pragma omp critical
                #endif
                inliers->indices.push_back(voxels_size[i] + p);
            }
        }
    }

    extract.setInputCloud(dst.makeShared());
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(dst);
}

//  点群から深度マップを生成する
void VoxelGridMap::create_depthmap(const std::vector<point_xyz<size_t> > &voxel_indexs, const Eigen::Vector3f &translation, const Eigen::Quaternionf &quaternion, cv::Mat &dst)
{
    if (this->_fx == NAN) throw std::runtime_error("Fx must not be nan.");
    if (this->_fy == NAN) throw std::runtime_error("Fy must not be nan.");
    if (this->_cx == NAN) throw std::runtime_error("Cx must not be nan.");
    if (this->_cy == NAN) throw std::runtime_error("Cy must not be nan.");

    dst = cv::Mat(this->_height, this->_width, CV_32FC1, cv::Scalar_<float_t>(INFINITY));

    size_t len = voxel_indexs.size();

    std::vector<cv::Mat> processing_dst;
    #ifdef _OPENMP
        int threads = omp_get_max_threads();
        processing_dst.assign(threads, dst.clone());
        #pragma omp parallel
    #else
        int threads = 1;
        processing_dst.assign(threads, dst.clone());
    #endif
    {
        #ifdef _OPENMP
            #pragma omp for
        #endif
        for (size_t i = 0ul; i < len; i++) {
            pcl::PointCloud<pcl::PointXYZL> points;
            this->transformPointCloud(this->_pointsmap_voxels[voxel_indexs[i].z][voxel_indexs[i].y][voxel_indexs[i].x].points, points, translation, quaternion, true);

            size_t p_len = points.points.size();

            for (size_t p = 0; p < p_len; p++) {
                pcl::PointXYZL *src_point = &(points.points[p]);

                float_t th_R = src_point->z * this->_tanR;
                float_t th_L = src_point->z * this->_tanL;
                float_t th_T = src_point->z * this->_tanT;
                float_t th_B = src_point->z * this->_tanB;

                if (th_L <= src_point->x && src_point->x <= th_R && th_T <= src_point->y && src_point->y <= th_B) {
                    int x, y;

                    if (src_point->x < 0.0f) x = static_cast<int>(this->_cx - roundf(this->_cx * src_point->x / (src_point->z * this->_tanL)));
                    else x = static_cast<int>(this->_cx + roundf((this->_width_f - this->_cx) * src_point->x / (src_point->z * this->_tanR)));

                    if (src_point->y < 0.0f) y = static_cast<int>(this->_cy - roundf(this->_cy * src_point->y / (src_point->z * this->_tanT)));
                    else y = static_cast<int>(this->_cy + roundf((this->_height_f - this->_cy) * src_point->y / (src_point->z * this->_tanB)));

                    if (0 <= x && x < this->_width && 0 <= y && y < this->_height) {
                        #ifdef _OPENMP
                            float_t *cv_dst_ptr = processing_dst[omp_get_thread_num()].ptr<float_t>(y);
                        #else
                            float_t *cv_dst_ptr = processing_dst[0].ptr<float_t>(y);
                        #endif
                        if (cv_dst_ptr[x] > src_point->z) {
                            cv_dst_ptr[x] = src_point->z;
                        }
                    }
                }
            }
        }

        #ifdef _OPENMP
            #pragma omp for
        #endif
        for (int y = 0; y < dst.rows; y++) {
            std::vector<float_t*> dst_ptrs;
            for (int i = 0; i < threads; i++) {
                dst_ptrs.push_back(processing_dst[i].ptr<float_t>(y));
            }
            float_t* cv_dst_ptr = dst.ptr<float_t>(y);
            for (int x = 0; x < dst.cols; x++) {
                for (int i = 0; i < threads; i++) {
                    cv_dst_ptr[x] = std::min(cv_dst_ptr[x], dst_ptrs[i][x]);
                }
            }
        }
    }
}

//  点群から深度マップを生成する
void VoxelGridMap::create_depthmap(const std::vector<point_xyz<size_t> > &voxel_indexs, const Eigen::Matrix4f &matrix, cv::Mat &dst)
{
    if (this->_fx == NAN) throw std::runtime_error("Fx must not be nan.");
    if (this->_fy == NAN) throw std::runtime_error("Fy must not be nan.");
    if (this->_cx == NAN) throw std::runtime_error("Cx must not be nan.");
    if (this->_cy == NAN) throw std::runtime_error("Cy must not be nan.");

    dst = cv::Mat(this->_height, this->_width, CV_32FC1, cv::Scalar_<float_t>(INFINITY));

    size_t len = voxel_indexs.size();

    std::vector<cv::Mat> processing_dst;
    #ifdef _OPENMP
        int threads = omp_get_max_threads();
        processing_dst.assign(threads, dst.clone());
        #pragma omp parallel
    #else
        int threads = 1;
        processing_dst.assign(threads, dst.clone());
    #endif
    {
        #ifdef _OPENMP
            #pragma omp for
        #endif
        for (size_t i = 0ul; i < len; i++) {
            pcl::PointCloud<pcl::PointXYZL> points;
            this->transformPointCloud(this->_pointsmap_voxels[voxel_indexs[i].z][voxel_indexs[i].y][voxel_indexs[i].x].points, points, matrix, true);

            size_t p_len = points.points.size();

            for (size_t p = 0; p < p_len; p++) {
                pcl::PointXYZL *src_point = &(points.points[p]);

                float_t th_R = src_point->z * this->_tanR;
                float_t th_L = src_point->z * this->_tanL;
                float_t th_T = src_point->z * this->_tanT;
                float_t th_B = src_point->z * this->_tanB;

                if (th_L <= src_point->x && src_point->x <= th_R && th_T <= src_point->y && src_point->y <= th_B) {
                    int x, y;

                    if (src_point->x < 0.0f) x = static_cast<int>(this->_cx - roundf(this->_cx * src_point->x / (src_point->z * this->_tanL)));
                    else x = static_cast<int>(this->_cx + roundf((this->_width_f - this->_cx) * src_point->x / (src_point->z * this->_tanR)));

                    if (src_point->y < 0.0f) y = static_cast<int>(this->_cy - roundf(this->_cy * src_point->y / (src_point->z * this->_tanT)));
                    else y = static_cast<int>(this->_cy + roundf((this->_height_f - this->_cy) * src_point->y / (src_point->z * this->_tanB)));

                    if (0 <= x && x < this->_width && 0 <= y && y < this->_height) {
                        #ifdef _OPENMP
                            float_t *cv_dst_ptr = processing_dst[omp_get_thread_num()].ptr<float_t>(y);
                        #else
                            float_t *cv_dst_ptr = processing_dst[0].ptr<float_t>(y);
                        #endif
                        if (cv_dst_ptr[x] > src_point->z) {
                            cv_dst_ptr[x] = src_point->z;
                        }
                    }
                }
            }
        }

        #ifdef _OPENMP
            #pragma omp for
        #endif
        for (int y = 0; y < dst.rows; y++) {
            std::vector<float_t*> dst_ptrs;
            for (int i = 0; i < threads; i++) {
                dst_ptrs.push_back(processing_dst[i].ptr<float_t>(y));
            }
            float_t* cv_dst_ptr = dst.ptr<float_t>(y);
            for (int x = 0; x < dst.cols; x++) {
                for (int i = 0; i < threads; i++) {
                    cv_dst_ptr[x] = std::min(cv_dst_ptr[x], dst_ptrs[i][x]);
                }
            }
        }
    }
}

//  点群からSemantic2dを生成する
void VoxelGridMap::create_semantic2d(const std::vector<point_xyz<size_t> > &voxel_indexs, const Eigen::Vector3f &translation, const Eigen::Quaternionf &quaternion, cv::Mat &dst, cv::Mat &depth)
{
    if (this->_fx == NAN) throw std::runtime_error("Fx must not be nan.");
    if (this->_fy == NAN) throw std::runtime_error("Fy must not be nan.");
    if (this->_cx == NAN) throw std::runtime_error("Cx must not be nan.");
    if (this->_cy == NAN) throw std::runtime_error("Cy must not be nan.");

    dst = cv::Mat::zeros(this->_height, this->_width, CV_8UC1);
    depth = cv::Mat(this->_height, this->_width, CV_32FC1, cv::Scalar_<float_t>(INFINITY));

    size_t len = voxel_indexs.size();

    std::vector<cv::Mat> processing_dst;
    std::vector<cv::Mat> processing_depth;
    #ifdef _OPENMP
        int threads = omp_get_max_threads();
        processing_dst.assign(threads, dst.clone());
        processing_depth.assign(threads, depth.clone());
        #pragma omp parallel
    #else
        int threads = 1;
        processing_dst.assign(threads, dst.clone());
        processing_depth.assign(threads, depth.clone());
    #endif
    {
        #ifdef _OPENMP
            #pragma omp for
        #endif
        for (size_t i = 0ul; i < len; i++) {
            pcl::PointCloud<pcl::PointXYZL> points;
            this->transformPointCloud(this->_pointsmap_voxels[voxel_indexs[i].z][voxel_indexs[i].y][voxel_indexs[i].x].points, points, translation, quaternion, true);

            size_t p_len = points.points.size();

            for (size_t p = 0; p < p_len; p++) {
                pcl::PointXYZL *src_point = &(points.points[p]);

                float_t th_R = src_point->z * this->_tanR;
                float_t th_L = src_point->z * this->_tanL;
                float_t th_T = src_point->z * this->_tanT;
                float_t th_B = src_point->z * this->_tanB;

                if (th_L <= src_point->x && src_point->x <= th_R && th_T <= src_point->y && src_point->y <= th_B) {
                    int x, y;

                    if (src_point->x < 0.0f) x = static_cast<int>(this->_cx - roundf(this->_cx * src_point->x / (src_point->z * this->_tanL)));
                    else x = static_cast<int>(this->_cx + roundf((this->_width_f - this->_cx) * src_point->x / (src_point->z * this->_tanR)));

                    if (src_point->y < 0.0f) y = static_cast<int>(this->_cy - roundf(this->_cy * src_point->y / (src_point->z * this->_tanT)));
                    else y = static_cast<int>(this->_cy + roundf((this->_height_f - this->_cy) * src_point->y / (src_point->z * this->_tanB)));

                    if (0 <= x && x < this->_width && 0 <= y && y < this->_height) {
                        #ifdef _OPENMP
                            float_t *cv_depth_ptr = processing_depth[omp_get_thread_num()].ptr<float_t>(y);
                            u_int8_t *cv_dst_ptr = processing_dst[omp_get_thread_num()].ptr<u_int8_t>(y);
                        #else
                            float_t *cv_depth_ptr = processing_depth[0].ptr<float_t>(y);
                            u_int8_t *cv_dst_ptr = processing_dst[0].ptr<u_int8_t>(y);
                        #endif
                        if (cv_depth_ptr[x] > src_point->z) {
                            cv_depth_ptr[x] = src_point->z;
                            cv_dst_ptr[x] = static_cast<u_int8_t>(src_point->label);
                        }
                    }
                }
            }
        }

        #ifdef _OPENMP
            #pragma omp for
        #endif
        for (int y = 0; y < dst.rows; y++) {
            std::vector<float_t*> depth_ptrs;
            std::vector<u_int8_t*> dst_ptrs;
            for (int i = 0; i < threads; i++) {
                depth_ptrs.push_back(processing_depth[i].ptr<float_t>(y));
                dst_ptrs.push_back(processing_dst[i].ptr<u_int8_t>(y));
            }
            float_t* cv_depth_ptr = depth.ptr<float_t>(y);
            u_int8_t* cv_dst_ptr = dst.ptr<u_int8_t>(y);
            for (int x = 0; x < dst.cols; x++) {
                for (int i = 0; i < threads; i++) {
                    if (depth_ptrs[i][x] < cv_depth_ptr[x]) {
                        cv_depth_ptr[x] = depth_ptrs[i][x];
                        cv_dst_ptr[x] = dst_ptrs[i][x];
                    }
                }
            }
        }
    }
}

//  点群からSemantic2dを生成する
void VoxelGridMap::create_semantic2d(const std::vector<point_xyz<size_t> > &voxel_indexs, const Eigen::Matrix4f &matrix, cv::Mat &dst, cv::Mat &depth)
{
    if (this->_fx == NAN) throw std::runtime_error("Fx must not be nan.");
    if (this->_fy == NAN) throw std::runtime_error("Fy must not be nan.");
    if (this->_cx == NAN) throw std::runtime_error("Cx must not be nan.");
    if (this->_cy == NAN) throw std::runtime_error("Cy must not be nan.");

    dst = cv::Mat::zeros(this->_height, this->_width, CV_8UC1);
    depth = cv::Mat(this->_height, this->_width, CV_32FC1, cv::Scalar_<float_t>(INFINITY));

    size_t len = voxel_indexs.size();

    std::vector<cv::Mat> processing_dst;
    std::vector<cv::Mat> processing_depth;
    #ifdef _OPENMP
        int threads = omp_get_max_threads();
        processing_dst.assign(threads, dst.clone());
        processing_depth.assign(threads, depth.clone());
        #pragma omp parallel
    #else
        int threads = 1;
        processing_dst.assign(threads, dst.clone());
        processing_depth.assign(threads, depth.clone());
    #endif
    {
        #ifdef _OPENMP
            #pragma omp for
        #endif
        for (size_t i = 0ul; i < len; i++) {
            pcl::PointCloud<pcl::PointXYZL> points;
            this->transformPointCloud(this->_pointsmap_voxels[voxel_indexs[i].z][voxel_indexs[i].y][voxel_indexs[i].x].points, points, matrix, true);

            size_t p_len = points.points.size();

            for (size_t p = 0; p < p_len; p++) {
                pcl::PointXYZL *src_point = &(points.points[p]);

                float_t th_R = src_point->z * this->_tanR;
                float_t th_L = src_point->z * this->_tanL;
                float_t th_T = src_point->z * this->_tanT;
                float_t th_B = src_point->z * this->_tanB;

                if (th_L <= src_point->x && src_point->x <= th_R && th_T <= src_point->y && src_point->y <= th_B) {
                    int x, y;

                    if (src_point->x < 0.0f) x = static_cast<int>(this->_cx - roundf(this->_cx * src_point->x / (src_point->z * this->_tanL)));
                    else x = static_cast<int>(this->_cx + roundf((this->_width_f - this->_cx) * src_point->x / (src_point->z * this->_tanR)));

                    if (src_point->y < 0.0f) y = static_cast<int>(this->_cy - roundf(this->_cy * src_point->y / (src_point->z * this->_tanT)));
                    else y = static_cast<int>(this->_cy + roundf((this->_height_f - this->_cy) * src_point->y / (src_point->z * this->_tanB)));

                    if (0 <= x && x < this->_width && 0 <= y && y < this->_height) {
                        #ifdef _OPENMP
                            float_t *cv_depth_ptr = processing_depth[omp_get_thread_num()].ptr<float_t>(y);
                            u_int8_t *cv_dst_ptr = processing_dst[omp_get_thread_num()].ptr<u_int8_t>(y);
                        #else
                            float_t *cv_depth_ptr = processing_depth[0].ptr<float_t>(y);
                            u_int8_t *cv_dst_ptr = processing_dst[0].ptr<u_int8_t>(y);
                        #endif
                        if (cv_depth_ptr[x] > src_point->z) {
                            cv_depth_ptr[x] = src_point->z;
                            cv_dst_ptr[x] = static_cast<u_int8_t>(src_point->label);
                        }
                    }
                }
            }
        }

        #ifdef _OPENMP
            #pragma omp for
        #endif
        for (int y = 0; y < dst.rows; y++) {
            std::vector<float_t*> depth_ptrs;
            std::vector<u_int8_t*> dst_ptrs;
            for (int i = 0; i < threads; i++) {
                depth_ptrs.push_back(processing_depth[i].ptr<float_t>(y));
                dst_ptrs.push_back(processing_dst[i].ptr<u_int8_t>(y));
            }
            float_t* cv_depth_ptr = depth.ptr<float_t>(y);
            u_int8_t* cv_dst_ptr = dst.ptr<u_int8_t>(y);
            for (int x = 0; x < dst.cols; x++) {
                for (int i = 0; i < threads; i++) {
                    if (depth_ptrs[i][x] < cv_depth_ptr[x]) {
                        cv_depth_ptr[x] = depth_ptrs[i][x];
                        cv_dst_ptr[x] = dst_ptrs[i][x];
                    }
                }
            }
        }
    }
}

//  ダウンサンプリングする
void VoxelGridMap::downsampling(float_t leaf_size)
{
    for (size_t x = 0ul; x < this->_voxels_len.x; x++) for (size_t y = 0ul; y < this->_voxels_len.y; y++) for (size_t z = 0ul; z < this->_voxels_len.z; z++) {
        pcl::PointCloud<pcl::PointXYZL> tmp;
        this->voxelGridFilter(this->_pointsmap_voxels[z][y][x].points, tmp, leaf_size);
        this->_pointsmap_voxels[z][y][x].points.swap(tmp);
    }
}

boost::shared_ptr<VoxelGridMap> VoxelGridMap_init(bool quite)
{
    boost::shared_ptr<VoxelGridMap> vgm(new VoxelGridMap(quite));
    return vgm;
}

}   //  namespace pointsmap
