#include "common.hpp"

namespace pointsmap {

//  変換行列の逆変換
void invert_transform(const Eigen::Matrix4f &in_matrix, Eigen::Matrix4f &out_matrix)
{
    Eigen::Matrix4f matrix;

    matrix.topLeftCorner(3, 3) = in_matrix.topLeftCorner(3, 3).transpose();
    matrix.topRightCorner(3, 1) = -(matrix.topLeftCorner(3, 3) * in_matrix.topRightCorner(3, 1));
    matrix.bottomRows(1) = in_matrix.bottomRows(1);

    out_matrix = matrix;
}

//  並進ベクトルとクォータニオンの逆変換
void invert_transform(const Eigen::Vector3f &in_tr, const Eigen::Quaternionf &in_q, Eigen::Vector3f &out_tr, Eigen::Quaternionf &out_q)
{
    Eigen::Vector3f translation;
    Eigen::Quaternionf quaternion;

    quaternion = in_q.conjugate();
    translation = -(quaternion * in_tr);

    out_tr = translation;
    out_q = quaternion;
}

//  変換行列を逆変換する (for Python)
np::ndarray invert_transform_matrix(const np::ndarray &matrix_4x4)
{
    Eigen::Matrix4f transform;
    ndarray2matrix(matrix_4x4, transform);
    invert_transform(transform, transform);
    return matrix2ndarray(transform);
}

//  並進ベクトルとクォータニオンを逆変換する (for Python)
py::tuple invert_transform_quaternion(const np::ndarray &translation, const np::ndarray &quaternion)
{
    Eigen::Vector3f translation_eigen;
    ndarray2translation(translation, translation_eigen);
    Eigen::Quaternionf quaternion_eigen;
    ndarray2quaternion(quaternion, quaternion_eigen);
    invert_transform(translation_eigen, quaternion_eigen, translation_eigen, quaternion_eigen);
    return py::make_tuple(translation2ndarray(translation_eigen), quaternion2ndarray(quaternion_eigen));
}

//  変換行列を並進ベクトルとクォータニオンに変換する
void matrix2quaternion(const Eigen::Matrix4f &in_matrix, Eigen::Vector3f &out_translation, Eigen::Quaternionf &out_quaternion)
{
    out_translation = in_matrix.topRightCorner(3, 1);
    Eigen::Matrix3f tmp_rotation = in_matrix.topLeftCorner(3, 3);
    out_quaternion = Eigen::Quaternionf(tmp_rotation);
}

//  変換行列を並進ベクトルとクォータニオンに変換する (for Python)
py::tuple matrix2quaternion_python(const np::ndarray &matrix_4x4)
{
    Eigen::Matrix4f transform;
    ndarray2matrix(matrix_4x4, transform);
    Eigen::Vector3f translation_eigen;
    Eigen::Quaternionf quaternion_eigen;
    matrix2quaternion(transform, translation_eigen, quaternion_eigen);
    return py::make_tuple(translation2ndarray(translation_eigen), quaternion2ndarray(quaternion_eigen));
}

//  並進ベクトルとクォータニオンを変換行列に変換する
void quaternion2matrix(const Eigen::Vector3f &in_translation, const Eigen::Quaternionf &in_quaternion, Eigen::Matrix4f &out_matrix)
{
    out_matrix = Eigen::Matrix4f::Identity();
    out_matrix.topRightCorner(3, 1) = in_translation;
    out_matrix.topLeftCorner(3, 3) = in_quaternion.toRotationMatrix();
}

//  並進ベクトルとクォータニオンを変換行列に変換する (for Python)
np::ndarray quaternion2matrix_python(const np::ndarray &translation, const np::ndarray &quaternion)
{
    Eigen::Vector3f translation_eigen;
    ndarray2translation(translation, translation_eigen);
    Eigen::Quaternionf quaternion_eigen;
    ndarray2quaternion(quaternion, quaternion_eigen);
    Eigen::Matrix4f transform;
    quaternion2matrix(translation_eigen, quaternion_eigen, transform);
    return matrix2ndarray(transform);
}

//  cv::Matをnp::ndarrayへ変換する
np::ndarray cvmat2ndarray(const cv::Mat &src)
{
    if (src.channels() == 1) {
        Py_intptr_t shape[2] = {src.rows, src.cols};

        if (src.depth() == CV_64F) {
            np::ndarray dst = np::zeros(2, shape, np::dtype::get_builtin<double_t>());
            std::copy(src.begin<double_t>(), src.end<double_t>(), reinterpret_cast<double_t*>(dst.get_data()));
            return dst;
        }
        else if (src.depth() == CV_32F) {
            np::ndarray dst = np::zeros(2, shape, np::dtype::get_builtin<float_t>());
            std::copy(src.begin<float_t>(), src.end<float_t>(), reinterpret_cast<float_t*>(dst.get_data()));
            return dst;
        }
        else if (src.depth() == CV_32S) {
            np::ndarray dst = np::zeros(2, shape, np::dtype::get_builtin<int32_t>());
            std::copy(src.begin<int32_t>(), src.end<int32_t>(), reinterpret_cast<int32_t*>(dst.get_data()));
            return dst;
        }
        else if (src.depth() == CV_16U) {
            np::ndarray dst = np::zeros(2, shape, np::dtype::get_builtin<u_int16_t>());
            std::copy(src.begin<u_int16_t>(), src.end<u_int16_t>(), reinterpret_cast<u_int16_t*>(dst.get_data()));
            return dst;
        }
        else if (src.depth() == CV_16S) {
            np::ndarray dst = np::zeros(2, shape, np::dtype::get_builtin<int16_t>());
            std::copy(src.begin<int16_t>(), src.end<int16_t>(), reinterpret_cast<int16_t*>(dst.get_data()));
            return dst;
        }
        else if (src.depth() == CV_8U) {
            np::ndarray dst = np::zeros(2, shape, np::dtype::get_builtin<u_int8_t>());
            std::copy(src.begin<u_int8_t>(), src.end<u_int8_t>(), reinterpret_cast<u_int8_t*>(dst.get_data()));
            return dst;
        }
        else if (src.depth() == CV_8S) {
            np::ndarray dst = np::zeros(2, shape, np::dtype::get_builtin<int8_t>());
            std::copy(src.begin<int8_t>(), src.end<int8_t>(), reinterpret_cast<int8_t*>(dst.get_data()));
            return dst;
        }
        throw std::invalid_argument("\"src\" has invalid bit depth.");
    }
    else {
        py::tuple shape = py::make_tuple(src.rows, src.cols, src.channels());
        py::tuple strides = py::make_tuple(src.cols * src.elemSize(), src.elemSize(), src.elemSize1());

        np::dtype dtype = np::dtype::get_builtin<double_t>();

        if (src.depth() == CV_64F) dtype = np::dtype::get_builtin<double_t>();
        else if (src.depth() == CV_32F) dtype = np::dtype::get_builtin<float_t>();
        else if (src.depth() == CV_32S) dtype = np::dtype::get_builtin<int32_t>();
        else if (src.depth() == CV_16U) dtype = np::dtype::get_builtin<u_int16_t>();
        else if (src.depth() == CV_16S) dtype = np::dtype::get_builtin<int16_t>();
        else if (src.depth() == CV_8U) dtype = np::dtype::get_builtin<u_int8_t>();
        else if (src.depth() == CV_8S) dtype = np::dtype::get_builtin<int8_t>();
        else throw std::invalid_argument("\"src\" has invalid bit depth.");

        np::ndarray dst = np::from_data(src.data, dtype, shape, strides, py::object());
        return dst;
    }

}

//  np::ndarrayをcv::Matへ変換する
void ndarray2cvmat(const np::ndarray &src, cv::Mat &dst)
{
    const int ndim = src.get_nd();
    const Py_intptr_t* shape = src.get_shape();
    const np::dtype dtype = src.get_dtype();

    void* data = PyArray_DATA((PyArrayObject*)src.ptr());

    int type, rows, cols, channels, depth;

    if (ndim == 3) {
        rows = shape[0];
        cols = shape[1];
        channels = shape[2];
    }
    else if (ndim == 2) {
        rows = shape[0];
        cols = shape[1];
        channels = 1;
    }
    else if (ndim == 1) {
        rows = 1;
        cols = shape[0];
        channels = 1;
    }
    else throw std::invalid_argument("\"src\" : ndim must be 1, 2 or 3.");

    if (dtype == np::dtype::get_builtin<double_t>()) depth = CV_64F;
    else if (dtype == np::dtype::get_builtin<float_t>()) depth = CV_32F;
    else if (dtype == np::dtype::get_builtin<int32_t>()) depth = CV_32S;
    else if (dtype == np::dtype::get_builtin<u_int16_t>()) depth = CV_16U;
    else if (dtype == np::dtype::get_builtin<int16_t>()) depth = CV_16S;
    else if (dtype == np::dtype::get_builtin<u_int8_t>()) depth = CV_8U;
    else if (dtype == np::dtype::get_builtin<int8_t>()) depth = CV_8S;
    else throw std::invalid_argument("\"src\" : invalid dtype.");

    type = CV_MAKETYPE(depth, channels);
    dst = cv::Mat(rows, cols, type, data);
}

//  深度を用いてフィルタリングを行う
void depth_filter(const pcl::PointCloud<pcl::PointXYZL> &src, pcl::PointCloud<pcl::PointXYZL> &dst, const float_t min, const float_t max)
{
    size_t len = src.points.size();

    if (min < 0.0f || min == NAN) throw std::runtime_error("\"min\" must be longer than 0.");
    if (max <= 0.0f || max == NAN) throw std::runtime_error("\"max\" must be longer than 0.");
    if (len == 0ul) {
        // std::cerr << "[WARNING]: There are no points in \"src\"." << std::endl;
        dst = src;
        return;
    }
    
    if (&src != &dst) {
        dst.points.resize(len);
        dst.width = src.width;
        dst.height = src.height;
        dst.is_dense = src.is_dense;
        dst.header = src.header;
    }

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    pcl::ExtractIndices<pcl::PointXYZL> extract;

    #ifdef _OPENMP
        #pragma omp parallel for
    #endif
    for (size_t i = 0ul; i < len; i++) {
        if (src.points[i].z < min || max < src.points[i].z) {
            #ifdef _OPENMP
                #pragma omp critical
            #endif
            inliers->indices.push_back(i);
        }
    }

    extract.setInputCloud(dst.makeShared());
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(dst);
}

//  並進ベクトルとクォータニオンで座標変換を行う
void transform_pointcloud(const pcl::PointCloud<pcl::PointXYZL> &src, pcl::PointCloud<pcl::PointXYZL> &dst, const Eigen::Vector3f &translation, const Eigen::Quaternionf &quaternion)
{
    size_t len = src.points.size();

    if (len == 0ul) {
        // std::cerr << "[WARNING]: There are no points in \"src\"." << std::endl;
        dst = src;
        return;
    }

    if (translation == Eigen::Vector3f::Zero() && quaternion.x() == 0.0f && quaternion.y() == 0.0f && quaternion.z() == 0.0f && quaternion.w() == 1.0f) {
        dst = src;
        return;
    }
    
    if (&src != &dst) {
        dst.points.resize(len);
        dst.width = src.width;
        dst.height = src.height;
        dst.is_dense = src.is_dense;
        dst.header = src.header;
    }

    #ifdef _OPENMP
        #pragma omp parallel for
    #endif
    for (size_t i = 0ul; i < len; i++) {
        pcl::PointXYZL src_point_pcl = src.points[i];
        Eigen::Vector3f src_point_eigen(src_point_pcl.x, src_point_pcl.y, src_point_pcl.z);
        Eigen::Vector3f dst_point_eigen = quaternion * src_point_eigen + translation;

        dst.points[i] = src_point_pcl;
        dst.points[i].x = dst_point_eigen[axisXYZ::X];
        dst.points[i].y = dst_point_eigen[axisXYZ::Y];
        dst.points[i].z = dst_point_eigen[axisXYZ::Z];
    }
}

//  変換行列で座標変換を行う
void transform_pointcloud(const pcl::PointCloud<pcl::PointXYZL> &src, pcl::PointCloud<pcl::PointXYZL> &dst, const Eigen::Matrix4f &matrix)
{
    size_t len = src.points.size();

    if (len == 0ul) {
        // std::cerr << "[WARNING]: There are no points in \"src\"." << std::endl;
        dst = src;
        return;
    }

    if (matrix == Eigen::Matrix4f::Identity()) {
        dst = src;
        return;
    }
    
    if (&src != &dst) {
        dst.points.resize(len);
        dst.width = src.width;
        dst.height = src.height;
        dst.is_dense = src.is_dense;
        dst.header = src.header;
    }

    #ifdef _OPENMP
        #pragma omp parallel for
    #endif
    for (size_t i = 0ul; i < len; i++) {
        pcl::PointXYZL src_point_pcl = src.points[i];
        Eigen::Vector4f src_point_eigen(src_point_pcl.x, src_point_pcl.y, src_point_pcl.z, 1.0f);
        Eigen::Vector4f dst_point_eigen = matrix * src_point_eigen;

        dst.points[i] = src_point_pcl;
        dst.points[i].x = dst_point_eigen[axisXYZ::X];
        dst.points[i].y = dst_point_eigen[axisXYZ::Y];
        dst.points[i].z = dst_point_eigen[axisXYZ::Z];
    }
}

//  深度マップをカラーマップへ変換
void depth2colormap(const cv::Mat &src, cv::Mat &dst, const float min, const float max, const int type, const bool invert)
{
    if (src.type() != CV_32FC1) throw std::invalid_argument("\"src\" must be CV_32FC1.");

    cv::Mat in_range = src.clone();
    in_range = (in_range - min) / (max - min) * 255.0f;

    cv::Mat min_mask, min_mask_f, max_mask, max_mask_f;
    cv::threshold(src, min_mask_f, min, 255.0, cv::THRESH_BINARY);
    min_mask_f.convertTo(min_mask, CV_8UC1);
    cv::threshold(src, max_mask_f, max, 255.0, cv::THRESH_BINARY_INV);
    max_mask_f.convertTo(max_mask, CV_8UC1);

    cv::Mat in_range_uint8, min_tmp, max_tmp;
    in_range.convertTo(in_range_uint8, CV_8UC1);
    if (invert == true) in_range_uint8 = 255 - in_range_uint8;
    cv::applyColorMap(in_range_uint8, min_tmp, type);

    cv::bitwise_and(min_tmp, min_tmp, max_tmp, min_mask);
    cv::bitwise_and(max_tmp, max_tmp, dst, max_mask);
}

//  深度マップをカラーマップへ変換
np::ndarray depth2colormap_python(const np::ndarray &src, const double_t min, const double_t max, const int type, const bool invert)
{
    cv::Mat depth, colormap;
    ndarray2cvmat(src, depth);
    depth2colormap(depth, colormap, static_cast<float_t>(min), static_cast<float_t>(max), type, invert);
    return cvmat2ndarray(colormap);
}

//  np::ndarrayをEigen::Vector3fへ変換
void ndarray2translation(const np::ndarray &src, Eigen::Vector3f &dst)
{
    if (src.get_nd() != 1) throw std::invalid_argument("\"src.ndim\" must be 1");
    auto t_shape = src.get_shape();
    if (t_shape[0] != 3) throw std::invalid_argument("\"src.shape\" must be (3,)");
    auto t_strides = src.get_strides();
    auto t_dtype = src.get_dtype();

    if (t_dtype == np::dtype::get_builtin<float_t>()) for (size_t y = 0ul; y < 3ul; y++) {
        dst[y] = *reinterpret_cast<float_t *>(src.get_data() + y * t_strides[0]);
    }
    else if (t_dtype == np::dtype::get_builtin<double_t>()) for (size_t y = 0ul; y < 3ul; y++) {
        dst[y] = static_cast<float_t>(*reinterpret_cast<double_t *>(src.get_data() + y * t_strides[0]));
    }
    else throw std::invalid_argument("\"src.dtype\" must be <numpy.float32> or <numpy.float64>");
}

//  np::ndarrayをEigen::Quaternionfへ変換
void ndarray2quaternion(const np::ndarray &src, Eigen::Quaternionf &dst)
{
    if (src.get_nd() != 1) throw std::invalid_argument("\"quaternion.ndim\" must be 1");
    auto q_shape = src.get_shape();
    if (q_shape[0] != 4) throw std::invalid_argument("\"quaternion.shape\" must be (4,)");
    auto q_strides = src.get_strides();
    auto q_dtype = src.get_dtype();

    if (q_dtype == np::dtype::get_builtin<float_t>()) {
        dst = Eigen::Quaternionf(
            *reinterpret_cast<float_t *>(src.get_data() + q_strides[0] * 3),
            *reinterpret_cast<float_t *>(src.get_data()),
            *reinterpret_cast<float_t *>(src.get_data() + q_strides[0]),
            *reinterpret_cast<float_t *>(src.get_data() + q_strides[0] * 2)
        );
    }
    else if (q_dtype == np::dtype::get_builtin<double_t>()) {
        dst = Eigen::Quaternionf(
            static_cast<float_t>(*reinterpret_cast<double_t *>(src.get_data() + q_strides[0] * 3)),
            static_cast<float_t>(*reinterpret_cast<double_t *>(src.get_data())),
            static_cast<float_t>(*reinterpret_cast<double_t *>(src.get_data() + q_strides[0])),
            static_cast<float_t>(*reinterpret_cast<double_t *>(src.get_data() + q_strides[0] * 2))
        );
    }
    else throw std::invalid_argument("\"quaternion.dtype\" must be <numpy.float32> or <numpy.float64>");
}

//  np::ndarrayをEigen::Matrix4fへ変換
void ndarray2matrix(const np::ndarray &src, Eigen::Matrix4f &dst)
{
    if (src.get_nd() != 2) throw std::invalid_argument("\"ndim\" must be 2");

    auto shape = src.get_shape();    
    if (shape[0] != 4 || shape[1] != 4) throw std::invalid_argument("\"shape\" must be (4, 4)");

    dst = Eigen::Matrix4f::Identity();
    auto strides = src.get_strides();
    auto dtype = src.get_dtype();

    if (dtype == np::dtype::get_builtin<float_t>()) for (size_t y = 0ul; y < 4ul; y++) for (size_t x = 0ul; x < 4ul; x++) {
        dst(y, x) = *reinterpret_cast<float_t *>(src.get_data() + y * strides[0] + x * strides[1]);
    }
    else if (dtype == np::dtype::get_builtin<double_t>()) for (size_t y = 0ul; y < 4ul; y++) for (size_t x = 0ul; x < 4ul; x++) {
        dst(y, x) = static_cast<float_t>(*reinterpret_cast<double_t *>(src.get_data() + y * strides[0] + x * strides[1]));
    }
    else throw std::invalid_argument("\"dtype\" must be <numpy.float32> or <numpy.float64>");
}

//  Eigen::Vector3fをnp::ndarrayへ変換
np::ndarray translation2ndarray(const Eigen::Vector3f &src)
{
    Py_intptr_t shape[1] = {3};
    np::ndarray dst = np::zeros(1, shape, np::dtype::get_builtin<float_t>());
    auto strides = dst.get_strides();

    for (size_t y = 0ul; y < 3ul; y++) {
        *reinterpret_cast<float_t *>(dst.get_data() + y * strides[0]) = src[y];
    }
    return dst;
}

//  Eigen::Quaternionfをnp::ndarrayへ変換
np::ndarray quaternion2ndarray(const Eigen::Quaternionf &src)
{
    Py_intptr_t shape[1] = {4};
    np::ndarray dst = np::zeros(1, shape, np::dtype::get_builtin<float_t>());
    auto strides = dst.get_strides();

    *reinterpret_cast<float_t *>(dst.get_data() + strides[0] * 3) = src.w();
    *reinterpret_cast<float_t *>(dst.get_data()) = src.x();
    *reinterpret_cast<float_t *>(dst.get_data() + strides[0]) = src.y();
    *reinterpret_cast<float_t *>(dst.get_data() + strides[0] * 2) = src.z();
    return dst;
}

//  Eigen::Matrix4fをnp::ndarrayへ変換
np::ndarray matrix2ndarray(const Eigen::Matrix4f &src)
{
    Py_intptr_t shape[2] = {4, 4};
    np::ndarray dst = np::zeros(2, shape, np::dtype::get_builtin<float_t>());
    auto strides = dst.get_strides();

    for (size_t y = 0ul; y < 4ul; y++) for (size_t x = 0ul; x < 4ul; x++) {
        *reinterpret_cast<float_t *>(dst.get_data() + y * strides[0] + x * strides[1]) = src(y, x);
    }
    return dst;
}

//  pcl::PointCloudをpoints型のnp::ndarrayへ変換
np::ndarray pointcloud2nppoints(const pcl::PointCloud<pcl::PointXYZL> &src)
{
    size_t src_len = src.points.size();
    Py_intptr_t points_shape[2] = {static_cast<int64_t>(src_len), 3};
    np::ndarray points = np::zeros(2, points_shape, np::dtype::get_builtin<float_t>());
    auto points_strides = points.get_strides();

    for (size_t i = 0ul; i < src_len; i++) {
        *reinterpret_cast<float_t *>(points.get_data() + i * points_strides[0]) = src.points[i].x;
        *reinterpret_cast<float_t *>(points.get_data() + i * points_strides[0] + points_strides[1]) = src.points[i].y;
        *reinterpret_cast<float_t *>(points.get_data() + i * points_strides[0] + 2 * points_strides[1]) = src.points[i].z;
    }

    return points;
}

//  pcl::PointCloudをsemantic3d型のnp::ndarrayへ変換
py::tuple pointcloud2npsemantic3d(const pcl::PointCloud<pcl::PointXYZL> &src)
{
    size_t src_len = src.points.size();

    Py_intptr_t points_shape[2] = {static_cast<int64_t>(src_len), 3};
    Py_intptr_t semantic1d_shape[1] = {static_cast<int64_t>(src_len)};

    np::ndarray points = np::zeros(2, points_shape, np::dtype::get_builtin<float_t>());
    np::ndarray semantic1d = np::zeros(1, semantic1d_shape, np::dtype::get_builtin<u_int8_t>());

    auto points_strides = points.get_strides();
    auto semantic1d_strides = semantic1d.get_strides();

    for (size_t i = 0ul; i < src_len; i++) {
        *reinterpret_cast<float_t *>(points.get_data() + i * points_strides[0]) = src.points[i].x;
        *reinterpret_cast<float_t *>(points.get_data() + i * points_strides[0] + points_strides[1]) = src.points[i].y;
        *reinterpret_cast<float_t *>(points.get_data() + i * points_strides[0] + 2 * points_strides[1]) = src.points[i].z;
        *reinterpret_cast<u_int8_t *>(semantic1d.get_data() + i * semantic1d_strides[0]) = static_cast<u_int8_t>(src.points[i].label);
    }

    return py::make_tuple(points, semantic1d);
}

//  pcl::PointCloudをcompoundへ変換
np::ndarray pointcloud2compound(const pcl::PointCloud<pcl::PointXYZL> &src, const bool label)
{
    py::list dtype_list;
    dtype_list.append(py::make_tuple("x", np::dtype::get_builtin<float_t>()));
    dtype_list.append(py::make_tuple("y", np::dtype::get_builtin<float_t>()));
    dtype_list.append(py::make_tuple("z", np::dtype::get_builtin<float_t>()));

    size_t src_len = src.points.size();
    Py_intptr_t shape[1] = {static_cast<int64_t>(src_len)};

    if (label == true) {
        dtype_list.append(py::make_tuple("label", np::dtype::get_builtin<u_int8_t>()));
        np::dtype dtype = np::dtype(dtype_list);
        np::ndarray compound = np::empty(1, shape, dtype);
        auto strides = compound.get_strides();

        #ifdef _OPENMP
            #pragma omp parallel for
        #endif
        for (size_t i = 0ul; i < src_len; i++) {
            *reinterpret_cast<float_t *>(compound.get_data() + i * strides[0]) = src.points[i].x;
            *reinterpret_cast<float_t *>(compound.get_data() + i * strides[0] + SIZEOF_FLOAT) = src.points[i].y;
            *reinterpret_cast<float_t *>(compound.get_data() + i * strides[0] + SIZEOF_FLOAT * 2) = src.points[i].z;
            *reinterpret_cast<u_int8_t *>(compound.get_data() + i * strides[0] + SIZEOF_FLOAT * 3) = src.points[i].label;
        }
        return compound;
    }
    else {
        np::dtype dtype = np::dtype(dtype_list);
        np::ndarray compound = np::empty(1, shape, dtype);
        auto strides = compound.get_strides();

        #ifdef _OPENMP
            #pragma omp parallel for
        #endif
        for (size_t i = 0ul; i < src_len; i++) {
            *reinterpret_cast<float_t *>(compound.get_data() + i * strides[0]) = src.points[i].x;
            *reinterpret_cast<float_t *>(compound.get_data() + i * strides[0] + SIZEOF_FLOAT) = src.points[i].y;
            *reinterpret_cast<float_t *>(compound.get_data() + i * strides[0] + SIZEOF_FLOAT * 2) = src.points[i].z;
        }
        return compound;
    }
}

//  points型のnp::ndarrayをpcl::PointCloudへ変換
void nppoints2pointcloud(const np::ndarray &points, pcl::PointCloud<pcl::PointXYZL> &dst)
{
    if (points.get_nd() != 2) {
        throw std::invalid_argument("\"points.ndim\" must be 2");
    }
    auto shape = points.get_shape();
    auto strides = points.get_strides();
    auto dtype = points.get_dtype();
    if (shape[1] != 3) {
        throw std::invalid_argument("\"points.shape\" must be (N, 3)");
    }

    if (dtype == np::dtype::get_builtin<float_t>()) {
        dst.resize(shape[0]);

        #ifdef _OPENMP
            #pragma omp parallel for
        #endif
        for (size_t i = 0; i < shape[0]; i++) {
            pcl::PointXYZL point;
            point.x = *reinterpret_cast<float_t *>(points.get_data() + i * strides[0]);
            point.y = *reinterpret_cast<float_t *>(points.get_data() + i * strides[0] + strides[1]);
            point.z = *reinterpret_cast<float_t *>(points.get_data() + i * strides[0] + strides[1] * 2);
            dst[i] = point;
        }
    }
    else if (dtype == np::dtype::get_builtin<double_t>()) {
        dst.resize(shape[0]);

        #ifdef _OPENMP
            #pragma omp parallel for
        #endif
        for (size_t i = 0; i < shape[0]; i++) {
            pcl::PointXYZL point;
            point.x = static_cast<float_t>(*reinterpret_cast<double_t *>(points.get_data() + i * strides[0]));
            point.y = static_cast<float_t>(*reinterpret_cast<double_t *>(points.get_data() + i * strides[0] + strides[1]));
            point.z = static_cast<float_t>(*reinterpret_cast<double_t *>(points.get_data() + i * strides[0] + strides[1] * 2));
            dst[i] = point;
        }
    }
    else {
        throw std::invalid_argument("\"points.dtype\" must be <numpy.float32> or <numpy.float64>");
    }
}

//  semantic3d型のnp::ndarrayをpcl::PointCloudへ変換
void npsemantic3d2pointcloud(const np::ndarray &points, const np::ndarray &semantic1d, pcl::PointCloud<pcl::PointXYZL> &dst)
{
    nppoints2pointcloud(points, dst);

    if (semantic1d.get_nd() != 1) {
        throw std::invalid_argument("\"semantic1d.ndim\" must be 1");
    }
    auto semantic1d_shape = semantic1d.get_shape();
    auto semantic1d_strides = semantic1d.get_strides();
    auto semantic1d_dtype = semantic1d.get_dtype();
    if (dst.points.size() != semantic1d_shape[0]) {
        throw std::invalid_argument("\"points.shape[0]\" == \"semantic1d.shape[0]\"");
    }

    if (semantic1d_dtype == np::dtype::get_builtin<u_int8_t>()) {
        #ifdef _OPENMP
            #pragma omp parallel for
        #endif
        for (size_t i = 0; i < semantic1d_shape[0]; i++) {
            dst.points[i].label = static_cast<u_int32_t>(*reinterpret_cast<u_int8_t *>(semantic1d.get_data() + i * semantic1d_strides[0]));
        }
    }
    else if (semantic1d_dtype == np::dtype::get_builtin<u_int16_t>()) {
        #ifdef _OPENMP
            #pragma omp parallel for
        #endif
        for (size_t i = 0; i < semantic1d_shape[0]; i++) {
            dst.points[i].label = static_cast<u_int32_t>(*reinterpret_cast<u_int16_t *>(semantic1d.get_data() + i * semantic1d_strides[0]));
        }
    }
    else if (semantic1d_dtype == np::dtype::get_builtin<u_int32_t>()) {
        #ifdef _OPENMP
            #pragma omp parallel for
        #endif
        for (size_t i = 0; i < semantic1d_shape[0]; i++) {
            dst.points[i].label = *reinterpret_cast<u_int32_t *>(semantic1d.get_data() + i * semantic1d_strides[0]);
        }
    }
    else {
        throw std::invalid_argument("\"semantic1d.dtype\" must be \"np.uint8\"");
    }
}

//  compoundをpcl::PointCloudへ変換
void compound2pointcloud(const np::ndarray &compound, pcl::PointCloud<pcl::PointXYZL> &dst)
{
    py::list dtype_list;
    dtype_list.append(py::make_tuple("x", np::dtype::get_builtin<float_t>()));
    dtype_list.append(py::make_tuple("y", np::dtype::get_builtin<float_t>()));
    dtype_list.append(py::make_tuple("z", np::dtype::get_builtin<float_t>()));
    np::dtype dtype = np::dtype(dtype_list);

    if (compound.get_dtype() == dtype) {
        auto shape = compound.get_shape();
        auto strides = compound.get_strides();
        dst.points.resize(shape[0]);

        #ifdef _OPENMP
            #pragma omp parallel for
        #endif
        for (size_t i = 0ul; i < shape[0]; i++) {
            dst.points[i].x = *reinterpret_cast<float_t *>(compound.get_data() + i * strides[0]);
            dst.points[i].y = *reinterpret_cast<float_t *>(compound.get_data() + i * strides[0] + SIZEOF_FLOAT);
            dst.points[i].z = *reinterpret_cast<float_t *>(compound.get_data() + i * strides[0] + SIZEOF_FLOAT * 2);
        }
        return;
    }

    dtype_list.append(py::make_tuple("label", np::dtype::get_builtin<u_int8_t>()));
    dtype = np::dtype(dtype_list);

    if (compound.get_dtype() == dtype) {
        auto shape = compound.get_shape();
        auto strides = compound.get_strides();
        dst.points.resize(shape[0]);

        #ifdef _OPENMP
            #pragma omp parallel for
        #endif
        for (size_t i = 0ul; i < shape[0]; i++) {
            dst.points[i].x = *reinterpret_cast<float_t *>(compound.get_data() + i * strides[0]);
            dst.points[i].y = *reinterpret_cast<float_t *>(compound.get_data() + i * strides[0] + SIZEOF_FLOAT);
            dst.points[i].z = *reinterpret_cast<float_t *>(compound.get_data() + i * strides[0] + SIZEOF_FLOAT * 2);
            dst.points[i].label = *reinterpret_cast<u_int8_t *>(compound.get_data() + i * strides[0] + SIZEOF_FLOAT * 3);
        }
        return;
    }

    throw std::invalid_argument("\"compound.dtype\" must be \"[('x', '<f4'), ('y', '<f4'), ('z', '<f4')]\" or \"[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('label', 'u1')]\".");
}


//  [mm]単位の16bit unsigned intの画像を[m]単位の32bit floatの画像に変換
void depth_openni2canonical(const cv::Mat &src, cv::Mat &dst)
{
    if (src.type() == CV_16UC1) {
        src.convertTo(dst, CV_32FC1, 1e-3);
        
        for (int y = 0; y < dst.rows; y++) {
            const u_int16_t *src_ptr = src.ptr<u_int16_t>(y);
            float_t *dst_ptr = dst.ptr<float_t>(y);

            for (int x = 0; x < dst.cols; x++) {
                if (src_ptr[x] == UINT16_MAX) {
                    dst_ptr[x] = INFINITY;
                }
                else if (src_ptr[x] == 0u) {
                    dst_ptr[x] = -INFINITY;
                }
            }
        }
    }
    else if (src.type() == CV_32FC1) {
        dst = src;
    }
    else {
        throw std::runtime_error("\"src.type()\" must be \"CV_16UC1\" or \"CV_32FC1\".");
    }
}

//  複数の変換行列を合成する
void combine_transforms(const std::vector<Eigen::Matrix4f> &src_matrixes, Eigen::Matrix4f &dst_matrix)
{
    dst_matrix = Eigen::Matrix4f::Identity();

    size_t src_len = src_matrixes.size();

    if (src_len < 1) throw std::runtime_error("src_matrixes.size() < 1");

    for (size_t i = 0ul; i < src_len; i++) {
        dst_matrix = src_matrixes[i] * dst_matrix;
    }
}

//  複数の並進ベクトルとクォータニオンを合成する
void combine_transforms(const std::vector<Eigen::Vector3f> &src_translations, const std::vector<Eigen::Quaternionf> &src_quaternions, Eigen::Vector3f &dst_translation, Eigen::Quaternionf &dst_quaternion)
{
    dst_translation = Eigen::Vector3f::Zero();
    dst_quaternion = Eigen::Quaternionf::Identity();

    size_t src_len = src_translations.size();

    if (src_len != src_quaternions.size()) throw std::runtime_error("src_translations.size() != src_quaternions.size()");
    if (src_len < 1) throw std::runtime_error("src_translations.size() < 1 and src_quaternions.size() < 1");

    for (size_t i = 0ul; i < src_len; i++) {
        dst_translation = dst_quaternion * src_translations[i] + dst_translation;
        dst_quaternion = dst_quaternion * src_quaternions[i];
    }
}

//  複数の変換行列を合成する (for Python)
np::ndarray combine_transforms_matrix(const py::list &src_matrixes)
{
    size_t src_matrixes_len = py::len(src_matrixes);
    std::vector<Eigen::Matrix4f> src_matrixes_eigen;

    for (size_t i = 0ul; i < src_matrixes_len; i++) {
        if (py::extract<np::ndarray>(src_matrixes[i]).check() == true) {
            Eigen::Matrix4f matrix_eigen;
            ndarray2matrix(py::extract<np::ndarray>(src_matrixes[i]), matrix_eigen);
            src_matrixes_eigen.push_back(matrix_eigen);
        }
    }

    Eigen::Matrix4f dst_matrix_eigen;
    combine_transforms(src_matrixes_eigen, dst_matrix_eigen);
    return matrix2ndarray(dst_matrix_eigen);
}

//  複数の並進ベクトルとクォータニオンを合成する (for Python)
py::tuple combine_transforms_quaternion(const py::list &src_translations, const py::list &src_quaternions)
{
    size_t src_translations_len = py::len(src_translations);
    std::vector<Eigen::Vector3f> src_translations_eigen;

    for (size_t i = 0ul; i < src_translations_len; i++) {
        if (py::extract<np::ndarray>(src_translations[i]).check() == true) {
            Eigen::Vector3f translation_eigen;
            ndarray2translation(py::extract<np::ndarray>(src_translations[i]), translation_eigen);
            src_translations_eigen.push_back(translation_eigen);
        }
    }

    size_t src_quaternions_len = py::len(src_quaternions);
    std::vector<Eigen::Quaternionf> src_quaternions_eigen;

    for (size_t i = 0ul; i < src_quaternions_len; i++) {
        if (py::extract<np::ndarray>(src_quaternions[i]).check() == true) {
            Eigen::Quaternionf quaternion_eigen;
            ndarray2quaternion(py::extract<np::ndarray>(src_quaternions[i]), quaternion_eigen);
            src_quaternions_eigen.push_back(quaternion_eigen);
        }
    }

    Eigen::Vector3f dst_translation_eigen;
    Eigen::Quaternionf dst_quaternion_eigen;
    combine_transforms(src_translations_eigen, src_quaternions_eigen, dst_translation_eigen, dst_quaternion_eigen);
    return py::make_tuple(translation2ndarray(dst_translation_eigen), quaternion2ndarray(dst_quaternion_eigen));
}

//  Voxelのインデックスをndarrayのtupleへ変換
py::tuple voxelindexs2tuple(const std::vector<point_xyz<size_t> > &voxel_indexs)
{
    size_t indexs_len = voxel_indexs.size();
    Py_intptr_t shape[1] = {indexs_len};
    np::dtype dtype = np::dtype::get_builtin<int64_t>();
    np::ndarray indexs_x = np::empty(1, shape, dtype);
    np::ndarray indexs_y = np::empty(1, shape, dtype);
    np::ndarray indexs_z = np::empty(1, shape, dtype);
    auto strides = indexs_x.get_strides();

    for (size_t i = 0ul; i < indexs_len; i++) {
        *reinterpret_cast<int64_t *>(indexs_x.get_data() + i * strides[0]) = static_cast<int64_t>(voxel_indexs[i].x);
        *reinterpret_cast<int64_t *>(indexs_y.get_data() + i * strides[0]) = static_cast<int64_t>(voxel_indexs[i].y);
        *reinterpret_cast<int64_t *>(indexs_z.get_data() + i * strides[0]) = static_cast<int64_t>(voxel_indexs[i].z);
    }

    return py::make_tuple(indexs_z, indexs_y, indexs_x);
}

}   //  namespace pointsmap
