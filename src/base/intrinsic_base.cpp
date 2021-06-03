#include "base/intrinsic_base.hpp"

namespace pointsmap {

IntrinsicBase::IntrinsicBase()
{
    //  pass
}

//  カメラ内部パラメータを設定する
template<typename T>
void IntrinsicBase::set_intrinsic(const T fx, const T fy, const T cx, const T cy)
{
    this->_fx = static_cast<float_t>(fx);
    this->_fy = static_cast<float_t>(fy);
    this->_cx = static_cast<float_t>(cx);
    this->_cy = static_cast<float_t>(cy);

    this->_tanR = (this->_width_f - this->_cx) / this->_fx;
    this->_tanL = -this->_cx / this->_fx;
    this->_tanT = -this->_cy / this->_fy;
    this->_tanB = (this->_height_f - this->_cy) / this->_fy;
}

//  カメラ内部パラメータを設定する
template<typename T>
void IntrinsicBase::set_intrinsic(const Eigen::Matrix<T, 3, 3> &K)
{
    float_t fx = K(0, 0);
    float_t fy = K(0, 2);
    float_t cx = K(1, 1);
    float_t cy = K(1, 2);

    this->set_intrinsic(fx, fy, cx, cy);
}

//  カメラ内部パラメータを設定する
void IntrinsicBase::set_intrinsic_python(const np::ndarray &K)
{
    if (K.get_nd() != 2) {
        throw std::invalid_argument("\"ndim\" must be 2");
    }
    auto shape = K.get_shape();
    if (shape[0] != 3 || shape[1] != 3) {
        throw std::invalid_argument("\"shape\" must be (3, 3)");
    }
    auto dtype = K.get_dtype();
    auto strides = K.get_strides();

    float_t fx, fy, cx, cy;

    if (dtype == np::dtype::get_builtin<double_t>()) {
        fx = static_cast<float_t>(*reinterpret_cast<double_t *>(K.get_data()));
        fy = static_cast<float_t>(*reinterpret_cast<double_t *>(K.get_data() + strides[0] + strides[1]));
        cx = static_cast<float_t>(*reinterpret_cast<double_t *>(K.get_data() + 2 * strides[1]));
        cy = static_cast<float_t>(*reinterpret_cast<double_t *>(K.get_data() + strides[0] + 2 * strides[1]));
    }
    else if (dtype == np::dtype::get_builtin<float_t>()) {
        fx = *reinterpret_cast<float_t *>(K.get_data());
        fy = *reinterpret_cast<float_t *>(K.get_data() + strides[0] + strides[1]);
        cx = *reinterpret_cast<float_t *>(K.get_data() + 2 * strides[1]);
        cy = *reinterpret_cast<float_t *>(K.get_data() + strides[0] + 2 * strides[1]);
    }
    else {
        throw std::invalid_argument("\"dtype\" must be <numpy.float32> or <numpy.float64>");
    }

    this->set_intrinsic(fx, fy, cx, cy);
}

//  設定されたカメラ内部パラメータを取得する
Eigen::Matrix3f IntrinsicBase::get_intrinsic()
{
    Eigen::Matrix3f intrinsic = Eigen::Matrix3f::Identity();

    intrinsic(0, 0) = this->_fx;
    intrinsic(0, 2) = this->_cx;
    intrinsic(1, 1) = this->_fy;
    intrinsic(1, 2) = this->_cy;

    return intrinsic;
}

//  設定されたカメラ内部パラメータを取得する
np::ndarray IntrinsicBase::get_intrinsic_python()
{
    Eigen::Matrix3f intrinsic_eigen = this->get_intrinsic();
    
    Py_intptr_t shape[2] = {3, 3};
    np::ndarray intrinsic = np::zeros(2, shape, np::dtype::get_builtin<double_t>());
    
    auto strides = intrinsic.get_strides();

    for (size_t y = 0ul; y < 3ul; y++) for (size_t x = 0ul; x < 3ul; x++) {
        *reinterpret_cast<double_t *>(intrinsic.get_data() + y * strides[0] + x * strides[1]) = static_cast<double_t>(intrinsic_eigen(y, x));
    }

    return intrinsic;
}

//  出力する画像のサイズを設定する
void IntrinsicBase::set_shape(const int height, const int width)
{
    if (height < 1) throw std::invalid_argument("\"height\" must be greater than 0");
    if (width < 1) throw std::invalid_argument("\"width\" must be greater than 0");

    this->_height = height;
    this->_width = width;

    this->_height_f = static_cast<float_t>(this->_height);
    this->_width_f = static_cast<float_t>(this->_width);

    this->_tanR = (this->_width_f - this->_cx) / this->_fx;
    this->_tanL = -this->_cx / this->_fx;
    this->_tanT = -this->_cy / this->_fy;
    this->_tanB = (this->_height_f - this->_cy) / this->_fy;
}

//  出力する画像のサイズを設定する
void IntrinsicBase::set_shape_python(const py::tuple &shape)
{
    auto len_shape = py::len(shape);
    if (len_shape != 2 && len_shape != 3) {
        throw std::invalid_argument("\"len(shape)\" must be 2 or 3");
    }
    int height = py::extract<int>(shape[0]);
    int width = py::extract<int>(shape[1]);

    this->set_shape(height, width);
}

//  設定した画像サイズを取得する
Eigen::Vector2i IntrinsicBase::get_shape()
{
    Eigen::Vector2i shape = {this->_height, this->_width};
    return shape;
}

//  設定した画像サイズを取得する
py::tuple IntrinsicBase::get_shape_python()
{
    return py::make_tuple(this->_height, this->_width);
}

}   //  pointsmap