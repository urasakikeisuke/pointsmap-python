#include "common.hpp"

namespace pointsmap {

#ifndef POINTSMAP_BASE_INTRINSIC_HPP_INCLUDE

#define POINTSMAP_BASE_INTRINSIC_HPP_INCLUDE

class IntrinsicBase
{
    public:
        //  Constructor
        IntrinsicBase();

        //  Methods

        //  カメラ内部パラメータを設定する
        template<typename T> void set_intrinsic(const T fx, const T fy, const T cx, const T cy);
        //  カメラ内部パラメータを設定する
        template<typename T> void set_intrinsic(const Eigen::Matrix<T, 3, 3> &K);
        //  カメラ内部パラメータを設定する
        void set_intrinsic_python(const np::ndarray &K);
        //  設定されたカメラ内部パラメータを取得する
        Eigen::Matrix3f get_intrinsic();
        //  設定されたカメラ内部パラメータを取得する
        np::ndarray get_intrinsic_python();
        //  出力する画像のサイズを設定する
        void set_shape(const int height, const int width);
        //  出力する画像のサイズを設定する
        void set_shape_python(const py::tuple &shape);
        //  設定した画像サイズを取得する
        Eigen::Vector2i get_shape();
        //  設定した画像サイズを取得する
        py::tuple get_shape_python();

    protected:
        //  Properties
        float_t _fx = NAN;          //  カメラパラメータ 焦点距離(x軸方向)
        float_t _fy = NAN;          //  カメラパラメータ 焦点距離(y軸方向)
        float_t _cx = NAN;          //  カメラパラメータ 中心座標(x軸方向)
        float_t _cy = NAN;          //  カメラパラメータ 中心座標(y軸方向)

        float_t _tanR = NAN;        //  画角のtangent (右方向)
        float_t _tanL = NAN;        //  画角のtangent (左方向)
        float_t _tanT = NAN;        //  画角のtangent (上方向)
        float_t _tanB = NAN;        //  画角のtangent (下方向)

        int _width = 0;             //  描画する画像の横幅 (int)
        int _height = 0;            //  描画する画像の縦幅 (int)
        float_t _width_f = 0.0f;    //  描画する画像の横幅 (float)
        float_t _height_f = 0.0f;   //  描画する画像の縦幅 (float)
};

#endif

}   //  pointsmap