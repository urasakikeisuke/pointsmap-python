#include <depth.hpp>

namespace pointsmap {

Depth::Depth()
:   DepthBase()
{

}

void Depth::set_depthmap(const np::ndarray& src)
{
    ndarray2cvmat(src, this->_depthmap);
}

void Depth::set_disparity(const np::ndarray& src)
{
    cv::Mat disparity_cv;
    ndarray2cvmat(src, disparity_cv);
    this->create_stereodepth(disparity_cv, this->_depthmap);
}

np::ndarray Depth::get_depthmap()
{
    return cvmat2ndarray(this->_depthmap);
}

} // pointsmap
