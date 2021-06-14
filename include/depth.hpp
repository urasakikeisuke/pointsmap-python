#include "base/depth_base.hpp"

namespace pointsmap {

class Depth : public DepthBase
{
    public:

        Depth();

        void set_depthmap(const np::ndarray& src);
        void set_disparity(const np::ndarray& src);
        np::ndarray get_depthmap();

    protected:

        cv::Mat _depthmap;

};

} // pointsmap
