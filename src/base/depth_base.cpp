#include "base/depth_base.hpp"

namespace pointsmap {

DepthBase::DepthBase()
:   IntrinsicBase()
{
    //  pass
}

//  出力する深度の範囲を設定する
void DepthBase::set_depth_range(const float_t min, const float_t max)
{
    if (min < 0.0f) throw std::invalid_argument("\"min\" must be 0.0 or higher");
    if (max <= 0.0f) throw std::invalid_argument("\"max\" must be greater than 0.0");
    if (min >= max) throw std::invalid_argument("\"max\" must be greater than \"min\"");

    this->_depth_min = min;
    this->_depth_max = max;
}

//  出力する深度の範囲を設定する
void DepthBase::set_depth_range_python(const py::tuple &depth_range)
{
    auto len_depth_range = py::len(depth_range);
    if (len_depth_range != 2) {
        throw std::invalid_argument("\"len(depth_range)\" must be 2");
    }

    this->set_depth_range(py::extract<float_t>(depth_range[0]), py::extract<float_t>(depth_range[1]));
}

//  設定した深度の範囲を取得する
Eigen::Vector2f DepthBase::get_depth_range()
{
    Eigen::Vector2f depth_range = {this->_depth_min, this->_depth_max};
    return depth_range;
}

//  設定した深度の範囲を取得する
py::tuple DepthBase::get_depth_range_python()
{
    return py::make_tuple(this->_depth_min, this->_depth_max);
}

//  ステレオカメラのカメラ間距離を設定する
void DepthBase::set_base_line(const float_t base_line)
{
    this->_base_line = base_line;
}

//  ステレオカメラのカメラ間距離を設定する (for Python)
void DepthBase::set_base_line_python(const double_t base_line)
{
    this->set_base_line(static_cast<float_t>(base_line));
}

//  設定したステレオカメラのカメラ間距離を取得する
float_t DepthBase::get_base_line()
{
    return this->_base_line;
}

//  設定したステレオカメラのカメラ間距離を取得する (for Python)
double_t DepthBase::get_base_line_python()
{
    return static_cast<double_t>(this->get_base_line());
}

//  点群から深度マップを生成する
void DepthBase::create_depthmap(const pcl::PointCloud<pcl::PointXYZL> &src, cv::Mat &dst)
{
    if (this->_fx == NAN) throw std::runtime_error("Fx must not be nan.");
    if (this->_fy == NAN) throw std::runtime_error("Fy must not be nan.");
    if (this->_cx == NAN) throw std::runtime_error("Cx must not be nan.");
    if (this->_cy == NAN) throw std::runtime_error("Cy must not be nan.");

    dst = cv::Mat(this->_height, this->_width, CV_32FC1, cv::Scalar_<float_t>(INFINITY));

    size_t len = src.points.size();

    std::vector<cv::Mat> processing_dst;
    int threads;
    #ifdef _OPENMP
        threads = omp_get_max_threads();
        processing_dst.assign(threads, dst.clone());
        #pragma omp parallel
    #else
        threads = 1;
        processing_dst.assign(threads, dst.clone());
    #endif
    {
        #ifdef _OPENMP
            #pragma omp for
        #endif
        for (size_t i = 0ul; i < len; i++) {
            pcl::PointXYZL src_point = src.points[i];

            float_t th_R = src_point.z * this->_tanR;
            float_t th_L = src_point.z * this->_tanL;
            float_t th_T = src_point.z * this->_tanT;
            float_t th_B = src_point.z * this->_tanB;

            if (th_L <= src_point.x && src_point.x <= th_R && th_T <= src_point.y && src_point.y <= th_B) {
                int x, y;

                if (src_point.x < 0.0f) x = static_cast<int>(this->_cx - roundf(this->_cx * src_point.x / (src_point.z * this->_tanL)));
                else x = static_cast<int>(this->_cx + roundf((this->_width_f - this->_cx) * src_point.x / (src_point.z * this->_tanR)));

                if (src_point.y < 0.0f) y = static_cast<int>(this->_cy - roundf(this->_cy * src_point.y / (src_point.z * this->_tanT)));
                else y = static_cast<int>(this->_cy + roundf((this->_height_f - this->_cy) * src_point.y / (src_point.z * this->_tanB)));

                if (0 <= x && x < this->_width && 0 <= y && y < this->_height) {
                    #ifdef _OPENMP
                        float_t *cv_dst_ptr = processing_dst[omp_get_thread_num()].ptr<float_t>(y);
                    #else
                        float_t *cv_dst_ptr = processing_dst[0].ptr<float_t>(y);
                    #endif
                    if (cv_dst_ptr[x] > src_point.z) {
                        cv_dst_ptr[x] = src_point.z;
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
void DepthBase::create_semantic2d(const pcl::PointCloud<pcl::PointXYZL> &src, cv::Mat &dst, cv::Mat &depth)
{
    if (this->_fx == NAN) throw std::runtime_error("Fx must not be nan.");
    if (this->_fy == NAN) throw std::runtime_error("Fy must not be nan.");
    if (this->_cx == NAN) throw std::runtime_error("Cx must not be nan.");
    if (this->_cy == NAN) throw std::runtime_error("Cy must not be nan.");

    dst = cv::Mat::zeros(this->_height, this->_width, CV_8UC1);
    depth = cv::Mat(this->_height, this->_width, CV_32FC1, cv::Scalar_<float_t>(INFINITY));

    size_t len = src.points.size();

    std::vector<cv::Mat> processing_dst;
    std::vector<cv::Mat> processing_depth;
    int threads;
    #ifdef _OPENMP
        threads = omp_get_max_threads();
        processing_dst.assign(threads, dst.clone());
        processing_depth.assign(threads, depth.clone());
        #pragma omp parallel
    #else
        threads = 1;
        processing_dst.assign(threads, dst.clone());
        processing_depth.assign(threads, depth.clone());
    #endif
    {
        #ifdef _OPENMP
            #pragma omp for
        #endif
        for (size_t i = 0ul; i < len; i++) {
            pcl::PointXYZL src_point = src.points[i];

            float_t th_R = src_point.z * this->_tanR;
            float_t th_L = src_point.z * this->_tanL;
            float_t th_T = src_point.z * this->_tanT;
            float_t th_B = src_point.z * this->_tanB;

            if (th_L <= src_point.x && src_point.x <= th_R && th_T <= src_point.y && src_point.y <= th_B) {
                int x, y;

                if (src_point.x < 0.0f) x = static_cast<int>(this->_cx - roundf(this->_cx * src_point.x / (src_point.z * this->_tanL)));
                else x = static_cast<int>(this->_cx + roundf((this->_width_f - this->_cx) * src_point.x / (src_point.z * this->_tanR)));

                if (src_point.y < 0.0f) y = static_cast<int>(this->_cy - roundf(this->_cy * src_point.y / (src_point.z * this->_tanT)));
                else y = static_cast<int>(this->_cy + roundf((this->_height_f - this->_cy) * src_point.y / (src_point.z * this->_tanB)));

                if (0 <= x && x < this->_width && 0 <= y && y < this->_height) {
                    #ifdef _OPENMP
                        float_t *cv_depth_ptr = processing_depth[omp_get_thread_num()].ptr<float_t>(y);
                        u_int8_t *cv_dst_ptr = processing_dst[omp_get_thread_num()].ptr<u_int8_t>(y);
                    #else
                        float_t *cv_depth_ptr = processing_depth[0].ptr<float_t>(y);
                        u_int8_t *cv_dst_ptr = processing_dst[0].ptr<u_int8_t>(y);
                    #endif
                    if (cv_depth_ptr[x] > src_point.z) {
                        cv_depth_ptr[x] = src_point.z;
                        cv_dst_ptr[x] = static_cast<u_int8_t>(src_point.label);
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

//  深度マップから三次元点群を生成する
void DepthBase::create_points_from_depthmap(const cv::Mat &depthmap, pcl::PointCloud<pcl::PointXYZL> &dst_points)
{
    if (this->_fx == NAN) throw std::runtime_error("Fx must not be nan.");
    if (this->_fy == NAN) throw std::runtime_error("Fy must not be nan.");
    if (this->_cx == NAN) throw std::runtime_error("Cx must not be nan.");
    if (this->_cy == NAN) throw std::runtime_error("Cy must not be nan.");

    if (this->_width != depthmap.cols) throw std::runtime_error("this->_width != depthmap.cols");
    if (this->_height != depthmap.rows) throw std::runtime_error("this->_height != depthmap.rows");

    cv::Mat depth_tmp;
    depth_openni2canonical(depthmap, depth_tmp);

    dst_points.resize(this->_width * this->_height);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    pcl::ExtractIndices<pcl::PointXYZL> extract;

    #ifdef _OPENMP
        #pragma omp parallel for
    #endif
    for (int y = 0; y < this->_height; y++) {
        float_t *depth_tmp_ptr = depth_tmp.ptr<float_t>(y);

        for (int x = 0; x < this->_width; x++) {
            float_t *depth_tmp_px = &(depth_tmp_ptr[x]);

            if (std::isinf(*depth_tmp_px) == true || std::isnan(*depth_tmp_px) == true ||
                *depth_tmp_px < this->_depth_min || this->_depth_max < *depth_tmp_px) {
                #ifdef _OPENMP
                    #pragma omp critical
                #endif
                inliers->indices.push_back(y * this->_width + x);
            }
            else {
                pcl::PointXYZL *point = &(dst_points.points[y * this->_width + x]);
                point->z = *depth_tmp_px;
                point->x = point->z * (static_cast<float_t>(x) - this->_cx) / this->_fx;
                point->y = point->z * (static_cast<float_t>(y) - this->_cy) / this->_fy;
            }
        }
    }

    extract.setInputCloud(dst_points.makeShared());
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(dst_points);
}

//  深度マップとSemantic2Dからラベル付き三次元点群を生成する
void DepthBase::create_points_from_depthmap_semantic2d(const cv::Mat &depthmap, const cv::Mat &semantic2d, pcl::PointCloud<pcl::PointXYZL> &dst_points)
{
    if (this->_fx == NAN) throw std::runtime_error("Fx must not be nan.");
    if (this->_fy == NAN) throw std::runtime_error("Fy must not be nan.");
    if (this->_cx == NAN) throw std::runtime_error("Cx must not be nan.");
    if (this->_cy == NAN) throw std::runtime_error("Cy must not be nan.");

    if (this->_width != depthmap.cols) throw std::runtime_error("this->_width != depthmap.cols");
    if (this->_height != depthmap.rows) throw std::runtime_error("this->_height != depthmap.rows");

    if (this->_width != semantic2d.cols) throw std::runtime_error("this->_width != semantic2d.cols");
    if (this->_height != semantic2d.rows) throw std::runtime_error("this->_height != semantic2d.rows");

    if (semantic2d.type() != CV_8UC1) {
        throw std::runtime_error("\"semantic2d.type()\" must be \"CV_8UC1\"");
    }

    cv::Mat depth_tmp;
    depth_openni2canonical(depthmap, depth_tmp);

    dst_points.resize(depth_tmp.cols * depth_tmp.rows);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    pcl::ExtractIndices<pcl::PointXYZL> extract;

    #ifdef _OPENMP
        #pragma omp parallel for
    #endif
    for (int y = 0; y < this->_height; y++) {
        float_t *depth_tmp_ptr = depth_tmp.ptr<float_t>(y);
        const u_int8_t *semantic2d_ptr = semantic2d.ptr<u_int8_t>(y);

        for (int x = 0; x < this->_width; x++) {
            float_t *depth_tmp_px = &(depth_tmp_ptr[x]);

            if (std::isinf(*depth_tmp_px) == true || std::isnan(*depth_tmp_px) == true ||
                *depth_tmp_px < this->_depth_min || this->_depth_max < *depth_tmp_px) {
                #ifdef _OPENMP
                    #pragma omp critical
                #endif
                inliers->indices.push_back(y * this->_width + x);
            }
            else {
                pcl::PointXYZL *point = &(dst_points.points[y * this->_width + x]);
                point->z = *depth_tmp_px;
                point->x = point->z * (static_cast<float_t>(x) - this->_cx) / this->_fx;
                point->y = point->z * (static_cast<float_t>(y) - this->_cy) / this->_fy;
                point->label = static_cast<u_int32_t>(semantic2d_ptr[x]);
            }
        }
    }

    extract.setInputCloud(dst_points.makeShared());
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(dst_points);
}

//  Semantic2Dを用いて三次元点群にラベルを付与する
void DepthBase::set_semanticlabel_from_semantic2d(const cv::Mat &semantic2d, pcl::PointCloud<pcl::PointXYZL> &target_points)
{
    throw std::runtime_error("set_semanticlabel_from_semantic2d : Not Implemented");
}

//  本来見えない位置にある点をDepthから除去する
void DepthBase::depth_visibility_filter(const cv::Mat &src, cv::Mat &dst, const float_t threshold, const int radius)
{
    if (this->_fx == NAN) throw std::runtime_error("Fx must not be nan.");
    if (this->_fy == NAN) throw std::runtime_error("Fy must not be nan.");
    if (this->_cx == NAN) throw std::runtime_error("Cx must not be nan.");
    if (this->_cy == NAN) throw std::runtime_error("Cy must not be nan.");

    if (this->_width != src.cols) throw std::runtime_error("this->_width != src.cols");
    if (this->_height != src.rows) throw std::runtime_error("this->_height != src.rows");

    if (src.type() != CV_32FC1) throw std::runtime_error("src.type() != CV_32FC1");

    if (radius < 1) throw std::runtime_error("\"radius\" must be larger then 0.");

    dst = src.clone();

    #ifdef _OPENMP
        #pragma omp parallel for
    #endif
    for (int y = 0; y < this->_height; y++) {
        const float_t *src_ptr = src.ptr<float_t>(y);

        for (int x = 0; x < this->_width; x++) {
            const float_t *src_px = &(src_ptr[x]);
            if (std::isinf(*src_px) == true || std::isnan(*src_px) == true) continue;

            Eigen::Vector3f src_point(
                (static_cast<float_t>(x) - this->_cx) * (*src_px) / this->_fx,
                (static_cast<float_t>(y) - this->_cy) * (*src_px) / this->_fy,
                *src_px
            );

            Eigen::Vector3f src_point_normalized = -src_point.normalized();

            std::vector<img_scan> kernels{
                {.x_begin = x - radius, .x_end = x, .y_begin = y - radius, .y_end = y},
                {.x_begin = x, .x_end = x + radius, .y_begin = y - radius, .y_end = y},
                {.x_begin = x - radius, .x_end = x, .y_begin = y, .y_end = y + radius},
                {.x_begin = x, .x_end = x + radius, .y_begin = y, .y_end = y + radius}
            };
            std::vector<float_t> max_corn_angles(kernels.size(), -1.0f);

            for (size_t n = 0ul; n < kernels.size(); n++) {
                img_scan *kernel = &(kernels[n]);
                float_t *max_corn_angle = &(max_corn_angles[n]);

                for (int j = kernel->y_begin; j <= kernel->y_end; j++) {
                    if (j < 0 || this->_height <= j) continue;
                    const float_t *tmp_ptr = src.ptr<float_t>(j);

                    for (int i = kernel->x_begin; i <= kernel->x_end; i++) {
                        if (i < 0 || this->_width <= i) continue;
                        const float_t *tmp_px = &(tmp_ptr[x]);

                        Eigen::Vector3f tmp_point = Eigen::Vector3f(
                            (static_cast<float_t>(i) - this->_cx) * (*tmp_px) / this->_fx,
                            (static_cast<float_t>(j) - this->_cy) * (*tmp_px) / this->_fy,
                            *tmp_px
                        ) - src_point;

                        tmp_point.normalize();

                        float_t dot = tmp_point.dot(src_point_normalized);

                        if (dot > *max_corn_angle) *max_corn_angle = dot;
                    }
                }
            }

            if (std::accumulate(max_corn_angles.begin(), max_corn_angles.end(), 0.0f) >= threshold) {
                dst.ptr<float_t>(y)[x] = INFINITY;
            }
        }
    }
}

//  本来見えない位置にある点をSemantic2Dから除去する
void DepthBase::semantic2d_visibility_filter(const cv::Mat &src_depth, cv::Mat &target_semantic2d, const float_t threshold, const int radius)
{
    if (this->_fx == NAN) throw std::runtime_error("Fx must not be nan.");
    if (this->_fy == NAN) throw std::runtime_error("Fy must not be nan.");
    if (this->_cx == NAN) throw std::runtime_error("Cx must not be nan.");
    if (this->_cy == NAN) throw std::runtime_error("Cy must not be nan.");

    if (this->_width != src_depth.cols) throw std::runtime_error("this->_width != src_depth.cols");
    if (this->_height != src_depth.rows) throw std::runtime_error("this->_height != src_depth.rows");

    if (this->_width != target_semantic2d.cols) throw std::runtime_error("this->_width != target_semantic2d.cols");
    if (this->_height != target_semantic2d.rows) throw std::runtime_error("this->_height != target_semantic2d.rows");

    if (src_depth.type() != CV_32FC1) throw std::runtime_error("src_depth.type() != CV_32FC1");
    if (target_semantic2d.type() != CV_8UC1) throw std::runtime_error("target_semantic2d.type() != CV_8UC1");

    if (radius < 1) throw std::runtime_error("\"radius\" must be larger then 0.");

    #ifdef _OPENMP
        #pragma omp parallel for
    #endif
    for (int y = 0; y < this->_height; y++) {
        const float_t *src_ptr = src_depth.ptr<float_t>(y);

        for (int x = 0; x < this->_width; x++) {
            const float_t *src_px = &(src_ptr[x]);
            if (std::isinf(*src_px) == true || std::isnan(*src_px) == true) continue;

            Eigen::Vector3f src_point(
                (static_cast<float_t>(x) - this->_cx) * (*src_px) / this->_fx,
                (static_cast<float_t>(y) - this->_cy) * (*src_px) / this->_fy,
                *src_px
            );

            Eigen::Vector3f src_point_normalized = -src_point.normalized();

            std::vector<img_scan> kernels{
                {.x_begin = x - radius, .x_end = x, .y_begin = y - radius, .y_end = y},
                {.x_begin = x, .x_end = x + radius, .y_begin = y - radius, .y_end = y},
                {.x_begin = x - radius, .x_end = x, .y_begin = y, .y_end = y + radius},
                {.x_begin = x, .x_end = x + radius, .y_begin = y, .y_end = y + radius}
            };
            std::vector<float_t> max_corn_angles(kernels.size(), -1.0f);

            for (size_t n = 0ul; n < kernels.size(); n++) {
                img_scan *kernel = &(kernels[n]);
                float_t *max_corn_angle = &(max_corn_angles[n]);

                for (int j = kernel->y_begin; j <= kernel->y_end; j++) {
                    if (j < 0 || this->_height <= j) continue;
                    const float_t *tmp_ptr = src_depth.ptr<float_t>(j);

                    for (int i = kernel->x_begin; i <= kernel->x_end; i++) {
                        if (i < 0 || this->_width <= i) continue;
                        const float_t *tmp_px = &(tmp_ptr[x]);

                        Eigen::Vector3f tmp_point = Eigen::Vector3f(
                            (static_cast<float_t>(i) - this->_cx) * (*tmp_px) / this->_fx,
                            (static_cast<float_t>(j) - this->_cy) * (*tmp_px) / this->_fy,
                            *tmp_px
                        ) - src_point;

                        tmp_point.normalize();

                        float_t dot = tmp_point.dot(src_point_normalized);

                        if (dot > *max_corn_angle) *max_corn_angle = dot;
                    }
                }
            }

            if (std::accumulate(max_corn_angles.begin(), max_corn_angles.end(), 0.0f) >= threshold) {
                target_semantic2d.ptr<u_int8_t>(y)[x] = 0u;
            }
        }
    }
}

//  Disparityから深度マップを生成する
void DepthBase::create_stereodepth(const cv::Mat& disparity, cv::Mat& dst)
{
    if (disparity.type() != CV_32FC1) throw std::runtime_error("disparity.type() != CV_32FC1");

    if (this->_fx == NAN) throw std::runtime_error("Fx must not be nan.");
    if (this->_fy == NAN) throw std::runtime_error("Fy must not be nan.");
    if (this->_base_line == NAN) throw std::runtime_error("\"base_line\" must not be nan.");

    if (this->_width != disparity.cols) throw std::runtime_error("this->_width != disparity.cols");
    if (this->_height != disparity.rows) throw std::runtime_error("this->_height != disparity.rows");

    float_t focal_length = (this->_fx + this->_fy) * 0.5f;

    dst = cv::Mat(this->_height, this->_width, CV_32FC1, cv::Scalar_<float_t>(INFINITY));

    for (size_t y = 0ul; y < this->_height; y++) {
        const float_t *disparity_ptr = disparity.ptr<float_t>(y);
        float_t *dst_ptr = dst.ptr<float_t>(y);

        for (size_t x = 0ul; x < this->_width; x++) {
            float_t depth = this->_base_line * focal_length / disparity_ptr[x];

            if ((this->_depth_min <= depth) && (depth <= this->_depth_max)) {
                dst_ptr[x] = depth;
            }
        }
    }
}

}   //  pointsmap
