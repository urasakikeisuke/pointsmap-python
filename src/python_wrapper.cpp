#include "common.hpp"
#include "voxelgridmap.hpp"
#include "points.hpp"
#include "depth.hpp"

namespace pointsmap {

BOOST_PYTHON_MODULE(libpointsmap)
{
    Py_Initialize();
    np::initialize();

    py::def("invert_transform", invert_transform_matrix);
    py::def("invert_transform", invert_transform_quaternion);
    py::def("matrix_to_quaternion", matrix2quaternion_python);
    py::def("quaternion_to_matrix", quaternion2matrix_python);
    py::def("depth_to_colormap", depth2colormap_python);
    py::def("combine_transforms", combine_transforms_matrix);
    py::def("combine_transforms", combine_transforms_quaternion);

    py::class_<IntrinsicBase>("intrinsicbase")
        //  Methods
        .def("set_intrinsic", &IntrinsicBase::set_intrinsic_python)
        .def("get_intrinsic", &IntrinsicBase::get_intrinsic_python)
        .def("set_shape", &IntrinsicBase::set_shape_python)
        .def("get_shape", &IntrinsicBase::get_shape_python)
    ;

    py::class_<DepthBase, py::bases<IntrinsicBase> >("depthbase")
        //  Methods
        .def("set_depth_range", &DepthBase::set_depth_range_python)
        .def("get_depth_range", &DepthBase::get_depth_range_python)
        .def("set_base_line", &DepthBase::set_base_line_python)
        .def("get_base_line", &DepthBase::get_base_line_python)
    ;

    py::class_<PointsBase, py::bases<DepthBase>, boost::shared_ptr<PointsBase> >("pointsbase", py::no_init)
        //  Constructor
        .def("__init__", py::make_constructor(&PointsBase_init, py::default_call_policies(), (py::arg("quiet")=false)))
        //  Methods
        .def("set_points", &PointsBase::set_pointsfile_python)
        .def("set_points", &PointsBase::set_pointsfiles_python)
        .def("set_points", &PointsBase::set_points_from_numpy)
        .def("set_semanticpoints", &PointsBase::set_semanticpoints_from_numpy)
        .def("get_points", &PointsBase::get_points_python)
        .def("get_semanticpoints", &PointsBase::get_semanticpoints_python)
    ;

    py::class_<Depth, py::bases<DepthBase>, boost::shared_ptr<Depth> >("depth", py::no_init)
        //  Methods
        .def("set_depthmap", &Depth::set_depthmap)
        .def("set_disparity", &Depth::set_disparity)
        .def("get_depthmap", &Depth::get_depthmap)
    ;

    py::class_<VoxelGridMap, py::bases<PointsBase>, boost::shared_ptr<VoxelGridMap> >("voxelgridmap", py::no_init)
        //  Constructor
        .def("__init__", py::make_constructor(&VoxelGridMap_init, py::default_call_policies(), (py::arg("quiet")=false)))
        //  Methods
        .def("set_pointsmap", &VoxelGridMap::set_pointsfile_python)
        .def("set_pointsmap", &VoxelGridMap::set_pointsfiles_python)
        .def("set_pointsmap", &VoxelGridMap::set_points_from_numpy)
        .def("set_semanticmap", &VoxelGridMap::set_semanticpoints_from_numpy)
        .def("set_voxelgridmap", &VoxelGridMap::set_voxelgridmap_python)
        .def("set_empty_voxelgridmap", &VoxelGridMap::set_empty_voxelgridmap_python)
        .def("get_pointsmap", &VoxelGridMap::get_points_python)
        .def("get_semanticmap", &VoxelGridMap::get_semanticpoints_python)
        .def("get_voxelgridmap", &VoxelGridMap::get_voxelgridmap_python)
        .def("get_voxel_size", &VoxelGridMap::get_voxel_size_python)
        .def("get_voxels_min", &VoxelGridMap::get_voxels_min_python)
        .def("get_voxels_max", &VoxelGridMap::get_voxels_max_python)
        .def("get_voxels_center", &VoxelGridMap::get_voxels_center_python)
        .def("get_voxels_origin", &VoxelGridMap::get_voxels_origin_python)
        .def("get_voxels_include_frustum", &VoxelGridMap::voxels_include_frustum_from_matrix)
        .def("get_voxels_include_frustum", &VoxelGridMap::voxels_include_frustum_from_quaternion)
        .def("set_voxels", &VoxelGridMap::set_voxels_python)
        .def("create_depthmap", &VoxelGridMap::create_depthmap_from_matrix)
        .def("create_depthmap", &VoxelGridMap::create_depthmap_from_quaternion)
        .def("create_semantic2d", &VoxelGridMap::create_semantic2d_from_matrix)
        .def("create_semantic2d", &VoxelGridMap::create_semantic2d_from_quaternion)
    ;

    py::class_<Points, py::bases<PointsBase>, boost::shared_ptr<Points> >("points", py::no_init)
        //  Constructor
        .def("__init__", py::make_constructor(&Points_init, py::default_call_policies(), (py::arg("quiet")=false)))
        //  Methods
        .def("create_depthmap", &Points::create_depthmap_from_matrix)
        .def("create_depthmap", &Points::create_depthmap_from_quaternion)
        .def("create_semantic2d", &Points::create_semantic2d_from_matrix)
        .def("create_semantic2d", &Points::create_semantic2d_from_quaternion)
        .def("set_depthmap", &Points::set_depthmap_matrix)
        .def("set_depthmap", &Points::set_depthmap_quaternion)
        .def("set_depthmap_semantic2d", &Points::set_depthmap_semantic2d_matrix)
        .def("set_depthmap_semantic2d", &Points::set_depthmap_semantic2d_quaternion)
        .def("transform", &Points::transform_matrix)
        .def("transform", &Points::transform_quaternion)
        .def("add_points", &Points::add_pointsfile_python)
        .def("add_points", &Points::add_pointsfiles_python)
        .def("add_points", &Points::add_points_from_numpy)
        .def("add_semanticpoints", &Points::add_semanticpoints_from_numpy)
        .def("downsampling", &Points::downsampling)
    ;

}

}   //  namespace pointsmap
