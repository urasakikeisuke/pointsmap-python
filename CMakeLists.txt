project(pointsmap)
cmake_minimum_required(VERSION 2.8)

cmake_policy(SET CMP0057 NEW)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -g3")

## OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

### pkgconfig (for pkg_check_modules)
find_package(PkgConfig REQUIRED)

### Python includes
pkg_check_modules(PYTHON3 python3 REQUIRED)
include_directories(${PYTHON3_INCLUDE_DIRS})
#find_package (Python3 COMPONENTS Interpreter Development)
find_package(PythonExtensions REQUIRED)

##  Boost includes
find_package(Boost COMPONENTS python numpy REQUIRED)

##  OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

##  PCL includes
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

##  Build
include_directories(include)
add_library(pointsmap SHARED src/common.cpp src/base/intrinsic_base.cpp src/base/depth_base.cpp src/base/points_base.cpp src/voxelgridmap.cpp src/points.cpp src/python_wrapper.cpp)

target_link_libraries(pointsmap ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

##  Install
install(TARGETS pointsmap LIBRARY DESTINATION pointsmap)
