cmake_minimum_required(VERSION 3.5)

# Include vcpkg toolchain if VCPKG_ROOT is defined
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "")
    message(STATUS "Using vcpkg toolchain file: ${CMAKE_TOOLCHAIN_FILE}")
endif()

if(NOT DEFINED ENV{VCPKG_ROOT})
    message(FATAL_ERROR "VCPKG_ROOT environment variable is not set. Please install vcpkg and set VCPKG_ROOT.")
endif()

project(matgeo LANGUAGES CXX)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

# Try to import all Python components potentially needed by nanobind
if(DEFINED ENV{CONDA_PREFIX})
    set(Python_ROOT_DIR "$ENV{CONDA_PREFIX}")
    message(STATUS "Setting Python hints for conda environment: ${Python_ROOT_DIR}")
    
    # Set Python executable based on platform
    if(WIN32)
        set(Python_EXECUTABLE "$ENV{CONDA_PREFIX}/python.exe")
        # Windows conda environments need explicit library and include paths
        set(Python_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include")
        set(Python_LIBRARY "$ENV{CONDA_PREFIX}/libs/python310.lib")
    else()
        set(Python_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
    endif()
    
    # For conda environments, let CMake find the Python installation automatically
    # by setting the root directory - this is more reliable than manually setting paths
endif()

find_package(Python 3.10
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

# We are now ready to compile the actual extension module
nanobind_add_module(
  # Name of the extension
  voronoi_cpp
  
  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI
  
  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC
  
  # Source code goes here
  src/voronoi.cpp
  )

nanobind_add_module(
  triangulation_cpp
  STABLE_ABI
  NB_STATIC
  src/triangulation.cpp
  )

include(ExternalProject)

## Build and link VORO++
# Use master branch which has CMake support for all platforms

# Set compiler flags for VORO++ based on compiler
set(VORO_CXX_FLAGS "")
set(VORO_C_FLAGS "")
if(MSVC)
    set(VORO_CXX_FLAGS "/wd4244 /wd4996")
    set(VORO_C_FLAGS "/wd4244 /wd4996")
endif()

ExternalProject_Add(voro++
    GIT_REPOSITORY https://github.com/chr1shr/voro.git
    GIT_TAG master
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_FLAGS=${VORO_CXX_FLAGS}
        -DCMAKE_C_FLAGS=${VORO_C_FLAGS}
    BUILD_BYPRODUCTS 
        <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}voro++${CMAKE_STATIC_LIBRARY_SUFFIX}
)
ExternalProject_get_property(voro++ INSTALL_DIR)
set(VOROPP_LIBRARIES ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}voro++${CMAKE_STATIC_LIBRARY_SUFFIX})
set(VOROPP_INCLUDE_DIRS ${INSTALL_DIR}/include)
message(STATUS "Found VORO: ${VOROPP_INCLUDE_DIRS} ${VOROPP_LIBRARIES}")

# Suppress MSVC warnings for our own code when using VORO++
if(MSVC)
    target_compile_options(voronoi_cpp PRIVATE /wd4244 /wd4996)
endif()

target_include_directories(voronoi_cpp PRIVATE ${VOROPP_INCLUDE_DIRS})
target_link_libraries(voronoi_cpp PRIVATE ${VOROPP_LIBRARIES})

## CGAL 
find_package(CGAL REQUIRED)

## OpenMesh
find_package(OpenMesh REQUIRED)

## Surface nets
add_library(SNLib STATIC 
    src/SNLib/MMSurfaceNet.cpp
    src/SNLib/MMCellMap.cpp
    src/SNLib/MMGeometryGL.cpp
    src/SNLib/MMGeometryOBJ.cpp
    src/SNLib/MMCellFlag.cpp
)
target_include_directories(SNLib PUBLIC src/SNLib)

# Dependencies
add_dependencies(voronoi_cpp voro++)
target_include_directories(triangulation_cpp PRIVATE ${CGAL_INCLUDE_DIRS} src/SNLib)
target_link_libraries(triangulation_cpp PRIVATE CGAL::CGAL SNLib OpenMeshCore OpenMeshTools)

# Add required preprocessor definitions for Windows
if(WIN32)
    target_compile_definitions(triangulation_cpp PRIVATE _USE_MATH_DEFINES)
endif()

# Install directive for scikit-build-core
install(TARGETS voronoi_cpp triangulation_cpp LIBRARY DESTINATION matgeo)