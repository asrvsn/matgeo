cmake_minimum_required(VERSION 3.20...3.26)

# Include vcpkg toolchain if VCPKG_ROOT is defined
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "")
    message(STATUS "Using vcpkg toolchain file: ${CMAKE_TOOLCHAIN_FILE}")
endif()

project(matgeo LANGUAGES CXX)

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
# See LAMMPS cmake files: https://github.com/lammps/lammps/blob/7ef9a93a75e873484d9aad8119e91c461f130d1c/cmake/Modules/Packages/VORONOI.cmake#L9

set(VORO_BUILD_CFLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${BTYPE}}")
if(APPLE)
  get_filename_component(VORO_CXX ${CMAKE_CXX_COMPILER} NAME_WE)
  set(VORO_BUILD_OPTIONS CXX=${VORO_CXX} CFLAGS=${VORO_BUILD_CFLAGS})
else()
  set(VORO_BUILD_OPTIONS CXX=${CMAKE_CXX_COMPILER} CFLAGS=${VORO_BUILD_CFLAGS})
endif()

ExternalProject_Add(voro++
    URL https://github.com/chr1shr/voro/archive/refs/tags/v0.4.6.tar.gz
    CONFIGURE_COMMAND "" 
    BUILD_COMMAND make ${VORO_BUILD_OPTIONS} 
    BUILD_IN_SOURCE 1 
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS <SOURCE_DIR>/src/libvoro++.a
    )
ExternalProject_get_property(voro++ SOURCE_DIR)
set(VOROPP_LIBRARIES ${SOURCE_DIR}/src/libvoro++.a)
set(VOROPP_INCLUDE_DIRS ${SOURCE_DIR}/src)
message(STATUS "Found VORO: ${VOROPP_INCLUDE_DIRS} ${VOROPP_LIBRARIES}")
target_include_directories(voronoi_cpp PRIVATE ${VOROPP_INCLUDE_DIRS})
target_link_libraries(voronoi_cpp PRIVATE ${VOROPP_LIBRARIES})

## CGAL 
find_package(CGAL REQUIRED)

# Dependencies
add_dependencies(voronoi_cpp voro++)
target_include_directories(triangulation_cpp PRIVATE ${Boost_INCLUDE_DIRS} ${CGAL_INCLUDE_DIRS})
target_link_libraries(triangulation_cpp PRIVATE CGAL::CGAL)

# Install directive for scikit-build-core
install(TARGETS voronoi_cpp triangulation_cpp LIBRARY DESTINATION matgeo)