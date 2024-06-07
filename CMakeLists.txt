cmake_minimum_required(VERSION 3.20...3.26)

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
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

# We are now ready to compile the actual extension module
nanobind_add_module(
  # Name of the extension
  vor_cpp
  
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
  src/vor.cpp
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
    URL http://math.lbl.gov/voro++/download/dir/voro++-0.4.6.tar.gz
    URL_MD5 2338b824c3b7b25590e18e8df5d68af9
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
include_directories(${VOROPP_INCLUDE_DIRS})
target_link_libraries(vor_cpp PRIVATE ${VOROPP_LIBRARIES})
add_dependencies(vor_cpp voro++)

# Install directive for scikit-build-core
install(TARGETS vor_cpp LIBRARY DESTINATION matgeo)
