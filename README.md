nanobind_example
================

|      CI              | status |
|----------------------|--------|
| pip builds           | [![Pip Action Status][actions-pip-badge]][actions-pip-link] |
| wheels               | [![Wheel Action Status][actions-wheels-badge]][actions-wheels-link] |

[actions-pip-link]:        https://github.com/wjakob/nanobind_example/actions?query=workflow%3APip
[actions-pip-badge]:       https://github.com/wjakob/nanobind_example/workflows/Pip/badge.svg
[actions-wheels-link]:     https://github.com/wjakob/nanobind_example/actions?query=workflow%3AWheels
[actions-wheels-badge]:    https://github.com/wjakob/nanobind_example/workflows/Wheels/badge.svg


This repository contains a tiny project showing how to create C++ bindings
using [nanobind](https://github.com/wjakob/nanobind) and
[scikit-build-core](https://scikit-build-core.readthedocs.io/en/latest/index.html). It
was derived from the corresponding _pybind11_ [example
project](https://github.com/pybind/scikit_build_example/) developed by
[@henryiii](https://github.com/henryiii).

Installation
------------

### Build Dependencies

#### Required for all platforms:
- [Miniforge3](https://github.com/conda-forge/miniforge) (recommended) or another conda distribution
- [vcpkg](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started?pivots=shell-bash) and set the `VCPKG_ROOT` environment variable
- CMake (automatically handled as build dependency, but if you encounter issues, install manually):
  - **macOS**: `brew install cmake`
  - **Ubuntu/Debian**: `sudo apt-get install cmake`
  - **Windows**: Download from [cmake.org](https://cmake.org/download/)

#### Platform-specific requirements:
- **Windows only**: [Visual Studio](https://visualstudio.microsoft.com/downloads/) with the "Desktop development with C++" workload selected

### Installation Steps

First, create and activate a conda environment if not already in one:
```bash
conda create -n segmentation python=3.10
conda activate segmentation
```

Some dependencies have to be installed globally in the environment.
```bash
pip install nanobind scikit-build-core
```

For a complete installation, follow these steps in order:

```bash
# 1. Clone the repository
git clone <repository-url>
cd matgeo

pip install --no-build-isolation -ve .
```

Some dependencies require special handling due to dependency conflicts:

**Note**: When installing `pyclesperanto-prototype` with `--no-deps`, ensure that your environment already has the necessary dependencies (like OpenCL drivers) for it to function properly.

**pyclesperanto-prototype**: This package has dependency constraints that may conflict with other packages in this project. Install it separately without its dependencies:

```bash
# 3. Install pyclesperanto-prototype without its dependencies
pip install pyclesperanto-prototype --no-deps
```

Afterwards, you should be able to issue the following commands (shown in an
interactive Python session):

```pycon
>>> import nanobind_example
>>> nanobind_example.add(1, 2)
3
```

CI Examples
-----------

The `.github/workflows` directory contains two continuous integration workflows
for GitHub Actions. The first one (`pip`) runs automatically after each commit
and ensures that packages can be built successfully and that tests pass.

The `wheels` workflow uses
[cibuildwheel](https://cibuildwheel.readthedocs.io/en/stable/) to automatically
produce binary wheels for a large variety of platforms. If a `pypi_password`
token is provided using GitHub Action's _secrets_ feature, this workflow can
even automatically upload packages on PyPI.


License
-------

_nanobind_ and this example repository are both provided under a BSD-style
license that can be found in the [LICENSE](./LICENSE) file. By using,
distributing, or contributing to this project, you agree to the terms and
conditions of this license.
