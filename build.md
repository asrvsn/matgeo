# 0. OS-specific prereqs
macOS:
```bash
brew install cmake pkg-config autoconf automake autoconf-archive
```

linux:


# 1. Install `vcpkg`
```bash
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg && ./bootstrap-vcpkg.sh
```

# 2. Update `conda` environment
```bash
mamba env update -n ENV_NAME -f environment.yml
```

# 3. Install `matgeo`
```bash
pip install --no-build-isolation -ve .
```
It's going to fail the first time. Run it again.
```bash
pip install --no-build-isolation -ve .
```

# 4. Dealing with `pyopencl` issues on

If you run into
```bash
pyopencl._cl.LogicError: clGetPlatformIDs failed: PLATFORM_NOT_FOUND_KHR
```
ensure `pyopencl` is installed and if on macOS run:
```bash
mamba install ocl_icd_wrapper_apple
```

# 5. (Optional) Clean and re-build

```bash
cd "$VCPKG_ROOT" && git pull && ./bootstrap-vcpkg.sh
```
Go back to the root of the project and run:
```bash
rm -rf build
pip install --no-build-isolation -ve .
```