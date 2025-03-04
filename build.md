# 0. OS-specific prereqs
macOS:
```bash
brew install cmake
brew install pkg-config
brew install autoconf
brew install automake
brew install autoconf-archive
```

linux:


# 1. Install `vcpkg`
```bash
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg && ./bootstrap-vcpkg.sh
```

# 2. Install `matgeo`
```bash
pip install nanobind scikit-build-core
pip install --no-build-isolation -ve .
```
It's going to fail the first time. Run it again.
```bash
pip install --no-build-isolation -ve .
```