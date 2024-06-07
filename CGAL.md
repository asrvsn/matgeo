```bash
brew install cgal
brew install tbb
git clone https://github.com/cgal/cgal-swig-bindings
cd cgal-swig-bindings
mkdir build
cd build
FOLDER=$(pip show numpy | grep Location | awk '{print $2}')
cmake -DCGAL_DIR=/usr/lib/CGAL -DBUILD_JAVA=OFF -DPYTHON_OUTDIR_PREFIX=$FOLDER ..
make -j 4   
```