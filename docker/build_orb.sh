#!/bin/bash  

mkdir -p thirdParty && cd thirdParty
install_path=$(pwd)/install
mkdir -p ${install_path}

python_prefix=$(python -c "import sys; print(sys.prefix)")  
python_include=${python_prefix}/include/python3.10/
python_lib=${python_prefix}/lib/libpython3.10.so
python_exe=${python_prefix}/bin/python
python_env=${python_prefix}/lib/python3/dist-packages/
numpy_include=$(python -c "import numpy; print(numpy.get_include())")  

if [ ! -f $python_lib ]; then
    python_lib=${python_prefix}/lib/x86_64-linux-gnu/libpython3.10.so
    echo "Using $python_lib"
    if [ ! -f $python_lib ]; then
        echo "libpython3.10.so NOT FOUND!"
        return 1
    fi
fi

if [ ! -w $python_env ]; then
    python_env=$(python -m site --user-site)
fi

echo ${python_env}

# # build pangolin
git clone -b v0.6 https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${install_path}
make install -j

# Ubuntu OpenCV path
opencv_dir=/usr/lib/x86_64-linux-gnu/cmake/opencv4

# build orbslam2
cd ../../
cd ORB-SLAM2-PYBIND
bash build.sh ${opencv_dir} ${install_path}
cd ../


# build pybind
# build boost
wget -t 999 -c https://netactuate.dl.sourceforge.net/project/boost/boost/1.80.0/boost_1_80_0.tar.gz
tar xzf boost_1_80_0.tar.gz
cd boost_1_80_0
./bootstrap.sh --with-libraries=python --prefix=${install_path} --with-python=${python_exe}

# ./b2
./b2 install --with-python include=${python_include} --prefix=${install_path}


# # build orbslam_pybind
cd ../pybind
mkdir -p build && cd build

cmake .. -DPYTHON_INCLUDE_DIRS=${python_include} \
         -DPYTHON_LIBRARIES=${python_lib} \
         -DPYTHON_EXECUTABLE=${python_exe} \
         -DBoost_INCLUDE_DIRS=${install_path}/include/boost \
         -DBoost_LIBRARIES=${install_path}/lib/libboost_python310.so \
         -DORB_SLAM2_INCLUDE_DIR=${install_path}/include/ORB_SLAM2 \
         -DORB_SLAM2_LIBRARIES=${install_path}/lib/libORB_SLAM2.so \
         -DOpenCV_DIR=${install_path}/lib/cmake/opencv4 \
         -DPangolin_DIR=${install_path}/lib/cmake/Pangolin \
         -DPYTHON_NUMPY_INCLUDE_DIR=${numpy_include} \
         -DCMAKE_INSTALL_PREFIX=${python_env}

make install -j
