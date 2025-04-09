FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS base

ARG MAX_JOBS 32
ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCHITECTURES=89;86
ENV TORCH_CUDA_ARCH_LIST="8.9;8.6" TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
ENV TZ=Asia/Shanghai LANG=C.UTF-8 LC_ALL=C.UTF-8 PIP_NO_CACHE_DIR=1 PIP_CACHE_DIR=/tmp/

RUN sed -i "s/archive.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list &&\
    sed -i "s/security.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list &&\
    rm -f /etc/apt/sources.list.d/* &&\
    rm -rf /opt/hpcx/ &&\
    apt-get update && apt-get upgrade -y &&\
    apt-get install -y --no-install-recommends \
        # Determined requirements and common tools / libs
        autoconf automake autotools-dev build-essential ca-certificates gnupg2 \
        make cmake yasm pkg-config gdb gcovr binutils-dev libiberty-dev \
        ninja-build ccache cppcheck doxygen graphviz plantuml \
        cimg-dev \
        clang-format \
        dh-autoreconf \
        libatlas-base-dev \
        libboost-all-dev \
        liblog4cplus-dev \
        libssh2-1-dev \
        libreadline-dev \
        libcereal-dev \
        libcgal-dev \
        libblas-dev libcxsparse3 libceres-dev libeigen3-dev libfftw3-dev liblapack-dev liblapacke-dev libnlopt-dev \
        libflann-dev metis libmetis-dev libopenblas-dev libsuitesparse-dev \
        libgtk2.0-dev libglew-dev libglm-dev libglu1-mesa-dev mesa-utils \
        freeglut3-dev libassimp-dev libglfw3-dev libproj-dev libqt5opengl5-dev \
        libxrandr-dev libxxf86vm-dev \
        libgflags-dev libgoogle-glog-dev libgtest-dev \
        libfreeimage-dev libjpeg-dev libpng-dev libtiff-dev libtiff5-dev \
        libavcodec-dev libavformat-dev libavutil-dev libavdevice-dev libv4l-dev v4l-utils \
        libpcap-dev libpostproc-dev libpq-dev libswscale-dev libswresample-dev libxine2-dev \
        intel-mkl libopenmpi-dev libtbb2 libtbb-dev  \
        daemontools krb5-user libkrb5-dev \
        ibverbs-providers libibverbs1 libnuma1 libnuma-dev libpmi2-0-dev \
        libhdf5-dev protobuf-compiler libprotobuf-dev librdmacm1 libsqlite3-dev libssl-dev libtool \
        libyaml-dev libyaml-cpp-dev libicu-dev libsimpleini-dev \
#        libopencv-dev libopenexr-dev \
        ffmpeg zlib1g-dev \
        ## Useful tools
        git curl wget unzip xz-utils nano vim-tiny \
        net-tools sudo htop iotop iputils-ping \
        cloc locate rsync screen tmux \
        daemontools nfs-common rsync proxychains \
        openssh-server openssh-client \
        software-properties-common \
        # Python
        python-is-python3 python3-dev python3.10-dev python3-pip python3-numpy \
        # PCL
#        libpcl-dev \
        # VTK
#        libvtk9-dev \
        # OSMesa build dependencies
        libosmesa6-dev \
        # EGL build dependencies
        libopengl-dev \
        libglvnd-dev \
        libgl-dev \
        libglx-dev \
        libegl-dev \
        # X11 utils
        mesa-utils \
        x11-apps \
        xorg-dev \
        # QT
        qtbase5-dev \
        # 
        libjsoncpp-dev \
    && apt-get purge unattended-upgrades \
    && rm /etc/ssh/ssh_host_ecdsa_key \
    && rm /etc/ssh/ssh_host_ed25519_key \
    && rm /etc/ssh/ssh_host_rsa_key \
    && cp /etc/ssh/sshd_config /etc/ssh/sshd_config_bak \
    && sed -i "s/^.*X11Forwarding.*$/X11Forwarding yes/" /etc/ssh/sshd_config \
    && sed -i "s/^.*X11UseLocalhost.*$/X11UseLocalhost no/" /etc/ssh/sshd_config \
    && grep "^X11UseLocalhost" /etc/ssh/sshd_config || echo "X11UseLocalhost no" >> /etc/ssh/sshd_config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

###############################################################
WORKDIR /tmp

# Install Determined AI and python deps
ENV PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONHASHSEED=0
ENV JUPYTER_CONFIG_DIR=/run/determined/jupyter/config
ENV JUPYTER_DATA_DIR=/run/determined/jupyter/data
ENV JUPYTER_RUNTIME_DIR=/run/determined/jupyter/runtime
RUN git clone https://github.com/LingzheZhao/determinedai-container-scripts &&\
    cd determinedai-container-scripts &&\
    git checkout v0.2.3 &&\
    pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple &&\
    pip install -U pip setuptools pathtools promise pybind11 &&\
    pip install determined && pip uninstall -y determined &&\
    pip install -r notebook-requirements.txt &&\
    ./add_det_nobody_user.sh &&\
    ./install_libnss_determined.sh &&\
    rm -rf /tmp/*

# gcc 11 for VTK9, >11 is not supported
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y
RUN apt-get install -y gcc-11 g++-11
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
   &&  update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-Linux-x86_64.sh -O cmake-3.31.6.sh \
   &&  chmod +x ./cmake-3.31.6.sh \
   &&  ./cmake-3.31.6.sh --skip-license --prefix=/usr/local \
   && rm -rf /tmp/*

# Downgrade pcl and vtk for GS_ICP_SLAM(#2)
ARG VTK_VERSION=9.1
ARG VTK_SUB_VERSION=9.1.0
RUN apt-get --purge remove -y "libpcl*" libvtk9-dev &&\
    apt-get update && apt-get install -y libx11-dev && apt-get clean &&\
    wget https://www.vtk.org/files/release/${VTK_VERSION}/VTK-${VTK_SUB_VERSION}.tar.gz &&\
    tar -xvzf VTK-${VTK_SUB_VERSION}.tar.gz &&\
    cd VTK-${VTK_SUB_VERSION} &&\
    cmake -B build -GNinja \
        -DVTK_USE_CUDA=ON \
        -DVTK_OPENGL_HAS_EGL:BOOL=ON \
        -DVTK_USE_X:BOOL=ON \
        -DVTK_DEFAULT_RENDER_WINDOW_HEADLESS:BOOL=ON \
        -DVTK_QT_VERSION:STRING=5 \
        -DQT_QMAKE_EXECUTABLE:PATH=`which qmake` \
        -DVTK_Group_Qt:BOOL=ON \
        -DCMAKE_PREFIX_PATH:PATH=`which cmake` \
        -DBUILD_SHARED_LIBS:BOOL=ON \
    &&\
    cmake --build build -t install &&\
    ldconfig &&\
    rm -rf /tmp
RUN git clone https://github.com/PointCloudLibrary/pcl.git &&\
    cd pcl &&\
    git checkout pcl-1.14.1 &&\
    cmake -B build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DVTK_DIR=/usr/local/include/vtk-${VTK_VERSION} \
    &&\
    cmake --build build -t install &&\
    ldconfig &&\
    rm -rf /tmp/*

# Install OpenEXR for OpenCV
ARG OPENEXR_VERSION=3.2.1
RUN wget https://github.com/AcademySoftwareFoundation/openexr/archive/refs/tags/v${OPENEXR_VERSION}.tar.gz &&\
    tar -xvzf v${OPENEXR_VERSION}.tar.gz &&\
    cd openexr-${OPENEXR_VERSION} &&\
    cmake -B build -GNinja &&\
    cmake --build build -t install &&\
    ldconfig && rm -rf /tmp/*

# Install OpenCV from source.
ARG OPENCV_VERSION="4.11.0"
RUN wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip &&\
    unzip $OPENCV_VERSION.zip &&\
    rm $OPENCV_VERSION.zip &&\
    wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip &&\
    unzip ${OPENCV_VERSION}.zip &&\
    cd opencv-${OPENCV_VERSION} &&\
    cmake \
        -B /tmp/opencv-${OPENCV_VERSION}/build \
        -GNinja \
        -DCMAKE_CXX_STANDARD=17 \
        -DOPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib-${OPENCV_VERSION}/modules \
        -DOPENCV_ENABLE_NONFREE=ON \
        -DWITH_TBB=ON \
        -DWITH_QT=ON \
        -DWITH_OPENGL=ON \
        -DWITH_VTK=ON \
        -DWITH_CUDA=ON \
        -DCUDA_ARCH_BIN=${TORCH_CUDA_ARCH_LIST} \
        -DHAVE_FFMPEG=ON \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_INSTALL_PREFIX=/usr/local &&\
    cmake --build build -t install &&\
    ldconfig && rm -rf /tmp/*

# libtorch
RUN cd / \
   &&  wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.2%2Bcu118.zip -O libtorch-cu118.zip \
   &&  unzip libtorch-cu118.zip \
   &&  rm libtorch-cu118.zip
ENV Torch_DIR=/libtorch/share/cmake/Torch

# Install pre-built pytorch and torchvision
RUN pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 &&\
    rm -rf /tmp/*

###############################################################
WORKDIR /opt

# Photo-SLAM
RUN git clone https://github.com/HuajianUP/Photo-SLAM.git /opt/Photo-SLAM \
   &&  cd /opt/Photo-SLAM \
   &&  cd ./ORB-SLAM3/Thirdparty/DBoW2 \
   &&  mkdir build \
   &&  cd build \
   &&  cmake .. -DCMAKE_BUILD_TYPE=Release \
   &&  make -j \
   &&  cp ../lib/*.so /usr/local/lib \
   &&  cd ../../g2o \
   &&  mkdir build \
   &&  cd build \
   &&  cmake .. -DCMAKE_BUILD_TYPE=Release \
   &&  make -j \
   &&  cp ../lib/*.so /usr/local/lib

# RTG-SLAM
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/RTG-SLAM/thirdParty/install/lib
ENV PATH=$PATH:/opt/RTG-SLAM/thirdParty/install/bin
RUN apt-get update && apt-get remove -y python3-blinker && apt-get clean &&\
    pip install -U "numpy<2.0" scipy plyfile open3d fvcore scikit-image \
                    tensorboard GPUtil torchmetrics trimesh pytorch_msssim rerun \
                    evo spatialmath-python \
                    icecream \
    &&\
    pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt211/download.html
RUN git clone https://github.com/LingzheZhao/RTG-SLAM \
   && cd RTG-SLAM \
   && git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --recursive \
   && chmod u+x build_orb.sh \
   && ./build_orb.sh

RUN cd /opt/RTG-SLAM/submodules/ &&\
    pip install ./simple-knn ./diff-gaussian-rasterizer-depth  ./cuda_utils

WORKDIR /tmp
