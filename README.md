# learn_tvm

学习tvm

# 编译 tvm

## 获取源码

git clone --recursive https://github.com/apache/tvm tvm

## 安装 cmake

系统自带的版本过低，去 https://github.com/Kitware/CMake/releases 下载最新二进制

```
mkdir cmake_pkg
cd cmake_pkg
wget https://github.com/Kitware/CMake/releases/download/v3.31.8/cmake-3.31.8-linux-x86_64.tar.gz
tar -zxvf cmake-3.31.8-linux-x86_64.tar.gz
sudo mv cmake-3.31.8-linux-x86_64 /opt/cmake
sudo ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
cmake --version
```

## 安装 llvm15 和 polly15

apt search llvm

```
llvm-15/noble 1:15.0.7-14build3 amd64
  Modular compiler and toolchain technologies

llvm-15-dev/noble 1:15.0.7-14build3 amd64
  Modular compiler and toolchain technologies, libraries and headers

llvm-15-doc/noble 1:15.0.7-14build3 all
  Modular compiler and toolchain technologies, documentation

llvm-15-examples/noble 1:15.0.7-14build3 all
  Modular compiler and toolchain technologies, examples

llvm-15-linker-tools/noble 1:15.0.7-14build3 amd64
  Modular compiler and toolchain technologies - Plugins

llvm-15-runtime/noble 1:15.0.7-14build3 amd64
  Modular compiler and toolchain technologies, IR interpreter

llvm-15-tools/noble 1:15.0.7-14build3 amd64
  Modular compiler and toolchain technologies, tools
```

apt search polly

```
libpolly-15-dev/noble 1:15.0.7-14build3 amd64
  High-level loop and data-locality optimizer
llvm-15-linker-tools/noble 1:15.0.7-14build3 amd64
  Modular compiler and toolchain technologies - Plugins
```
sudo apt install -y llvm-15 llvm-15-dev llvm-15-linker-tools llvm-15-runtime llvm-15-tools libpolly-15-dev llvm-15-linker-tools

## 编译

按照 https://tvm.apache.org/docs/install/from_source.html 进行配置

```
cd tvm
rm -rf build && mkdir build && cd build
# Specify the build configuration via CMake options
cp ../cmake/config.cmake .
```

config.cmake 最后写入。注意其中的 `llvm-config` 需要改写为 `llvm-config-15`

```
set(USE_OPENCL_EXTN_QCOM OFF)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

set(USE_LLVM "llvm-config-15 --ignore-libllvm --link-static")

set(HIDE_PRIVATE_SYMBOLS ON)

set(USE_CUDA   OFF)

set(USE_METAL  OFF)

set(USE_VULKAN OFF)

set(USE_OPENCL OFF)

set(USE_CUBLAS OFF)

set(USE_CUDNN  OFF)

set(USE_CUTLASS OFF)
```

安装基本编译工具 `sudo apt install build-essential cmake`

编译 `cmake .. && cmake --build . --parallel $(nproc)`

缺少 zstd 报错 `sudo apt-get install libzstd-dev`

## 安装

先安装 python 3.11/3.12，创建虚拟环境

或者 pip 开发者模式安装，也可以去掉 -e 全量安装，这样容易跳转到 python 代码实现

```bash
export TVM_LIBRARY_PATH=/home/mdk/repo/learn_tvm/tvm/build
pip install -e /home/mdk/repo/learn_tvm/tvm/python
```

验证安装成功

```bash
python -c "import tvm; print(tvm.__file__)"
/home/mdk/repo/learn_tvm/tvm/python/tvm/__init__.py
python -c "import tvm; print(tvm.base._LIB)"
<CDLL '/home/mdk/repo/learn_tvm/tvm/build/libtvm.so', handle 2a24d10 at 0x7f5eef248140>
```
