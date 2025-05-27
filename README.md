# learn_tvm
学习tvm

# 编译 tvm

## 获取源码

git clone --recursive https://github.com/apache/tvm tvm

## 安装 cmake

系统自带的版本过低，去 https://github.com/Kitware/CMake/releases 下载最新二进制

解压后链接 sudo ln -s /workspaces/learn_tvm/cmake3.31.7/cmake-3.31.7-linux-x86_64/bin/cmake /usr/local/bin/cmake

```
@madokast ➜ /workspaces/learn_tvm (main) $ cmake --version
cmake version 3.31.7

CMake suite maintained and supported by Kitware (kitware.com/cmake).
```

## 安装 llvm15（跳过）

```
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 15`
export CC=/usr/bin/clang-15
export CXX=/usr/bin/clang++-15
export LLVM_CONFIG=/usr/bin/llvm-config-15
```

这个安装没有 polly，需要从源码安装 llvm

## 编译 llvm

git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout -b release/20.x origin/release/20.x
mkdir build
cd build
cmake -G "Unix Makefiles" ../llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;lld;lldb;polly;mlir;openmp" \
  -DCMAKE_INSTALL_PREFIX=/usr/local/llem-20
make -j$(nproc)
make install

## 编译

按照 https://tvm.apache.org/docs/install/from_source.html 进行配置



