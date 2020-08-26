
## Build/install instructions

A standard build can be done by invoking in the 'build' directory the following 
```shell 
$> export IPOPT_PATH=/path/to/iptopbuild
$> cmake ..
$> make 
$> make test
$> make install
```

NOTE: requires that Ipopt was previously compiled as a static library

## CMake-ing examples

One can disable fault handling
```shell
$> export IPOPT_PATH=/home/petra1/work/projects/gocompet/Ipopt-gollnlp/build
$> rm -rf *; cmake -DGOLLNLP_FAULT_HANDLING=OFF .. && make -j
```

### Kron reduction

Certain HSL libraries are needed for factorizing complex symmetric Y bus matrix. Also, matrix clases used to perform the remaining ops needed by the reduction are from HiOp.

To build with support for kron reduction

```shell
$> export IPOPT_PATH=/home/petra1/work/projects/gocompet/Ipopt-gollnlp/build
$> rm -rf *; cmake -DGOLLNLP_WITH_KRON_REDUCTION=ON  -DHIOP_DIR=/Users/petra1/work/projects/hiop/_dist-DEBUG/ -DUMFPACK_DIR=/Users/petra1/work/installs/SuiteSparse-5.7.1 .. && make -j
```

To build with support for kron reduction and GPU support 
```shell
export IPOPT_PATH=/ccs/home/cpetra/work/projects/gocompet/Ipopt-gollnlp
rm -rf *
CC=/sw/summit/gcc/8.1.1/bin/gcc \
CXX=/sw/summit/gcc/8.1.1/bin/g++ \
cmake -DGOLLNLP_USE_GPU=ON \
      -DGOLLNLP_WITH_KRON_REDUCTION=ON \
      -DHIOP_DIR=/ccs/home/cpetra/work/projects/hiop/_dist-DEBUG \
      -DUMFPACK_DIR=/ccs/home/cpetra/work/installs/SuiteSparse-5.7.2 .. && \
make -j
```
This assumes that Cuda and Magma are available on the system. If they're not or one wants to use custom versions of such libraries, then manual specification of the locations of customized versions is possible, see `Find` cmake scripts in `cmake/` folder.



