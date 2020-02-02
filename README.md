
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

### To build with support for kron reduction
```shell
$> export IPOPT_PATH=/home/petra1/work/projects/gocompet/Ipopt-gollnlp/build
$> rm -rf *; cmake -DGOLLNLP_FAULT_HANDLING=OFF -DGOLLNLP_WITH_KRON_REDUCTION=ON -DME57_DIR=/home/petra1/work/installs/hsl_me57-1.1.0 .. && make -j
```