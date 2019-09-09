
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
