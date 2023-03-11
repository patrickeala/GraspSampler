This is the installation of the grasp sampler to UBUNTU 18.04 with fresh python 3.8

### Prerequisits
Make sure that core packages are installed

```pip install numpy
pip install matplotlib
pip install trimesh
```

Install octomap:

```
git clone git://github.com/OctoMap/octomap.git
cd octomap/
mkdir build
cd build/
cmake ..
make
sudo make install
```


Install libccd:

```
git clone https://github.com/danfis/libccd
cd libccd/
cd src/
sudo apt install m4
m4 -DUSE_DOUBLE ccd/config.h.m4 >ccd/config.h
cd ..
mkdir build
cd build/
cmake -G "Unix Makefiles" ..
make
sudo make install
```


Install fcl using as following:

```
git clone https://github.com/flexible-collision-library/fcl
cd fcl
git checkout 7075caf # version 0.5.0
mkdir build
cd build
cmake ..
make -j4
sudo make install
```


Finally, install python-fcl:
```pip install python-fcl```

### Install

Just clone this repository, for now no need to install

```git clone https://github.com/patrickeala/grasper```