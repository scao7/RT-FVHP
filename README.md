# RT-FVHP
Real-Time Free-viewport Holographic Patient APP

Supplementary video:

[![Video Title](https://img.youtube.com/vi/37sN7_GQOdo/hqdefault.jpg)](https://youtu.be/37sN7_GQOdo)

## :desktop_computer: Requirements
<!-- --- -->
NVIDIA GPUs are required for this project.

Linux server: The implementation of the server code is tested on: 

```
Distributor ID: Ubuntu
Description: Ubuntu 18.04.6 LTS
Release: 18.04
Codename: bionic
```
Unity client is tested on windows 11

- simple_romp https://github.com/Arthur151/ROMP, we modified the package and make it able to produce SMPL masks
- simple_knn https://github.com/camenduru/simple-knn
- diff-gaussian-rasterazation https://github.com/graphdeco-inria/diff-gaussian-rasterization 



## Download SMPL Models
Register and download SMPL models [here](https://smplify.is.tue.mpg.de/download.php). Put the downloaded models in the folder smpl_models. Only the neutral one is needed.

```
humanModel
-- assets
----SMPL_NEUTRAL.pkl
-- smpl
---- smpl_numpy.py
-- smpl-meta
---- faces.npy
---- J_regressor_body25.npy
---- parents.npy
---- SMPL_NEURAL.pkl
---- smpl_uv.obj
---- weights.npy
-- smplx 
```


## Environment setup 
```
conda create -n RTFVHP python=3.8

pip install Cython==3.0.10
pip install numpy==1.24.1
pip install opencv-python==4.9.0.80
pip install tqdm

pip install chumpy
# comment out the line in chumpy __init__.py : 
# from numpy import bool, int, float, complex, object, unicode, str, nan, inf
```
### install pytorch with compatable cuda, we used cuda 11.7 

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2


### install customized simple_romp
cd RTFVHP/submodules/ROMP/simple_romp
python setup.py install
<!-- bash build.sh -->


## Real patient youtube view links

https://www.youtube.com/watch?v=IV_IsstW-gA&t=5s
## TODO

