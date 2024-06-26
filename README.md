# HUP-3D: Rendering of 3D multi-view synthetic images for assisted-egocentric hand-ultrasound pose estimation

<img src="assets/images/camera_sphere_rotation.gif" width="300" height="300" alt="Description">

- [Project page](http://medicalaugmentedreality.org/handobject.html) <!-- - [Paper](http://arxiv.org/abs/2004.13449) -->
- [Synthetic Grasp Generation](https://github.com/BatFaceWayne/POV_Surgery)
- [Baseline repos](TBD)


This grasp renderer is based on the [Obman dataset generation pipeline](https://github.com/hassony2/obman_render).
The synthetic grasps needed for this renderer can be generated with the [Grasp Generator](https://github.com/jonashein/grasp_generator).

Our synthetic dataset is available on the [project page](http://medicalaugmentedreality.org/handobject.html).

## Table of Content

- [Setup](#setup)
- [Demo](#demo)
- [Render grasps](#render-grasps)
- [Citations](#citations)

## Setup

### Clone repository (download the source code)

```sh
git clone [https://github.com/jonashein/grasp_renderer.git](https://github.com/manuelbirlo/HUP-3D_renderer.git
cd HUP-3D_renderer
```

### Download and install prerequisites

#### Download [Blender 2.82a](https://download.blender.org/release/Blender2.82/):
```sh
wget https://download.blender.org/release/Blender2.82/blender-2.82a-linux64.tar.xz
tar -xvf blender-2.82a-linux64.tar.xz
```

Install dependencies using pip:
```sh
wget https://bootstrap.pypa.io/get-pip.py
blender-2.82a-linux64/2.82/python/bin/python3.7m get-pip.py 
blender-2.82a-linux64/2.82/python/bin/pip install -r requirements.txt
```

#### Download assets from [SURREAL (Synthetic hUmans foR REAL tasks)](https://www.di.ens.fr/willow/research/surreal/data/):

- Go to SURREAL [dataset request page](https://www.di.ens.fr/willow/research/surreal/data/)
- Create an account, and receive an email with a username (= your email address) and password for data download
- Download SURREAL data dependencies using the following commands

```sh
cd assets/SURREAL
sh download_smpl_data.sh ../ your_username your_password
cd ..
```

#### Download [SMPL (A Skinned Multi-Person Linear)](http://smpl.is.tue.mpg.de/) Model

- Go to [SMPL website](http://smpl.is.tue.mpg.de/)
- Create an account by clicking *Sign Up* and provide your information
- Download and unzip `SMPL for Python users` (click on 'Download version <latest_version> for Python 2.7 (female/male/neutral, 300 shape PCs)' if you want the neutral human gender model to be included or on 'Download version <latest_version> for Python 2.7 (female/male. 10 shape PCs)' if you just want to use the male and female models), copy the content of the `models` folder (the .pkl files) to `assets/models`.  Note that all code and data from this download falls under the [SMPL license](http://smpl.is.tue.mpg.de/license_body).

#### Download body+hand textures and grasp information

- Request data on the [ObMan webpage](https://www.di.ens.fr/willow/research/obman/data/). 
  You should receive a link that will allow you to download `bodywithands.zip`.
- Download texture zips
- Unzip texture zip

```sh
cd assets/textures
mv path/to/downloaded/bodywithands.zip .
unzip bodywithands.zip
cd ../../
```

- Your structure should look like this:

```
grasp_renderer/
  assets/
    models/
      basicModel_f_lbs_10_207_0_v1.1.0.pkl
      basicModel_m_lbs_10_207_0_v1.1.0.pkl
      ...
```

#### Download [MANO](http://mano.is.tue.mpg.de/) model

- Go to [MANO website](http://mano.is.tue.mpg.de/)
- Create an account by clicking *Sign Up* and provide your information
- Download Models and Code (the downloaded file should have the format mano_v*_*.zip). Note that all code and data from this download falls under the [MANO license](http://mano.is.tue.mpg.de/license).
- unzip the file mano_v*_*.zip: `unzip mano_v*_*.zip`
- set environment variable: `export MANO_LOCATION=/path/to/mano_v*_*`

#### Modify mano code to be Python3 compatible

- Remove `print 'FINITO'` at the end of file `webuser/smpl_handpca_wrapper.py` (line 144)

```diff
-    print 'FINITO'
```

- Replace `import cPickle as pickle` by `import pickle`

```diff
-    import cPickle as pickle
+    import pickle
```

  - at top of `webuser/smpl_handpca_wrapper.py` (line 23)
  - at top of `webuser/serialization.py` (line 30)
- Fix pickle encoding
  - in `webuser/smpl_handpca_wrapper.py` (line 74)

```diff
-    smpl_data = pickle.load(open(fname_or_dict))
+    smpl_data = pickle.load(open(fname_or_dict, 'rb'), encoding='latin1')
```

  - in `webuser/serialization.py` (line 90)

```diff
-    dd = pickle.load(open(fname_or_dict))
+    dd = pickle.load(open(fname_or_dict, 'rb'), encoding='latin1')
```

- Fix model paths in `webuser/smpl_handpca_wrapper.py` (line 81-84)

```diff
-    with open('/is/ps2/dtzionas/mano/models/MANO_LEFT.pkl', 'rb') as f:
-        hand_l = load(f)
-    with open('/is/ps2/dtzionas/mano/models/MANO_RIGHT.pkl', 'rb') as f:
-        hand_r = load(f)
+    with open('/path/to/mano_v*_*/models/MANO_LEFT.pkl', 'rb') as f:
+        hand_l = load(f, encoding='latin1')
+    with open('/path/to/mano_v*_*/models/MANO_RIGHT.pkl', 'rb') as f:
+        hand_r = load(f, encoding='latin1')
```

At the time of writing the instructions mano version is 1.2 so use 

```diff
-    with open('/is/ps2/dtzionas/mano/models/MANO_LEFT.pkl', 'rb') as f:
-        hand_l = load(f)
-    with open('/is/ps2/dtzionas/mano/models/MANO_RIGHT.pkl', 'rb') as f:
-        hand_r = load(f)
+    with open('/path/to/mano_v1_2/models/MANO_LEFT.pkl', 'rb') as f:
+        hand_l = load(f, encoding='latin1')
+    with open('/path/to/mano_v1_2/models/MANO_RIGHT.pkl', 'rb') as f:
+        hand_r = load(f, encoding='latin1')
```



## Demo
<!-- We provide [exemplary grasps](assets/grasps/drill_grasps.txt) for the 3D drill model used in our synthetic and real datasets. -->
COMING SOON.

The 3D drill model can be downloaded [here](https://drive.google.com/file/d/1j3V2CTVEVPzI3Ybh159dfLtRXaoTqa00/view?usp=sharing).

Our synthetic dataset is available on the [project page](http://medicalaugmentedreality.org/handobject.html).

## Render Grasps

<!-- To generate synthetic samples using the provided [exemplary grasps](assets/grasps/drill_grasps.txt) and [drill model](https://drive.google.com/file/d/1j3V2CTVEVPzI3Ybh159dfLtRXaoTqa00/view?usp=sharing), run the following command: -->
To create samples for custom 3D models, generated the required grasps with the [Grasp Generator](https://github.com/jonashein/grasp_generator) and adjust the paths in the arguments accordingly:
```
blender-2.82a-linux64/blender -noaudio -t 8 -P grasp_renderer.py -- '{"max_grasps_per_object": 300, "renderings_per_grasp": 50, "split": "train", "grasp_folder": "assets/grasps/", "backgrounds_path": "assets/backgrounds/", "results_root": "datasets/synthetic/"}'
```

## Citations

If you find this code useful for your research, please consider citing:

* the publication that this code was adapted for
```
@article{hein2021towards,
  title={Towards markerless surgical tool and hand pose estimation},
  author={Hein, Jonas and Seibold, Matthias and Bogo, Federica and Farshad, Mazda and Pollefeys, Marc and F{\"u}rnstahl, Philipp and Navab, Nassir},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  volume={16},
  number={5},
  pages={799--808},
  year={2021},
  publisher={Springer}
}
```

* the publication it builds upon and that this code was originally developed for
```
@inproceedings{hasson19_obman,
  title     = {Learning joint reconstruction of hands and manipulated objects},
  author    = {Hasson, Yana and Varol, G{\"u}l and Tzionas, Dimitris and Kalevatykh, Igor and Black, Michael J. and Laptev, Ivan and Schmid, Cordelia},
  booktitle = {CVPR},
  year      = {2019}
}
```
