# FastAER Det

FastAER Det is a object detector optimized for detecting objects in aerial images on a Nvidia Jetson AGX running in real-time. This repository contains the code for training and testing the models aswell as links to the models itself.

Cite:
```
@article{Wolf_2021,
  title={FastAER Det: Fast Aerial Embedded Real-Time Detection},
  volume={13},
  ISSN={2072-4292},
  url={http://dx.doi.org/10.3390/rs13163088},
  DOI={10.3390/rs13163088},
  number={16},
  journal={Remote Sensing},
  publisher={MDPI AG},
  author={Wolf, Stefan and Sommer, Lars and Schumann, Arne},
  year={2021},
  month={Aug},
  pages={3088}}
```

The code is based on the following repositories:
- MMDetection for the detection framework
- The code for the oriented bounding boxes has been copied from [AerialDetection](https://github.com/dingjiansw101/AerialDetection/)
- The code for exporting the models to ONNX is from https://github.com/open-mmlab/mmdetection/pull/1386

Pretrained models can be downloaded from https://github.com/wolfstefan/fastaer-det/releases/tag/published

## Training Models

### Data Preperation

The iSAID images has to be converted and cropped with the iSAID DevKit. We use the defaualt settings (150 pixel overlap) for validation. This should be located in data/iSAID/iSAID_patches/. For the training we used the iSAID DevKit to create crops with 400 pixel overlap and placed them in data/iSAID/iSAID_patches_overlap_400/. Both folders should contain train/, test/ and val/ with the respective images and annotations in COCO format.

### Installing MMDetection

```bash
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
conda install pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.1 -c pytorch
pip install mmcv==0.3.1 future tensorboard
pip install -r requirements/build.txt
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
python setup.py develop
cd DOTA_devkit
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```

### Running Training

The models for 60 and 90 fps only differ in the evaluation settings and thus a single training is enough.

```bash
bash ./tools/dist_train.sh configs/fastaer_det/15_fps.py 1 --validate
bash ./tools/dist_train.sh configs/fastaer_det/30_fps.py 1 --validate
bash ./tools/dist_train.sh configs/fastaer_det/60_fps.py 1 --validate
bash ./tools/dist_train.sh configs/fastaer_det/90_fps.py 1 --validate
```

Since MMDetection does not support evalaution for OBBs, we run the OBB training without validate.

```bash
python tools/train.sh configs/fastaer_det/obb.py
```

## Testing Models

The code for inference is in the onnx-export branch.

### Installing MMDetection on Jetson with JetPack 4.3

```bash
# libjpeg-dev is needed for Pillow
# libfreetype6-dev is needed for matplotlib
sudo -E apt install virtualenv python3-dev libjpeg-dev libfreetype6-dev libblas-dev liblapack-dev gfortran libgeos-dev
virtualenv -p python3 mmdetection
ln -s /usr/lib/python3.6/dist-packages/cv2/python-3.6/cv2.cpython-36m-aarch64-linux-gnu.so mmdetection/lib/python3.6
ln -s /usr/lib/python3.6/dist-packages/tensorrt/ mmdetection/lib/python3.6/
source mmdetection/bin/activate
# nvidia pytorch builds: https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano-version-1-3-0-now-available/
wget https://nvidia.box.com/shared/static/phqe92v26cbhqjohwtvxorrwnmrnfx1o.whl -O torch-1.3.0-cp36-cp36m-linux_aarch64.whl
pip3 install numpy torch-1.3.0-cp36-cp36m-linux_aarch64.whl
pip install torchvision==0.4.1
git clone --single-branch --branch onnx-export https://github.com/wolfstefan/fastaer-det.git
cd mmdetection
pip install pyyaml six addict requests
# --no-deps since OpenCV is provided but can't be found by pip
pip install --no-deps mmcv==0.2.14
pip install Cython
pip install pycocotools
pip install pybind11==2.2.3
pip install matplotlib
python setup.py develop
pip install scikit-image
pip install --no-deps imgaug
pip install pycuda
```

### Build TensorRT ONNX Parser with NMS Plugin

#### Build CMake

JetPack 4.3 has CMake version which is too old.

```bash
wget https://github.com/Kitware/CMake/releases/download/v3.17.3/cmake-3.17.3.tar.gz
tar xfz cmake-3.17.3.tar.gz
cd cmake-3.17.3
mkdir build
cd build
cmake ..
make -j6
cd ../..
export PATH=`pwd`/cmake-3.17.3/build/bin:$PATH
```

#### Build TensorRT ONNX Parser

GPU\_ARCHS=72 is only valid for Jetson AGX. Adjust for the used architecture.

```bash
git submodule update --init --recursive
git clone --single-branch --branch release/6.0 https://github.com/NVIDIA/TensorRT.git
cd TensorRT
git submodule update --init --recursive
rm -r parsers/onnx/
cp -r ../onnx-tensorrt parsers/onnx
mkdir -p build && cd build
cmake .. -DTRT_BIN_DIR=out/ -DCMAKE_BUILD_TYPE=Release -DGPU_ARCHS="72"
make -j6 onnx_plugins onnx2trt
cp parsers/onnx/out/libonnx_plugins.so ../../
```

### Build TensorRT profiler

It is not necessary to build the TensorRT profiler to run an inference. In this case importing mmdet.trt.profiler has to be removed from tool/test_trt.py.

```bash
cd mmdet/trt/
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) profiler.cpp -o profiler$(python3-config --extension-suffix)
```

### Create TensorRT engine

We have reduced the max_per_img parameter to 200 in the config files to reduce the validation time during training. Thus, it should be set to 1000 before exporting the model to ONNX.

```bash
python ../tools/export.py --check --checkpoint <model_checkpoint.pth> <mmdet_config.py> model.onnx
python ../tools/get_onnx_nms_nodes.py model.onnx
```

The last script outputs the names of the NMS's output tensors. The first tensor describes the
detections and the second tensor contains the number of detections. We need these to cut the
model after the NMS. Afterwards, the model contain operations to slice the detections based on
the number of detections to the right size. However, TensorRT has problems with dynaimc shapes
and thus we cut directly after the NMS and slice the tensor on the CPU.

```bash
python tools/onnx_edit.py --outputs <first NMS output>[float32],<second NMS output>[int64] --skipverify model.onnx model_cut.onnx
TensorRT/build/parsers/onnx/out/onnx2trt model_cut.onnx -o engine.trt -b 1 -d 16
```

### Test model

```bash
python tools/test_trt.py <config.py> --model <engine.trt> --json_out <output.json>
```

The output .json file can be evaluated with the iSAID DevKit to obtain results. For evaluating OBB models, the adjusted iSAID DevKint included in this repository should be used.
