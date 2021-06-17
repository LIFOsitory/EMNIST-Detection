# EMNIST detection with fine-tuned Mobilenet_v2-SSD on Raspberry Pi 4
[![TensorFlow 2.5.0](https://img.shields.io/badge/TensorFlow-2.5.0-FF6F00?logo=tensorflow&style=flat-square)](https://github.com/tensorflow/tensorflow/releases/tag/v2.5.0)
[![Python 3.8.8](https://img.shields.io/badge/Python-3.8.8-3776AB?logo=python&style=flat-square)](https://www.python.org/downloads/release/python-388/)
[![Anaconda 4.10.1](https://img.shields.io/badge/Anaconda-4.10.1-44A833?logo=anaconda&style=flat-square)](https://github.com/conda/conda/releases/tag/4.10.1)
[![Raspberry Pi 4](https://img.shields.io/badge/Raspberry%20Pi-4%20Model%20B-A22846?logo=Raspberry%20Pi&style=flat-square)](https://www.raspberrypi.org/products/raspberry-pi-4-model-b/)

> This repository is forked from [tensorflow/models](https://github.com/tensorflow/models) and modified by [LIFOsitory](https://github.com/LIFOsitory)

![overview](overview.png)

### Appended Directories and Files
- custom_models
- model_zoo
- dataset
- generate_dataset_old.py
- generate_dataset.py
- visualize_dataset.py
- generate_tfrecord.py
- generate_tflite.py
- infer_ckpt.py
- test_tflite.py
- detect.tflite
- detect_old.tflite

🧡 [research](https://github.com/LIFOsitory/EMNIST-Detection/tree/master/research) 안에서 작업하였습니다.

🧡 visualize_dataset.py을 제외하곤 레퍼런스를 바탕으로 전부 직접 수정하거나 생성하였습니다.

🧡 dataset 파일이 너무 큰 관계로 train, test 폴더는 제외하였습니다.
generate dataset.py를 통해 생성할 수 있습니다.

## Install CUDA and cuDNN on Windows
> [CUDA TOOLKIT DOCUMENTATION](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

> [NVIDIA CUDNN DOCUMENTATION](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)

## Install Object Detection API with TensorFlow 2
> [Object Detection API with TensorFlow 2](research/object_detection/g3doc/tf2.md#installation)

### Python Package Installation

```bash
cd models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow
pip install tensorflow-gpu
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install --use-feature=2020-resolver .
# Test the installation.
python object_detection/builders/model_builder_tf2_test.py
```

## Generate EMNIST letters Object Detection dataset
> [MNIST Object Detection dataset](https://github.com/hukkelas/MNIST-ObjectDetection)

![Image of Dataset Example](dataset_example.jpg)

Dataset for object detection on EMNIST letters with COCO 2017 Background. By default, the script generates a dataset with the following attributes:

- 10,000 images in train. 1,000 images in test
- 26 Classes(A ~ Z)
- Between 1 and 5 letters per image
- Gaussian Blur
- Threshold 200
    - If the pixel value of EMNIST exceeds the threshold, replace it with COCO image

### Generate dataset

![Image of Dataset Generate Example](dataset_generate_example.png)

```bash
    python generate_dataset.py --data_dir="d:/tensorflow_dataset"
``` 

❗ [Tensorflow Dataset](https://www.tensorflow.org/datasets/overview)을 통해 자동으로 다운로드 되도록 구성하였습니다.

❗ COCO 2017을 사용하므로 다운로드(25.20 GiB) 및 압축 해제 시간이 오래 걸립니다.(1~2시간)

### Visualize dataset

The dataset can be visualized with the following command:

```bash
    python visualize_dataset.py
``` 

## Generate the TFRecord file
> [Preparing Inputs](research/object_detection/g3doc/using_your_own_dataset.md)

![Image of TFRecord Generate Example](tfrecord_generate_example.png)

The dataset can be converted to TFRecord file with the following command: 

```bash
    python generate_tfrecord.py
```

## Train and Evaluate with TensorFlow 2
> [Training and evaluation guide (CPU, GPU, or TPU)](research/object_detection/g3doc/tf2_training_and_evaluation.md#Local)

### Training Command

A local training job can be run with the following command:

```bash
    python object_detection/model_main_tf2.py --pipeline_config_path="model_zoo/ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config" --model_dir="custom_models/ssd_mobilenet_v2_320x320_coco17_tpu-8" --alsologtostderr
```

💡 Traing Step: 50000

### Evaluation Command

A local evaluation job can be run with the following command:

```bash
    python object_detection/model_main_tf2.py --pipeline_config_path="model_zoo/ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config" --model_dir="custom_models/ssd_mobilenet_v2_320x320_coco17_tpu-8" --checkpoint_dir="custom_models\ssd_mobilenet_v2_320x320_coco17_tpu-8" --alsologtostderr
```

![Image of Prediction Example](dataset_pred.png)

### Running Tensorboard

Progress for training and eval jobs can be inspected using Tensorboard. If using the recommended directory structure, Tensorboard can be run using the following command:

```bash
    tensorboard --logdir="custom_models/ssd_mobilenet_v2_320x320_coco17_tpu-8"
```

![Image of Loss Example](loss_example.jpg)

### Run inference with checkpoint file
> [Run inference with models from the zoo](research/object_detection/colab_tutorials/inference_tf2_colab.ipynb)

```bash
    python infer_ckpt.py --pipeline_config_path="model_zoo/ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config" --checkpoint_dir="custom_models/ssd_mobilenet_v2_320x320_coco17_tpu-8/" --checkpoint_number="ckpt-52" --test_image_path="dataset/emnist_letters_detection/test/images/541.jpg" 
```

## Run TF2 Models on Raspberry Pi
> [Running TF2 Detection API Models on mobile](research/object_detection/g3doc/running_on_mobile_tf2.md#step-1-export-tflite-inference-graph)

### Export TFLite inference grpah

An intermediate SavedModel that can be used with the TFLite Converter via commandline or Python API can be generated with the following command: 

```bash
    python object_detection/export_tflite_graph_tf2.py --pipeline_config_path "model_zoo/ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config" --trained_checkpoint_dir "custom_models/ssd_mobilenet_v2_320x320_coco17_tpu-8" --output_directory "custom_models/ssd_mobilenet_v2_320x320_coco17_tpu-8"
```

### Convert to TFLite

The SavedModel can be converted to TFLite with the following command: 

```bash
    python generate_tflite.py
```

You can infer the TFLite file with the following command:

```bash
    python test_tflite.py
```

### Run TFLite Model on Raspberry Pi 4

#### C++

[Pi image installation instructions](https://github.com/Qengineering/TensorFlow_Lite_SSD_RPi_64-bits)

#### Python

[TensorFlow Lite Python object detection example with Pi Camera](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/raspberry_pi)

🚩 둘 다 속도는 빠르나 카메라 사용시 정확도가 현저히 떨어집니다. (TヘTo)

🚩 Dataset 생성 방식을 바꾸거나 다른 Dataset의 활용을 고려해봐야겠습니다.

## Legacy

파일이나 폴더명 뒤에 _old가 붙은 경우 다음의 조건에서 생성된 파일:

- fixed size 300 x 300 by resizing
- 4 Classes(A ~ D)
- Threshold 255

### Generate dataset

![Image of Dataset Legacy Example](dataset_example_old.jpg)

### Evaluation

![Image of Prediction Example](dataset_predict_old.png)

일반적인 글자 이미지에 대해서 학습하지 않아 구글 이미지에 대해서는 성능이 좋지 못함.
 
![A in google image](example_a_old.jpg)

## Reference
- [Training and Evaluation with TensorFlow 2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_training_and_evaluation.md)
- [2.1. Custom Dataset으로 TFRecord 파일 만들기](https://ballentain.tistory.com/48)
- [How to Create to a TFRecord File for Computer Vision and Object Detection](https://blog.roboflow.com/create-tfrecord/)
- [Install 64 bit OS on Raspberry Pi 4 + USB boot](https://qengineering.eu/install-raspberry-64-os.html)
- [TensorFlow Datasets, A collection of ready-to-use datasets](https://www.tensorflow.org/datasets/overview)
- Cohen, G., Afshar, S., Tapson, J., & Van Schaik, A. (2017, May). EMNIST: Extending MNIST to handwritten letters. In 2017 International Joint Conference on Neural Networks (IJCNN) (pp. 2921-2926). IEEE.
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4510-4520).
- Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C. Y., & Berg, A. C. (2016, October). Ssd: Single shot multibox detector. In European conference on computer vision (pp. 21-37). Springer, Cham.