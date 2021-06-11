# EMNIST detection with fine-tuned Mobilenet-SSD on Raspberry Pi 4

## Install Object Detection API with TensorFlow 2
[installation instructions](research/object_detection/g3doc/tf2.md#installation)

## EMNIST(letters) Object Detection Dataset
❗ A, B, C, D만 사용하였습니다.

![Image of Dataset Generate Example](dataset_generate_example.png)
- research/generate_data.py 실행

![Image of Dataset Example](dataset_example.jpg)
- research/visualize_dataset.py 실행

## TFRecord
- Run generate_tfrecord.py
```powershell
    python generate_tfrecord.py
```

## Training and Evaluation with TensorFlow 2
> [Training and evaluation guide (CPU, GPU, or TPU)](research/object_detection/g3doc/tf2_training_and_evaluation.md#Local)

- Training Command
```powershell
    python object_detection/model_main_tf2.py --pipeline_config_path="model_zoo/ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config" --model_dir="custom_models/ssd_mobilenet_v2_320x320_coco17_tpu-8" --alsologtostderr
```
- Evaluation Command
```powershell
    python object_detection/model_main_tf2.py --pipeline_config_path="model_zoo/ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config" --model_dir="custom_models/ssd_mobilenet_v2_320x320_coco17_tpu-8" --checkpoint_dir="custom_models\ssd_mobilenet_v2_320x320_coco17_tpu-8" --alsologtostderr
```
- Running Tensorboard
```powershell
    tensorboard --logdir="custom_models/ssd_mobilenet_v2_320x320_coco17_tpu-8"
```

## TFLite Convertor
- ckpt to pb
```powershell
    python object_detection/export_tflite_graph_tf2.py --pipeline_config_path "model_zoo/ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config" --trained_checkpoint_dir "custom_models\ssd_mobilenet_v2_320x320_coco17_tpu-8" --output_directory "custom_models/ssd_mobilenet_v2_320x320_coco17_tpu-8"
```
- pb to tflite
    - research/generate_tflite.py 실행

## Raspberry Pi 4
[Pi image installation instructions](https://github.com/Qengineering/TensorFlow_Lite_SSD_RPi_64-bits)

## Reference
- [Training and Evaluation with TensorFlow 2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_training_and_evaluation.md)
- [MNIST Object Detection dataset](https://github.com/hukkelas/MNIST-ObjectDetection)
- [2.1. Custom Dataset으로 TFRecord 파일 만들기](https://ballentain.tistory.com/48)
- [Install 64 bit OS on Raspberry Pi 4 + USB boot](https://qengineering.eu/install-raspberry-64-os.html)