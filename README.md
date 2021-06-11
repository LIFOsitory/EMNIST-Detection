# Charcter detection with fine-tuned Mobilenet-SSD on Raspberry Pi 4

## Install Object Detection API with TensorFlow 2
[installation instructions](research/object_detection/g3doc/tf2.md#installation)

## EMNIST(letters) Object Detection Dataset
❗ A, B, C, D만 사용하였습니다.

![Image of Dataset Example](dataset_example.jpg)

![Image of Dataset Generate Example](dataset_generate_example.png)

- research/generate_data.py 실행

## TFRecord
- research/generate_tfrecord.py 실행

## Reference
- [Training and Evaluation with TensorFlow 2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_training_and_evaluation.md)
- [MNIST Object Detection dataset](https://github.com/hukkelas/MNIST-ObjectDetection)
- [2.1. Custom Dataset으로 TFRecord 파일 만들기](https://ballentain.tistory.com/48)
- [Install 64 bit OS on Raspberry Pi 4 + USB boot](https://qengineering.eu/install-raspberry-64-os.html)