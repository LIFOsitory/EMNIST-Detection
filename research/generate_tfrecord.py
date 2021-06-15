import cv2, os, io
import numpy as np
import tensorflow.compat.v1 as tf
import PIL.Image
import hashlib

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from tqdm import tqdm

import contextlib2
from object_detection.dataset_tools import tf_record_creation_util

flags = tf.app.flags
flags.DEFINE_string('label_map_path', 'dataset/emnist_letters_detection/label_map.pbtxt', 'Path to label map proto')
FLAGS = flags.FLAGS

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_to_tf_data(img_path, label_path, label_map_dict):
    filename = img_path.split('/')[-1]
    # THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    # img_path = os.path.join(THIS_FOLDER, img_path)
    # image
    full_path = img_path
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
      encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
      raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width, height = image.size

    classes, classes_text = [], []
    xmins, ymins, xmaxs, ymaxs = [], [], [], []
    with tf.io.gfile.GFile(label_path) as f:
        lines = f.readlines()
        content = [line.split(',') for line in lines][1:]
        for class_idx, xmin, ymin, xmax, ymax in content:
          classes.append(int(class_idx))
          classes_text.append(label_map_dict[int(class_idx)].encode('utf8'))
          xmins.append(float(xmin) / width)
          ymins.append(float(ymin) / height)
          xmaxs.append(float(xmax) / width)
          ymaxs.append(float(ymax) / height)

    feature_dict = {    
        'image/height':             dataset_util.int64_feature(height),
        'image/width':              dataset_util.int64_feature(width),
        'image/filename':           dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id':          dataset_util.bytes_feature(filename.encode('utf8')),
        'image/key/sha256':         dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded':            dataset_util.bytes_feature(encoded_jpg),
        'image/format':             dataset_util.bytes_feature('jpg'.encode('utf8')),
        'image/object/bbox/xmin':   dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax':   dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin':   dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax':   dataset_util.float_list_feature(ymaxs),
        'image/object/class/text':  dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes)
        }
    tf_data = tf.train.Example(features=tf.train.Features(feature=feature_dict))
 
    return tf_data

def main(_):

  base_dir = "./dataset/emnist_letters_detection"
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
  label_map_dict = {value: key for key, value in label_map_dict.items()}
  num_shards = 10
  
  for dataset in ["train", "test"]:
    img_names = os.listdir(os.path.join(base_dir, dataset, "images"))
    label_names = os.listdir(os.path.join(base_dir, dataset, "labels"))
    img_paths = [os.path.join(base_dir, dataset, "images", name) for name in img_names]
    label_paths = [os.path.join(base_dir, dataset, "labels", name) for name in label_names]
    output_file_dir = os.path.join(base_dir, dataset, "tfrecord")
    tf.io.gfile.mkdir(output_file_dir)
    # sharding when dataset is train
    if dataset == "train":
      with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(tf_record_close_stack, os.path.join(output_file_dir, f"{dataset}.record"), num_shards)
        for index, (img_path, label_path) in tqdm(enumerate(zip(img_paths, label_paths)), desc="Generating tfrecord", total=len(img_paths)):
          tf_example = image_to_tf_data(img_path, label_path, label_map_dict)
          output_shard_index = index % num_shards
          output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
    else:
      tf_writer = tf.io.TFRecordWriter(os.path.join(output_file_dir, f"{dataset}.record"))
      for img_path, label_path in tqdm(zip(img_paths, label_paths), desc="Generating tfrecord", total=len(img_paths)):
          tf_example = image_to_tf_data(img_path, label_path, label_map_dict)
          tf_writer.write(tf_example.SerializeToString())
      tf_writer.close()

if __name__ == '__main__':
  tf.app.run()