import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("custom_models/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model") # path to the SavedModel directory
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

# Save the model.
with open('detect.tflite', 'wb') as f:
  f.write(tflite_model)