import tensorflow as tf

# Convert the model
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file="./ssd_mobilenet_fine_tune/tflite/tflite_graph.pb",
    input_arrays=['input'],
    input_shapes={'input' : [1, 300, 300,3]},
    output_arrays=['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3']
  )
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

# Save the model.
with open('detectv9.tflite', 'wb') as f:
  f.write(tflite_model)