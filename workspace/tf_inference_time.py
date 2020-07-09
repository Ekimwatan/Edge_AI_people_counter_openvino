import time
import tensorflow as tf
import numpy as np
#from PIL import Image
import cv2



model_dir='ssd_mobilenet_v2_coco_2018_03_29/saved_model'
model = tf.saved_model.load(model_dir)
model = model.signatures['serving_default']
image=cv2.imread('image.jpg')

#print(model.inputs.shape)


image = np.asarray(image)
input_tensor = tf.convert_to_tensor(image)
input_tensor = input_tensor[tf.newaxis,...]

# run do pre-processing

start_time = time.time()
# run inference
detection = model(input_tensor)

end_time = time.time()
inference_time = end_time - start_time

print(inference_time)