#preprocessing

import numpy as np
from PIL import Image

def preprocess2(image):
    # Resize
    ratio = 800.0 / min(image.size[0], image.size[1])
    image = image.resize((int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR)

    # Convert to BGR
    image = np.array(image)[:, :, [2, 1, 0]].astype('float32')

    # HWC -> CHW
    image = np.transpose(image, [2, 0, 1])

    # Normalize
    mean_vec = np.array([102.9801, 115.9465, 122.7717])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]

    # Pad to be divisible of 32
    import math
    padded_h = int(math.ceil(image.shape[1] / 32) * 32)
    padded_w = int(math.ceil(image.shape[2] / 32) * 32)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, :image.shape[1], :image.shape[2]] = image
    image = padded_image

    return image


  
  ############# inference ###############
  
  #MaskRCNN-10.onnx
import onnxruntime as rt
import numpy as np

import cv2


#import and resize the frame to test 


image = Image.open('face.png')
# input
img = Image.open('face.png')
img_data = preprocess2(img)






# Inference using onnx
session = rt.InferenceSession('MaskRCNN-10.onnx')
#session.set_providers(['CUDAExecutionProvider'])

#to check if the inference is running using the cpu or the gpu
print('the is doing using the : ', session.get_providers())


#to calculate execution time of testing the Vgg19 on his onnx format
import time

start = time.time()
input_name = session.get_inputs()[0].name
out = session.run(None, {input_name: img_data})[0]

print('Execution time : ', time.time() - start)
print('Output on the model ', out.size)

