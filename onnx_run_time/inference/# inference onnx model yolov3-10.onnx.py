#preprocessing

import numpy as np
from PIL import Image

# this function is from yolo3.utils.letterbox_image
def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data

  ######### inference #########
  import onnxruntime as rt
import numpy as np

import cv2


#import and resize the frame to test 
image = Image.open('face.png')
# input
image_data = preprocess(image)
image_size = np.array([image.size[1], image.size[0]], dtype=np.float32).reshape(1, 2)



# Inference using onnx
session = rt.InferenceSession('yolov3-10.onnx')
#session.set_providers(['CUDAExecutionProvider'])

#to check if the inference is running using the cpu or the gpu
print('the is doing using the : ', session.get_providers())


#to calculate execution time of testing the Vgg19 on his onnx format
import time

start = time.time()


input_1 = session.get_inputs()[0].name
image_shape = session.get_inputs()[1].name
    
out,b, c= session.run(None, {"input_1": image_data, "image_shape": image_size})

print('Execution time : ', time.time() - start)
print('Output on the model ', out.size)   
