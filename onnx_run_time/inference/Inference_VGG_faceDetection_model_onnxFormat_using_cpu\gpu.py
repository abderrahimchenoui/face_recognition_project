import onnxruntime as rt
import numpy as np

import cv2


#import and resize the frame to test 
img = cv2.imread('face.png')
im = cv2.resize(img, (224, 224))

im = np.expand_dims(im.astype(np.float32), axis=0)

im = np.einsum('bwhc->bcwh', im)



# Inference using onnx
session = rt.InferenceSession('vgg19-caffe2-9.onnx')
session.set_providers(['CUDAExecutionProvider'])

#to check if the inference is running using the cpu or the gpu
print('the is doing using the : ', session.get_providers())


#to calculate execution time of testing the Vgg19 on his onnx format
import time

start = time.time()
out = session.run(None, {'data_0': im})[0]

print('Execution time : ', time.time() - start)

