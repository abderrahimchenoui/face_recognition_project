# face_recognition_project
   Our goal of this project is to realize the following pipline : face detection  ,recognition of the detected faces to match them with the company employees faces
   existing in the company database ,next come the deployment step ,optimizing the neuron networks ,which is  aimed at building a face recognition system on real time .

1  Choosing two models ,one for face detection and the second for face recognition :
  we test and compare the result of inference using different pretrained deep learning models , than we choose the best one in terms of accuracy and required time to infer one frame.

2  Deploy models :
  
  2-1 Deployment in the inference engine onnxruntime :
       First we must pull or load the onnxrutime image and generate a container with the suitable environment(the port ,the volume and the required package),then we     convert the two choosed model to onnx format to test them on the inference engine onnxruntime using the container ,and try to value the gain on execution time 
    and the losse on precision.   
  
  2-2 Deployment in the inference engine TensorRT :
      First we must pull or load the tensorRT image and generate a container with the suitable environment(the port ,the volume and the required package),then we       convert the two choosed model to TesnorRT format to test them on the inference engine onnxruntime using the container ,and try to value the gain on execution       time and the losse on precision.   
  
  2-3 Deployment using onnx with tensorRT :
      First thing to do is to convert the onnx format models to tensorRT format , then with follow the same deployment steps on tensorRT container ,then we value 
    the on execution time and the losse on precision.  
  
  2-4 Compare between the different deployment pipelines ,to choose the one with the shortest execution time without ignoring the precision preservating .
