######################
#author:  libo3      #              
#date  :  20180131   #
######################

import sys
sys.path.insert(0, "./")
from tensorflow_serving_client.protos import predict_pb2, prediction_service_pb2
import cv2
from grpc.beta import implementations
import tensorflow as tf
from tensorflow.python.framework import dtypes
import time
import numpy as np
import fnmatch,os,math
import re
from sklearn import cross_validation
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import os
import sys

if __name__ == '__main__':

    #saved_model_dir = 'model_save/export_model'
    #saved_model_dir = 'model_save'
    #print ('Exporting trained model to'), saved_model_dir

    pred_x = cv2.imread("S132_002_00000004#6.png")
    pred3d_x = np.zeros((1 ,96 ,96 ,3))
    pred_x = cv2.resize(pred_x,(96,96),interpolation=cv2.INTER_CUBIC)
    pred3d_x[0] = pred_x
    pred3d_x = np.reshape(np.average(pred3d_x,axis = 3),(1,96,96,1))
    print ("#######hape#######",pred3d_x.shape)

    #记个时
    start_time = time.time()
    #建立连接
    channel = implementations.insecure_channel("211.159.164.43", 8000)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)


    # request.model_spec.name = 'mnist' 
    # request.model_spec.signature_name = 'predict_images' 


    request = predict_pb2.PredictRequest()
    print ("$$$$$$  request  $$$$$$",request)
    #这里由保存和运行时定义，第一个是运行时配置的模型名，第二个是保存时输入的方法名
    request.model_spec.name = "test_saved_model"
    request.model_spec.signature_name = "predict_image"
    #入参参照入参定义
    print (">>>>>>>>>>>>>>>>>>",request)
    keepprob = np.array([1.])
    request.inputs["input"].ParseFromString(tf.contrib.util.make_tensor_proto(pred3d_x, dtype=dtypes.float32, shape=[1, 96, 96,1]).SerializeToString())
    request.inputs["keep_prob"].ParseFromString(tf.contrib.util.make_tensor_proto(keepprob, dtype=dtypes.float32, shape=[1,]).SerializeToString())
    #第二个参数是最大等待时间，因为这里是block模式访问的
    print ("$$$$$$$ stub $$$$$",stub)
    response = stub.Predict(request, 10.0)
    print ("DDDDD response DDDDDD",response)
    results = {}
    for key in response.outputs:
        tensor_proto = response.outputs[key]
        nd_array = tf.contrib.util.make_ndarray(tensor_proto)
        results[key] = nd_array
    print("cost %ss to predict: " % (time.time() - start_time))
    print(results["output"])
