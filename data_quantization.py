import tensorflow as tf
import tensorflow.keras.applications.inception_v3 as inception_v3
#import tensorflow.keras as keras
'''
import model_zoo.net.resnet_v2 as resnet_v2
import model_zoo.net.inception_v3 as inception_v3
import model_zoo.net.mobilenet_v1 as mobilenet_v1
import tensorflow.contrib.slim as slim
from tensorflow.python.util import deprecation
from tensorflow.python.ops.gen_array_ops import *
deprecation._PRINT_DEPRECATION_WARNINGS = False
'''
import timeit
import sys
sys.path.append("./")
import numpy as np
import time
import pandas as pd
import model_zoo.utils as utils
import cv2
import pickle
def manual_dequantization():
    pass
def manual_quantization_file():
    layer_name_list = ["input", "Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3",
                       "MaxPool_3a_3x3", "Conv2d_3b_1x1", "Conv2d_4a_3x3",
                       "MaxPool_5a_3x3", "Mixed_5b", "Mixed_5c",
                       "Mixed_5d", "Mixed_6a", "Mixed_6b", "Mixed_6c",
                       "Mixed_6d", "Mixed_6e", "Mixed_7a",
                       "Mixed_7b", "Mixed_7c"]
    model_name = "inception_v3"
    # 1. read the data
    time_dict = {}

    writer = pd.ExcelWriter("inception_quant_images_guitar_non_normal.xlsx")

    # print("原始input",input_data)
    time_cost = []

    with tf.device("/device:CPU:0"):
        # start_data = np.asarray(np.load("middle_data/" + model_name + "/input_guitar" + ".npy"), dtype=np.float32)[
        #             np.newaxis, :]
        # np.asarray(self.read_image("middle_data/airplay.jpg",data_format[1],data_format[2]), dtype=np.float32)[np.newaxis, :]
        # , "airplay.jpg", "cat.jpg", "chicken.jpg", "dog.jpg", "pig.jpg", "scorpion.jpg","snack.jpg"
        average_writer = pd.ExcelWriter("quantization/per_tensor/" + model_name + "/tf/" + model_name + "_quant_images_tf_gpu.xlsx")
        for file_name in ["guitar.jpg","cat.jpg", "scorpion.jpg","dog.jpg", "pig.jpg"]:
            # 1. record the quantization details of each partition layer under specific images.
            details_writer = pd.ExcelWriter("quantization/per_tensor/"+model_name+"/tf/"+model_name+"_"+file_name.split(".")[0]+"_quant_gpu_details.xlsx")
            layer_details = []
            for layer_name in layer_name_list[1:]:
                details = []
                input_data = np.asarray(np.load("middle_data/" + model_name + "/"+layer_name+"_"+file_name.split(".")[0] + ".npy"))
                # 1.1 test 200 times for each layer under specific images.
                for i in range(10):
                    a = time.time()
                    # min_value = np.min(input_data, axis=tuple(range(0, len(input_data.shape) - 1)))
                    min_value = np.min(input_data)
                    b = time.time()
                    # max_value = np.max(input_data, axis=tuple(range(0, len(input_data.shape) -
                    max_value = np.max(input_data)
                    c = time.time()
                    # result = tf.quantization.quantize(input_data, min_value, max_value, tf.quint8,
                    #                                  axis=len(input_data.shape) - 1)
                    result = tf.quantization.quantize(input_data, min_value, max_value, tf.quint8)
                    d = time.time()

                    details.append([len(pickle.dumps(input_data)),round(b-a,3),round(c-b,3),round(d-c,3),round(d-a,3)])

                    '''
                    min_value = np.tile(min_value, input_data.shape[0:len(input_data.shape) - 1] + (1,))
                    d = time.time()
                    max_value = np.tile(max_value, input_data.shape[0:len(input_data.shape) - 1] + (1,))
                    e = time.time()
                    range_value = np.subtract(max_value, min_value)
                    f = time.time()
                    range_value[range_value == 0] = 1
                    g = time.time()
                    result = (np.subtract(input_data, min_value) * 255 / range_value).astype(np.uint8)
                    h = time.time()
                    restore = (result * range_value / 255.0 + min_value).astype(np.float32)
                    l = time.time()
                    # np.save("quantization/inception_v3/"+layer_name+"_guitar_quan.npy",result)
                    # np.save("quantization/inception_v3/max_value/" + layer_name + "_guitar_quan_max.npy", max_value)
                    # np.save("quantization/inception_v3/min_value/" + layer_name + "_guitar_quan_min.npy", min_value)
                    details.append(
                        [len(pickle.dumps(input_data)), round(b - a, 3), round(c - b, 3), round(d - c, 3),
                         round(e - d, 3), round(f - e, 3), round(g - f, 3), round(h - g, 3),
                         round(h - a, 3), round(l - h + g - c, 3)])
                    '''
                # 1.2 record layer details as a sheet
                result_pd = pd.DataFrame(data=details, columns=["size", "min", "max", "cal", "total"]
                                         , index=range(200))
                result_pd.to_excel(excel_writer=details_writer, sheet_name=layer_name)
                # 1.3. get the average value
                print("details",file_name,layer_name)
                layer_details.append(np.around(np.average(np.array(details), axis=0), 3))
            details_writer.save()
            details_writer.close()
            '''
            result_pd = pd.DataFrame(data=layer_details,
                                     columns=["size", "min", "max", "min_tile", "max_tile",
                                              "max-min", "replace", "cal", "total", "restore"],
                                     index=layer_name_list[1:-1])
            result_pd.to_excel(excel_writer=writer, sheet_name=file_name.split(".")[0])
            '''
            result_pd = pd.DataFrame(data=layer_details,
                                     columns=["size", "min", "max", "cal", "total"],
                                     index=layer_name_list[1:])
            result_pd.to_excel(excel_writer=average_writer, sheet_name=file_name.split(".")[0])
        average_writer.save()
        average_writer.close()
def manual_quantization_tf22():
    @tf.function
    def qunat(value):
      #print("Tracing with", a)
      #value =  tf.function(a)
      a = time.time()
      max_range = tf.math.reduce_min(value)
      b = time.time()
      min_range = tf.math.reduce_min(value)
      c = time.time()
      result = tf.quantization.quantize(value,min_range,max_range,tf.quint8)
      d = time.time()
      return b-a,c-b,d-c,d-a

    layer_name_list = [ "Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3",
                       "MaxPool_3a_3x3", "Conv2d_3b_1x1", "Conv2d_4a_3x3",
                       "MaxPool_5a_3x3", "Mixed_5b", "Mixed_5c",
                       "Mixed_5d", "Mixed_6a", "Mixed_6b", "Mixed_6c",
                       "Mixed_6d", "Mixed_6e", "Mixed_7a",
                       "Mixed_7b", "Mixed_7c"]
    model_name = "inception_v3"
    result_pd = {}
    average_writer = pd.ExcelWriter("quantization/per_tensor/" + model_name + "/tf_22/" + model_name + "_quant_images_tf_gpu.xlsx")
    for file_name in ["guitar.jpg", "cat.jpg", "scorpion.jpg", "dog.jpg", "pig.jpg"]:
        details_writer = pd.ExcelWriter("quantization/per_tensor/" + model_name + "/tf_22/" + model_name + "_" + file_name.split(".")[0] + "_quant_gpu_details.xlsx")
        layer_details = []
        for layer_name in layer_name_list:
            data = np.asarray(np.load("middle_data/" + model_name + "/" + layer_name + "_" + file_name.split(".")[0] + ".npy"),dtype=np.float32)
            details = []
            for i in range(300):
                temp = qunat(data)
                details.append([temp[0].numpy(),temp[1].numpy(),temp[2].numpy(),temp[3].numpy()])
            result_pd = pd.DataFrame(data=details,columns=[ "min", "max", "cal", "total"]
                                     , index=range(300))
            result_pd.to_excel(excel_writer=details_writer, sheet_name=layer_name)
            print(file_name, layer_name)
            layer_details.append(np.around(np.average(utils.get_2_sigma(np.array(details),2), axis=0), 3))

        details_writer.save()
        details_writer.close()

        result_pd = pd.DataFrame(data=layer_details,
                                 columns=["min", "max", "cal", "total"],
                                 index=layer_name_list)
        result_pd.to_excel(excel_writer=average_writer, sheet_name=file_name.split(".")[0])
    average_writer.save()
    average_writer.close()
def manual_quantization_tf14():
    layer_name_list = ["input", "Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3",
                       "MaxPool_3a_3x3", "Conv2d_3b_1x1", "Conv2d_4a_3x3",
                       "MaxPool_5a_3x3", "Mixed_5b", "Mixed_5c",
                       "Mixed_5d", "Mixed_6a", "Mixed_6b", "Mixed_6c",
                       "Mixed_6d", "Mixed_6e", "Mixed_7a",
                       "Mixed_7b", "Mixed_7c"]
    model_name = "inception_v3"
    # 1. read the data
    time_dict = {}
    layer_details = []

    #print("原始input",input_data)
    time_cost = []

    with tf.device("/device:GPU:0"):
        # start_data = np.asarray(np.load("middle_data/" + model_name + "/input_guitar" + ".npy"), dtype=np.float32)[
        #             np.newaxis, :]
        #np.asarray(self.read_image("middle_data/airplay.jpg",data_format[1],data_format[2]), dtype=np.float32)[np.newaxis, :]
        #, "airplay.jpg", "cat.jpg", "chicken.jpg", "dog.jpg", "pig.jpg", "scorpion.jpg","snack.jpg"
        average_writer = pd.ExcelWriter("quantization/per_tensor/" + model_name + "/tf/" + model_name + "_quant_images_tf_gpu.xlsx")
        for file_name in ["cat.jpg", "scorpion.jpg","guitar.jpg","dog.jpg", "pig.jpg"]:#"guitar.jpg","dog.jpg", "pig.jpg", "scorpion.jpg"
            start_data = utils.read_image(file_name, 299, 299,False)[np.newaxis, :]
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False,device_count={"CPU": 1,"GPU":1})) as sess:
                with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
                    input_images = tf.placeholder(dtype=tf.float32, shape=[None,299,299,3], name='input')
                    # Note that endpoints does not include layer Input.
                    out, endpoints = inception_v3.inception_v3(inputs= input_images)
                    sess.run(tf.global_variables_initializer())
                    saver = tf.train.Saver()
                    saver.restore(sess, "model_zoo/weights/inception_v3.ckpt")
                    layer_details = []
                    details_writer = pd.ExcelWriter("quantization/per_tensor/"+model_name+"/tf/"+model_name+"_"+file_name.split(".")[0]+"_quant_gpu_details.xlsx")
                    for layer_name in layer_name_list[1:]:
                        details = []
                        for i in range(10):
                            input_data = sess.run(endpoints[layer_name],feed_dict={input_images:start_data})[0]
                            a = time.time()
                            # min_range = np.min(input_data,axis=tuple(range(0, len(input_data.shape) - 1)))
                            min_range = np.min(input_data[0])
                            b = time.time()
                            # max_range = np.max(input_data,axis=tuple(range(0, len(input_data.shape) - 1)))
                            max_range = np.max(input_data[0])
                            c = time.time()
                            #result = tf.quantization.quantize(input_data, min_range, max_range, tf.quint8,
                            #                                  axis=len(input_data.shape) - 1)
                            result = tf.quantization.quantize(input_data,min_range,max_range,tf.quint8)
                            d = time.time()
                            dequant_value = tf.quantization.dequantize(result.output,result.output_min,result.output_max)
                            f = time.time()
                            #print(dequant_value)
                            '''
                            for i in range(1):
                                e = time.time()
                                temp = dequant_value.eval(session=sess)#sess.run(dequant_value)
                                f = time.time()
                                print(round(f-e,3))
                            '''
                            #print(round(e-d,3),round(f-e,3))
                            time_cost = np.around(np.average(np.array(time_cost)),3)

                            details.append([len(pickle.dumps(input_data)),round(b-a,3),round(c-b,3),round(d-c,3),round(d-a,3),round(e-b,3),round(f-e,3)])

                            '''
                            min_value = np.tile(min_value,input_data.shape[0:len(input_data.shape) - 1]+(1,))
                            d = time.time()
                            max_value = np.tile(max_value, input_data.shape[0:len(input_data.shape) - 1] + (1,))
                            e = time.time()
                            range_value = np.subtract(max_value,min_value)
                            f = time.time()
                            range_value[range_value == 0] = 1
                            g = time.time()
                            result =(np.subtract(input_data,min_value)*255/range_value).astype(np.uint8)
                            h = time.time()
                            restore = (result*range_value/255.0+min_value).astype(np.float32)
                            l = time.time()
                            details.append([len(pickle.dumps(input_data)),round(b-a,3),round(c-b,3),round(d-c,3),round(e-d,3),round(f-e,3),round(g-f,3),round(h-g,3),round(h-a,3),round(l-h+g-c,3)])

                            '''
                            #np.save("quantization/inception_v3/"+layer_name+"_guitar_quan.npy",result)
                            #np.save("quantization/inception_v3/max_value/" + layer_name + "_guitar_quan_max.npy", max_value)
                            #np.save("quantization/inception_v3/min_value/" + layer_name + "_guitar_quan_min.npy", min_value)
                        #print(np.array(details).shape)
                        #details = np.array(details)
                        #details = utils.get_2_sigma(details,2)
                        '''
                        result_pd = pd.DataFrame(data = details,columns=["size","min","max","min_tile","max_tile",
                                                                           "max-min","replace","cal","total","restore"]
                                                 ,index = range(200))
                        '''
                        result_pd = pd.DataFrame(data = details,columns=["size","min","max","cal","total","restore_deq","restore_run"]
                                                 ,index = range(200))
                        result_pd.to_excel(excel_writer=details_writer,sheet_name=layer_name)
                        print(file_name,layer_name)
                        layer_details.append(np.around(np.average(details,axis=0),3))


                    details_writer.save()
                    details_writer.close()
                    '''
                    result_pd = pd.DataFrame(data = layer_details,columns=["size","min","max","min_tile","max_tile",
                                                                           "max-min","replace","cal","total","restore"],
                                             index=layer_name_list[1:])
                    '''
                    result_pd = pd.DataFrame(data = layer_details,columns=["size","min","max","cal","total","restore_deq","restore_run"],
                                             index=layer_name_list[1:])
                    result_pd.to_excel(excel_writer=average_writer,sheet_name=file_name.split(".")[0])

        average_writer.save()
        average_writer.close()
def get_quant_size():
    layer_name_list = ["input", "Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3",
                       "MaxPool_3a_3x3", "Conv2d_3b_1x1", "Conv2d_4a_3x3",
                       "MaxPool_5a_3x3", "Mixed_5b", "Mixed_5c",
                       "Mixed_5d", "Mixed_6a", "Mixed_6b", "Mixed_6c",
                       "Mixed_6d", "Mixed_6e", "Mixed_7a",
                       "Mixed_7b", "Mixed_7c"]
    for layer_name in layer_name_list:
        result = np.load("quantization/inception_v3/"+layer_name+"_guitar_quan.npy")
        pickle_result = pickle.dumps(result)
        print(layer_name,len(pickle_result),4*result.shape[0]*result.shape[1]*result.shape[2],4*result.shape[0]*result.shape[1]*result.shape[2]/len(pickle_result))
def anaylze_data_affect():
    data_formats = {'input': [299, 299, 3], 'Conv2d_1a_3x3': [149, 149, 32],
                    'Conv2d_2a_3x3': [147, 147, 32],
                    'Conv2d_2b_3x3': [147, 147, 64], 'MaxPool_3a_3x3': [73, 73, 64],
                    'Conv2d_3b_1x1': [73, 73, 80], 'Conv2d_4a_3x3': [71, 71, 192],
                    'MaxPool_5a_3x3': [5, 35, 192], 'Mixed_5b': [35, 35, 256],
                    'Mixed_5c': [35, 35, 288], 'Mixed_5d': [35, 35, 288], 'Mixed_6a': [17, 17, 768],
                    'Mixed_6b': [17, 17, 768], 'Mixed_6c': [17, 17, 768], 'Mixed_6d': [17, 17, 768],
                    'Mixed_6e': [17, 17, 768], 'Mixed_7a': [8, 8, 1280], 'Mixed_7b': [8, 8, 2048],
                    'Mixed_7c': [8, 8, 2048]}

    writer = pd.ExcelWriter("./test_random_int.xlsx")
    k = 0
    for data_format in data_formats.values():
        print(data_format)
        layer_details = []
        for i in range(10):
            input_data = np.random.randint(0,255,data_format)
            details = []
            for i in range(200):
                a = time.time()
                min_value = np.min(input_data, axis=tuple(range(0, len(input_data.shape) - 1)))
                b = time.time()
                max_value = np.max(input_data, axis=tuple(range(0, len(input_data.shape) - 1)))
                c = time.time()
                min_value = np.tile(min_value, input_data.shape[0:len(input_data.shape) - 1] + (1,))
                d = time.time()
                max_value = np.tile(max_value, input_data.shape[0:len(input_data.shape) - 1] + (1,))
                e = time.time()
                range_value = np.subtract(max_value, min_value)
                f = time.time()
                range_value[range_value == 0] = 1
                g = time.time()
                result = (np.subtract(input_data, min_value) * 255 / range_value).astype(np.uint8)
                h = time.time()
                restore = (result * range_value / 255.0 + min_value).astype(np.float32)
                l = time.time()
                # np.save("quantization/inception_v3/"+layer_name+"_guitar_quan.npy",result)
                # np.save("quantization/inception_v3/max_value/" + layer_name + "_guitar_quan_max.npy", max_value)
                # np.save("quantization/inception_v3/min_value/" + layer_name + "_guitar_quan_min.npy", min_value)
                details.append(
                    [len(pickle.dumps(input_data)), round(b - a, 3), round(c - b, 3), round(d - c, 3),
                     round(e - d, 3), round(f - e, 3), round(g - f, 3), round(h - g, 3),
                     round(h - a, 3), round(l - h + g - c, 3)])
            layer_details.append(np.around(np.average(details, axis=0), 3))
        result_pd = pd.DataFrame(data=layer_details,
                                 columns=["size", "min", "max", "min_tile", "max_tile",
                                          "max-min", "replace", "cal", "total", "restore"],
                                 index=range(10))
        result_pd.to_excel(excel_writer=writer, sheet_name=str(k))
        k = k+1
    writer.save()
    writer.close()
'''
#anaylze_data_affect()
@tf.function()
def fun_string(a):
    return a+a
fun = fun_string.get_concrete_function(tf.constant("a"))
for node in fun.graph.as_graph_def().node:
  print(f'{node.input} -> {node.name}')

tf.config.run_functions_eagerly(True)
image = "middle_data/guitar.jpg"
with open(image, 'rb') as f:
    data = f.read()
    print(len(data))
'''
# manual_quantization_tf22()
# manual_quantization_file()
# get_quant_size()

def measure_graph_size(f, *args):
  g = f.get_concrete_function(*args).graph
  print("{}({}) contains {} nodes in its graph".format(
      f.__name__, ', '.join(map(str, args)), len(g.as_graph_def().node)))
'''
@tf.function
def train(dataset):
  loss = tf.constant(0)
  print("=====================coming here")
  for x, y in dataset:
    loss += tf.abs(y - x) # Some dummy computation.
  return loss

small_data = [(1, 1)] * 3
big_data = [(1, 1)] * 10
measure_graph_size(train, small_data)
measure_graph_size(train, big_data)

measure_graph_size(train, tf.data.Dataset.from_generator(
    lambda: small_data, (tf.int32, tf.int32)))
measure_graph_size(train, tf.data.Dataset.from_generator(
    lambda: big_data, (tf.int32, tf.int32)))
'''
# Create an oveerride model to classify pictures
'''
class SequentialModel(tf.keras.Model):
  def __init__(self, **kwargs):
    super(SequentialModel, self).__init__(**kwargs)
    self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
    self.dense_1 = tf.keras.layers.Dense(128, activation="relu")
    self.dropout = tf.keras.layers.Dropout(0.2)
    self.dense_2 = tf.keras.layers.Dense(10)

  def call(self, x):
    x = self.flatten(x)
    x = self.dense_1(x)
    x = self.dropout(x)
    x = self.dense_2(x)
    return x

input = tf.keras.layers.Input(shape=(28, 28))
x = tf.keras.layers.Flatten(input_shape=(28, 28))(input)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(10)(x)
y = tf.keras.Model(input,x,name="test")
input_data = tf.random.uniform([60, 28, 28])
graph_model = tf.function(y)
# 运行结果
print("Eager time:", timeit.timeit(lambda: y(input_data), number=10000))
print("Graph time:", timeit.timeit(lambda: graph_model(input_data), number=10000))
'''
from vis.utils import utils
import numpy as np
inception = inception_v3.InceptionV3()
layer_name_list =["input_1","conv2d","conv2d_1","conv2d_2","max_pooling2d","conv2d_3","conv2d_4","max_pooling2d_1"
                 ,"mixed0","mixed1","mixed2","mixed3","mixed4","mixed5","mixed6","mixed7","mixed8","mixed9","mixed10","predictions"]
layer_name = "mixed0"

layer_idx = utils.find_layer_idx(inception, layer_name)
i = 40
target_layer = None
for layer in inception.layers:
    if i==40:
        target_layer = layer

input = tf.constant(np.asarray(np.expand_dims(np.random.rand(299,299,3,), axis=0),dtype=np.float32))
# print(input.dtype,inception.input.dtype)

partition_model = tf.keras.Model(inputs=tf.keras.Input(target_layer.input.shape),outputs=inception.output,name="test")
#result = partition_model(input)
#print(np.argmax(result[0]))
