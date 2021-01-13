"""
tensorflow 모델 save and load 참조 여기서 함.
https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
"""

import tensorflow as tf
import numpy as np
import scipy.misc
from datetime import timedelta
import os
import csv
from glob import glob
import matplotlib.pyplot as plt

from Generator_Utils import *

# 학습에 필요한 설정값들을 지정
IMAGE_SHAPE_KITTI = (160, 576)

DATA_DIR = "data_road"

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it.
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed.
        tf.import_graph_def(graph_def, name="")
    return graph

def run():
    print("Start inference")

    # GPU
    tf.debugging.set_log_device_placement(True)
    gpu = tf.config.experimental.list_physical_devices('GPU')
    if gpu:
        try:
            tf.config.experimental.set_memory_growth(gpu[0], True)
        except RuntimeError as e:
            print(e)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # training 데이터와 validation 데이터 개수를 불러옴.
    training_labels_count = len(glob(os.path.join(DATA_DIR, 'training/gt_image_2/*_road_*.png')))
    training_images_count = len(glob(os.path.join(DATA_DIR, 'training/image_2/*.png')))
    training_projection_count = len(glob(os.path.join(DATA_DIR, 'training/projection/*.png')))
    testing_images_count = len(glob(os.path.join(DATA_DIR, 'testing/image_2/*.png')))
    validating_labels_count = len(glob(os.path.join(DATA_DIR, 'validating/gt_image_2/*_road_*.png')))
    validating_images_count = len(glob(os.path.join(DATA_DIR, 'validating/image_2/*.png')))
    validating_projection_count = len(glob(os.path.join(DATA_DIR, 'validating/projection/*.png')))

    assert not (training_images_count == training_labels_count == testing_images_count == 0), \
        'Kitti dataset not found. Extract Kitti dataset in {}'.format(DATA_DIR)
    assert training_images_count == 259, 'Expected 259 training images, found {} images.'.format(
        training_images_count)  # 289
    assert training_labels_count == 259, 'Expected 259 training labels, found {} labels.'.format(
        training_labels_count)  # 289
    assert training_projection_count == 259, 'Expected 259 training projection images, found {} images.'.format(
        training_projection_count)
    assert testing_images_count == 290, 'Expected 290 testing images, found {} images.'.format(
        testing_images_count)
    assert validating_labels_count == 30, 'Expected 30 validating images, found {} images.'.format(
        validating_images_count)
    assert validating_labels_count == 30, 'Expected 30 validating images, found {} labels.'.format(
        validating_labels_count)
    assert validating_projection_count == 30, 'Expected 30 validaing projection images, found {} images.'.format(
        validating_projection_count)

    # We use our "load_graph" function
    graph = load_graph("./frozen_model.pb")

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)
        # name/Placeholder/inputs_placeholder
        # ...
        # name/Accuracy/predictions

    # We access the input and output nodes
    input_lidar = graph.get_tensor_by_name('input_lidar:0')
    output_lid = graph.get_tensor_by_name('output:0')
    keep_probability = graph.get_tensor_by_name('keep_probability:0')

    # We launch a Session
    with tf.Session(graph=graph) as sess:
        # Note: we don't need to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants
        image_output = gen_test_output(sess, output_lid, keep_probability, input_lidar,
                                       os.path.join(DATA_DIR, 'validating'),
                                       IMAGE_SHAPE_KITTI)

        output_dir = os.path.join(DATA_DIR, 'generated_output_from_load_model')

        total_processing_time = 0
        for name, image, processing_time in image_output:
            # print(image)

            # plt.imshow(image)
            # plt.show()
            scipy.misc.imsave(os.path.join(output_dir, name), image)
            print(os.path.join(output_dir, name))
            total_processing_time += processing_time

        print("Average processing time is : ", total_processing_time / 30)


if __name__ == '__main__':
    run()