import cv2 as cv
import tensorflow as tf
import numpy as np
import scipy.misc
from datetime import timedelta
import os
import csv
import time
from glob import glob
import matplotlib.pyplot as plt

from Generator_Utils import *

# 학습에 필요한 설정값들을 지정
KEEP_PROB = 0.2
MAX_ITERATION = 1e-2
NUM_OF_CLASSESS_IMG = 2
NUM_OF_CLASSESS_LID = 1
# IMAGE_SHAPE_KITTI = (128, 480)
IMAGE_SHAPE_KITTI = (160, 576)
# IMAGE_SHAPE_KITTI = (192, 704)
# IMAGE_SHAPE_KITTI = (384, 1280)
# IMAGE_SHAPE_KITTI = (713, 1280)
BATCH_SIZE = 2
EPOCHS = 2101
LEARNING_RATE = 5e-4

DATA_DIR = "data_road"


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False):
    """ SepConv with BN between depthwise & pointwise. Optinally add activation after BN
        Implements right "SAME" padding for even kernel sizes

    :param x: input tensor
    :param filters: num of filters in pointwise convolution
    :param prefix: prefix before name
    :param stride: stride at depthwise conv
    :param kernel_size: kernel size for depthwise convolution
    :param rate: atrous rate for depthwise convolution
    :param depth_activation: flag to use activation between depthwise & pointwise convs
    :param epsilon: epsilon to use in BN layer
    :return:
    """
    if stride == 1:
        depth_padding = 'SAME'
    else:
        x = Zero_Padding(x, paddings=1, name=prefix+'zero_paddings')
        depth_padding = 'VALID'

    if not depth_activation:
        x = ReLU(x)

    # Atrous separable convolution contains atruos depthwise & pointwise convolution
    filter_shape = [kernel_size, kernel_size, x.get_shape()[-1], 1]
    filter = tf.get_variable(prefix+'_filter', shape=filter_shape, dtype=tf.float32)

    x = tf.nn.depthwise_conv2d(x, filter=filter, strides=[1, stride, stride, 1], rate=[rate, rate], padding=depth_padding, name=prefix+'_depthwise')

    x = Batch_Normalization(x)

    if depth_activation:
        x = ReLU(x)

    x = Conv2D_Block(x, filters, filter_height=1, filter_width=1, stride=1, padding='SAME', batch_normalization=True,
                     relu=True, name=prefix+"_pointwise_BN")

    return x

def Xception_block(x, depth_list, prefix, skip_connection_type, stride, rate=1, depth_activation=False, return_skip=False):
    """ Basic building block of modified Xception network
    :param x: input tensor
    :param depth_list: number of filters in each SepConv layer. len(depth_list) == 3
    :param prefix: prefix before name
    :param skip_connection_type: one of {'conv', 'sum', 'none'}
    :param stride: stride at last depthwise conv
    :param rate: atrous rate for depthwise convolution
    :param depth_activation: flag to use activation between depthwise & pointwise convs
    :param return_skip: flag to return additional tensor after 2 SepConvs for decoder
    :return:
    """
    residual = x
    outputs = x
    skip = x
    for i in range(3):
        residual = SepConv_BN(residual, depth_list[i], prefix+'_separable_conv{}'.format(i+1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = Conv2D_Block(x, depth_list[-1], filter_height=1, filter_width=1, stride=stride, padding='SAME',
                                batch_normalization=True, name=prefix+"_shortcut")
        outputs = tf.add(shortcut, residual, name=prefix+'_add_conv')
    elif skip_connection_type == 'sum':
        outputs = tf.add(residual, x, name=prefix+'_add_sum')
    elif skip_connection_type == 'none':
        outputs = residual

    if return_skip:
        return outputs, skip
    else:
        return outputs

def LiDAR2IMAGE_Generator(x_lid, keep_prob, num_classess_lid):
    """
        Instantiates the DeepLabV3+ architecture.
    :param x:
    :param input_shape:
    :param num_classess:
    :param out_stride:
    :param alpha:
    :param activation:
    :return:
    """
    entry_block3_stride = 2
    middle_block_rate = 1
    exit_block_rates = (1, 2)
    atrous_rate = (6, 12, 18)

    # Entry flow for LiDAR image
    entry_conv1_lid = Conv2D_Block(x_lid, 32, filter_height=3, filter_width=3, stride=2, batch_normalization=True,
                               relu=True, name='entry_flow_conv1_1_BN_lid')

    entry_conv1_2_lid = Conv2D_Block(entry_conv1_lid, 64, filter_height=3, filter_width=3, stride=1,
                                 batch_normalization=True, relu=True, name='entry_flow_conv1_2_BN_lid')

    xblock_entry_1_lid = Xception_block(entry_conv1_2_lid, [128, 128, 128], 'entry_flow_block1_lid',
                                    skip_connection_type='conv', stride=2, depth_activation=False)

    xblock_entry_2_lid, skip1_lid = Xception_block(xblock_entry_1_lid, [256, 256, 256], 'entry_flow_block2_lid',
                                           skip_connection_type='conv', stride=2, depth_activation=False,
                                           return_skip=True)

    xblock_entry_3_lid = Xception_block(xblock_entry_2_lid, [728, 728, 728], 'entry_flow_block3',
                                    skip_connection_type='conv', stride=entry_block3_stride, depth_activation=False)

    # Middle flow for LiDAR image
    for i in range(16):
        xblock_entry_3_lid = Xception_block(xblock_entry_3_lid, [728, 728, 728], 'middle_flow_unit_{}_lid'.format(i + 1),
                                        skip_connection_type='sum', stride=1, rate=middle_block_rate,
                                        depth_activation=False)

    # Exit flow for LiDAR image
    xblock_exit_1_lid = Xception_block(xblock_entry_3_lid, [728, 728, 1024], 'exit_flow_block1_lid',
                                   skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                                   depth_activation=False)

    extracted_feature_lid = Xception_block(xblock_exit_1_lid, [1536, 1536, 2048], 'exit_flow_block2_lid',
                                       skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                                       depth_activation=True)

    # end of feature extractor

    # branching for Atrous Spatia Pyramid Pooling
    # Image Featrue branch
    b4_lid = Global_Avg_Pool(extracted_feature_lid, name='b4_lid')

    # from (b_size, channels)->(b_size, 1, 1, channels)
    # b4 = Lambda(tf.expand_dims(x, dim=1))(b4)
    # b4 = Lambda(tf.expand_dims(x, dim=1))(b4)
    b4_lid = tf.expand_dims(b4_lid, dim=1)
    b4_lid = tf.expand_dims(b4_lid, dim=1)

    b4_lid = Conv2D_Block(b4_lid, 256, filter_height=1, filter_width=1, stride=1, padding='SAME',
                          batch_normalization=True, relu=True, name='image_pooling_lid')

    # Upsampling. have to use compat because of the option align_corners
    size_before = tf.shape(extracted_feature_lid)
    b4_lid = Resize_Bilinear(b4_lid, size_before[1:3], name='upsampling_after_global_avg_pooling_lid')

    # simple 1x1
    b0_lid = Conv2D_Block(extracted_feature_lid, 256, filter_height=1, filter_width=1, stride=1, padding='SAME',
                          batch_normalization=True, relu=True, name='aspp0_lid')

    # rate = 6 (12)
    b1_lid = SepConv_BN(extracted_feature_lid, 256, 'aspp1_lid', rate=atrous_rate[0], depth_activation=True)

    # rate = 12 (24)
    b2_lid = SepConv_BN(extracted_feature_lid, 256, 'aspp2_lid', rate=atrous_rate[1], depth_activation=True)

    # rate = 18 (36)
    b3_lid = SepConv_BN(extracted_feature_lid, 256, 'aspp3_lid', rate=atrous_rate[2], depth_activation=True)

    # concatenate ASPP branches & project
    # print("b4", b4_lid.shape)
    # print("b0", b0_lid.shape)
    # print("b1", b1_lid.shape)
    # print("b2", b2_lid.shape)

    concatenated_lid = Concat([b4_lid, b0_lid, b1_lid, b2_lid, b3_lid], axis=-1, name="concatenation_lid")
    concat_conv_lid = Conv2D_Block(concatenated_lid, 256, filter_height=1, filter_width=1, stride=1, padding='SAME',
                                   batch_normalization=True, relu=True, name='concatenation_conv_lid')
    concat_conv_lid = Dropout(concat_conv_lid, keep_prob=keep_prob)

    # Feature projection
    # x4 (x2) block
    size_before2 = tf.shape(xblock_entry_1_lid)
    dec_up1_lid = Resize_Bilinear(concat_conv_lid, size_before2[1:3], name="Upsampling2_lid")

    dec_skip1_lid = Conv2D_Block(skip1_lid, 48, filter_height=1, filter_width=1, stride=1, padding='SAME',
                                 batch_normalization=True, relu=True, name="feature_projection0_lid")
    # print("size_before2", xblock_entry_1_lid.shape)
    # print("dec_up", dec_up1_lid.shape)
    # print("skip", dec_skip1_lid.shape)

    dec_concatenated_lid = Concat([dec_up1_lid, dec_skip1_lid], axis=-1, name="concatenation_decoder0_lid")
    dec_block1_lid = SepConv_BN(dec_concatenated_lid, 256, 'decoder_conv0_lid', depth_activation=True)
    dec_block2_lid = SepConv_BN(dec_block1_lid, 256, 'decoder_conv1_lid', depth_activation=True)

    last_layer_lid = Conv2D_Block(dec_block2_lid, num_classess_lid, filter_height=1, filter_width=1, stride=1,
                                  padding='SAME', name='Last_layer_lid')

    size_before3 = tf.shape(x_lid)
    last_layer = Resize_Bilinear(last_layer_lid, size_before3[1:3], name="Last_Upsampling_lid")

    outputs_lid = tf.math.tanh(last_layer, name="output")

    return outputs_lid

def run():
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

    # Training 데이터 셋을 불러옴
    keep_probability = tf.placeholder(tf.float32, name="keep_probability")
    LiDAR_IMAGE = tf.placeholder(tf.float32, shape=[None, IMAGE_SHAPE_KITTI[0], IMAGE_SHAPE_KITTI[1], 3],
                                 name="input_lidar")

    prediction_lid = tf.placeholder(tf.float32, shape=[None, IMAGE_SHAPE_KITTI[0], IMAGE_SHAPE_KITTI[1], 1],
                                    name="prediction")

    # Network 선언
    logits_lid = LiDAR2IMAGE_Generator(LiDAR_IMAGE, keep_probability, NUM_OF_CLASSESS_LID)

    # Global step 선언
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # Tensorboard를 위한 summary들을 지정
    tf.summary.image('input_lidar', LiDAR_IMAGE, max_outputs=1)
    # 손실 함수를 선언하고 손실 함수에 대한 summary들을 지정
    loss = tf.losses.mean_squared_error(labels=prediction_lid, predictions=logits_lid)

    tf.summary.scalar('loss', loss)

    # 옵티마이저를 선언하고 파라미터를 한 스텝 업데이트하는 train_step 연산을 정의
    optimizer = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE, beta1=0.9, beta2=0.999, epsilon=1e-8)
    # train_step = optimizer.minimize(loss)

    # Constant to scale sum of gradient
    const = tf.compat.v1.constant(1 / BATCH_SIZE * 3)

    # Get all trainable variables
    t_vars = tf.compat.v1.trainable_variables()

    # Create a copy of all trainable variables with '0' as initial values
    accum_tvars = [tf.compat.v1.Variable(tf.compat.v1.zeros_like(t_var.initialized_value()), trainable=False) for t_var
                   in t_vars]

    # Create a op to initialize all accums vars
    zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_tvars]

    # Compute gradients for a batch
    batch_grads_vars = optimizer.compute_gradients(loss, t_vars)

    # Collect the (scaled by const) batch gradient into accumulated vars
    accum_ops = [accum_tvars[i].assign_add(tf.scalar_mul(const, batch_grad_var[0])) for i, batch_grad_var in
                 enumerate(batch_grads_vars)]

    # Apply accums gradients
    train_step = optimizer.apply_gradients(
        [(accum_tvars[i], batch_grad_var[1]) for i, batch_grad_var in enumerate(batch_grads_vars)], global_step=global_step)

    # Tensorboard를 위한 summary를 하나로 merge
    print("Setting up summary up")
    summary_op = tf.summary.merge_all()

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

    # training 데이터를 불러옴
    get_batches_fn = gen_batch_function(os.path.join(DATA_DIR, 'training'), IMAGE_SHAPE_KITTI)

    # 세션을 염
    sess = tf.Session(config=config)

    # 학습된 파라미터를 저장하기 위한 tf.train.Saver()
    # tensorboard summary들을 저장하기 위한 tf.summary.FileWriter를 선언
    print("Setting up Saver...")
    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter("logs/", sess.graph)

    # 변수들을 초기화하고 저장된 ckpt 파일이 있으면 저장된 파라미터를 불러옴
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state("./model")
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    else:
        sess.run(tf.global_variables_initializer())

    csv_path = os.path.join(DATA_DIR, "loss.csv")
    csvfile = open(csv_path, "w", newline="")
    start = time.time()  # 시작 시간 저장
    for epoch in range(880, EPOCHS):
        s_time = time.time()
        costs = []
        # 학습 데이터를 불러오고 feed_dict에 데이터를 지정
        for lidars, lid_labels in get_batches_fn(batch_size=BATCH_SIZE):
            feed_dict = {LiDAR_IMAGE: lidars, prediction_lid: lid_labels, keep_probability: KEEP_PROB}

            # Initialize the accumulated grads
            sess.run(zero_ops)
            for i in range(len(lidars)):
                sess.run(accum_ops, feed_dict=feed_dict)

            # train_step을 실행해서 파라미터를 한 스텝 업데이트 함
            _, cost = sess.run([train_step, loss], feed_dict=feed_dict)

            costs.append(cost)

            # Tensorboard 를 위한 sess.run()
            summary = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary, global_step=sess.run(global_step))

        if epoch != 0 | epoch % 30 == 0:
            # 매 30 epoch 마다 학습된 파라미터 저장
            # os.makedirs(os.path.join('model_{0}_Epochs'.format(epoch)))
            saver.save(sess, './model/ImageGenerator.ckpt', global_step=global_step)
            saver.save(sess, './model_{0}_Epochs/ImageGenerator.ckpt'.format(epoch), global_step=global_step)
            tf.io.write_graph(sess.graph_def, '.', './model/ImageGenerator.pb', as_text=False)
            tf.io.write_graph(sess.graph_def, '.', './model_{0}_Epochs/ImageGenerator.pb'.format(epoch), as_text=False)
            tf.io.write_graph(sess.graph_def, '.', './model/ImageGenerator.pbtxt', as_text=True)
            tf.io.write_graph(sess.graph_def, '.', './model_{0}_Epochs/ImageGenerator.pbtxt'.format(epoch), as_text=True)
            print("Training step in {0} epoch is finished".format(epoch))

            # 훈련 30 epoch 마다 validation 데이터 셋으로 테스트
            output_dir = os.path.join(DATA_DIR, 'generated_output_in_{0}_epoch'.format(epoch))
            print("Training Finished. Saving test images to: {}".format(output_dir))
            image_output = gen_test_output(sess, logits_lid, keep_probability, LiDAR_IMAGE,
                                           os.path.join(DATA_DIR, 'validating'),
                                           IMAGE_SHAPE_KITTI)

            for name, image, processing_time in image_output:
                scipy.misc.imsave(os.path.join(output_dir, name), image)

        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(costs)

        print("[Epoch: {0}/{1} Time: {2}]".format(epoch + 1, EPOCHS, str(timedelta(seconds=(time.time() - s_time)))))
    csvfile.close()
    print("Time: ", time.time() - start)  # 현재 시각 - 시작 시간 = 실행 시간
    print("Training Successfully")

    # 훈련이 끝나고 테스트 데이터 셋으로 테스트
    output_dir = os.path.join(DATA_DIR, 'generated_output')
    print("Training Finished. Saving test images to: {}".format(output_dir))
    image_output = gen_test_output(sess, logits_lid, keep_probability, LiDAR_IMAGE, os.path.join(DATA_DIR, 'validating'),
                                   IMAGE_SHAPE_KITTI)

    total_processing_time = 0
    for name, image, processing_time in image_output:
        print(image)

        plt.imshow(image)
        plt.show()
        scipy.misc.imsave(os.path.join(output_dir, name), image)
        total_processing_time += processing_time

    print("Average processing time is : ", total_processing_time / 30)


if __name__ == '__main__':
    run()