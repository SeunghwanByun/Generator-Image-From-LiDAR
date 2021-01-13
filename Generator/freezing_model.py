# https://blog.metaflow.fr/tensorflow-saving-restoring-and-mixing-multiple-models-c4c94d5d7125

import tensorflow as tf
import argparse

def freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant

    :param model_dir: the root folder containing the checkpoint state file
    :param output_node_names: a string, containing all the output node's names, comma separated
    :return: graph
    """
    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    # absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    # output_graph = absolute_model_dir + "/frozen_model.pb"
    output_graph = "./frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            print("Success")
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def

def main(model_dir, output_node_names):
    freeze_graph(model_dir, output_node_names)

if __name__ == '__main__':
    # If you want use 'argparse' module, you can write codes like below.
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_dir", type=str, default="", help="Model folder to export")
    # parser.add_argument("--output_node_names", type=stf, default="", help="The name of the output nodes, comma separated.")
    # args = parser.parse_args()

    model_dir = 'model_2100_Epochs'
    output_node_names = "input_lidar,output,keep_probability,entry_flow_conv1_2_BN_lid/Conv2D,entry_flow_block1_lid_add_conv,entry_flow_block2_lid_separable_conv2_pointwise_BN/Conv2D,middle_flow_unit_16_lid_separable_conv3_pointwise_BN/Conv2D,exit_flow_block2_lid_separable_conv3_pointwise_BN/Conv2D"

    main(model_dir, output_node_names)