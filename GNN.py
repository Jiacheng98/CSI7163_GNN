import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  

import stellargraph as sg
from stellargraph import StellarGraph
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import GCNSupervisedGraphClassification
from stellargraph import StellarGraph

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

np.random.seed(1)
tf.random.set_seed(1)

import logging
Log_Format = "%(message)s"
logging.basicConfig(filename = "logfile.log",
                    filemode = "w+",
                    format = Log_Format, 
                    level = logging.INFO)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logger = logging.getLogger()


activities = {
    1: 'WALKING',
    2: 'WALKING_UPSTAIRS',
    3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING',
    5: 'STANDING',
    6: 'LAYING',
}


features = {
    "body_acc_x": 0, 
    "body_acc_y": 1,
    "body_acc_z": 2,
    "body_gyro_x": 3,
    "body_gyro_y": 4,
    "body_gyro_z": 5,
    "total_acc_x": 6,
    "total_acc_y": 7,
    "total_acc_z": 8
    }


timestep = 128
features_number = 9

lr = 0.001
epochs = 200
batch_size = 100



def main():
    for data in ['train', 'test']:
        # convert training and testing data into numpy array
        input_features, label = convert_data_into_numpy(data)
        # load numpy array to StellarGraph objects, each with nodes features and graph structures
        graphs_list = load_to_stellargraph(input_features)

        # create a data generator in order to feed data into the tf.Keras model
        generator = PaddedGraphGenerator(graphs=graphs_list)
        sample_index = [i for i in range(input_features.shape[0])]

        if data == "train":
            train_gen = generator.flow(sample_index, targets = label, batch_size = batch_size)
        elif data == "test":
            test_gen = generator.flow(sample_index, targets = label, batch_size = batch_size)

    model = graph_classificaiton_model(generator)
    # apply early stopping and save the best model
    es = EarlyStopping(monitor="val_loss", min_delta=0, patience=50, restore_best_weights=True)
    mc = ModelCheckpoint('gnn_model', monitor='val_loss', mode='min', save_best_only=True)
    history = model.fit(train_gen, epochs=epochs, verbose=0, validation_data=test_gen, shuffle=True, callbacks=[es, mc])
    plot(history, "acc", "accuracy")
    plot(history, "loss", "loss")

    # convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model('gnn_model/')
    tflite_model = converter.convert()
    # save the model as .tflite
    with open('gnn_model/gnn.tflite', 'wb') as f:
      f.write(tflite_model)



def plot(history, metrics, full_name):
    plt.plot(history.history[metrics])
    plt.plot(history.history[f'val_{metrics}'])
    if full_name == "accuracy":
        plt.yticks(np.arange(0.2, 1.1, 0.1))
    else:
        plt.yticks(np.arange(0, 1.5, 0.2))
    plt.title(f'GNN {full_name}')
    plt.ylabel(full_name)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    plt.savefig(f"plot/GNN_{full_name}.jpeg")
    plt.close()



def convert_data_into_numpy(file_name):
    df = pd.DataFrame() 
    data_path = f"UCI HAR Dataset/{file_name}"

    # labels
    label = []
    with open(f"{data_path}/y_{file_name}.txt", "r") as f:
        for position, line in enumerate(f):
            if len(line) > 0:
                label.append(int(line))
    label = np.array(label)
    logger.info(f"Number of {file_name} samples: {len(label)}")

    # load input features, shape: (7352, 128, 9)
    input_features = np.zeros((len(label), timestep, features_number), dtype=np.float32)
    logger.info(f"Shape of input features: {np.shape(input_features)}")

    input_features_file_path = f"{data_path}/Inertial Signals"
    for filename in os.listdir(input_features_file_path):
        with open(f"{input_features_file_path}/{filename}", "r") as f:
            for position, line in enumerate(f):
                values_list = line.split(" ")
                values_list = [float(x) for x in line.split(" ") if len(x) != 0]
                assert len(values_list) == timestep
                feature_name = '_'.join(x for x in filename.split('_')[:-1])
                feature_order = features[feature_name]
                for each_timestamp in range(timestep):
                    input_features[position][each_timestamp][feature_order] = values_list[each_timestamp]

    assert len(input_features) == len(label)

    # dictionary: {1: {'body_acc_x': [...], 'body_acc_y': [...], ...}, 2: {'body_acc_x': [...], 'body_acc_y': [...], ...}}
    # used to plot each feature for each class, prove features can be distinguished among different classes
    output_input_dict = dict()
    for class_type in activities:
        output_input_dict[class_type] = dict()
        for each_feature in features:
            output_input_dict[class_type][each_feature]= []

    features_inverse = {v: k for k, v in features.items()}
    for sample_index in range(len(input_features)):
        for timestamp_index in range(len(input_features[sample_index])):
            for feature_index in range(len(input_features[sample_index][timestamp_index])):
                feature_name = features_inverse[feature_index]
                output_input_dict[label[sample_index]][feature_name].append(input_features[sample_index][timestamp_index][feature_index])
    # print(output_input_dict)

    # plot the 9 fetures for each class
    for class_type in activities:
        for each_feature in features:
            plt.plot(np.arange(0, 50), output_input_dict[class_type][each_feature][:50], label = each_feature)
        plt.legend(title=activities[class_type],title_fontsize=10,loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(f"plot/{activities[class_type]}.jpeg", bbox_inches='tight')
        plt.close()

    return input_features, label



def load_to_stellargraph(input_features):
    # create graph edges
    source_node = []
    target_node = []
    for node in range(timestep - 1):
        source_node.append(node)
        target_node.append(node + 1)
    edges = pd.DataFrame({"source": source_node, "target": target_node})

    graphs_list = []
    for each_sample in range(np.shape(input_features)[0]):
        each_graph_feature_array = input_features[each_sample]
        # construct StellarGraph by passing the nodes features and graph edges
        each_graph = StellarGraph(each_graph_feature_array, edges)
        graphs_list.append(each_graph)
    logger.info(f"A graph sample: {graphs_list[0].info()}")

    summary = pd.DataFrame(
        [(g.number_of_nodes(), g.number_of_edges()) for g in graphs_list],
        columns=["nodes", "edges"],
    )
    logger.info(f"All graphs summary: {summary.describe().round(1)}")
    return graphs_list


def graph_classificaiton_model(generator):
    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=[64, 64, 64],
        activations=["relu", "relu", "relu"],
        generator=generator,
        dropout=0.1
    )
    x_inp, x_out = gc_model.in_out_tensors()
    predictions = Dense(units=32, activation="relu")(x_out)
    predictions = Dense(units=16, activation="relu")(predictions)
    predictions = Dense(units=7, activation="softmax")(predictions)

    # create the Keras model
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(lr), loss=sparse_categorical_crossentropy, metrics=["acc"])

    model.summary(print_fn=logger.info)
    return model



if __name__ == "__main__":
    main()


