# See: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from typing import Dict, List, Tuple
import os, random, json, pickle
from data.visu_density import plot_heatmap_2d
from data.plot_samples import plot_samples_2d
from data.data_manager import Dataset
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import tensorflow as tf
from main_MAF_classification import load_dataset


def train(config: Dict):
    # load data:
    if config['train']['data_type'] == 'real_data':
        batched_train_data_list, train_data_list, val_data_list, test_data_list = load_dataset(config=config, split_data_again=config['train']['real_data']['split_data_again'])
        n_classes = len(batched_train_data_list)
    elif config['train']['data_type'] == 'toy_data':
        n_classes = config['train']['toy_data']['n_classes']

    for class_index in tqdm(range(n_classes), desc='Classes'):
        # dataset:
        if config['train']['data_type'] == 'toy_data':
            dataset = Dataset(dataset_name=config['train']['toy_data']['dataset_name'], batch_size=config['train']['batch_size'], data_size=config['train']['toy_data']['dataset_size'], classification=True, category=class_index)
            batched_train_data, train_data, val_data, test_data = dataset.get_data2()
        elif config['train']['data_type'] == 'real_data':
            batched_train_data, train_data, val_data, test_data = batched_train_data_list[class_index], train_data_list[class_index], val_data_list[class_index], test_data_list[class_index]
        else:
            raise ValueError('The data_type is config is not valid!')

        # save the data of classes:
        if not os.path.exists(config['train']['log_path']+f"class_{class_index}/data/"): os.makedirs(config['train']['log_path']+f"class_{class_index}/data/")
        np.save(config['train']['log_path']+f"class_{class_index}/data/train_data.npy", train_data)
        np.save(config['train']['log_path']+f"class_{class_index}/data/val_data.npy", val_data)
        np.save(config['train']['log_path']+f"class_{class_index}/data/test_data.npy", test_data)
        n_dimensions = train_data.shape[1]
        if config['train']['plot_data']:
            if n_dimensions != 2: raise AssertionError('The dimensionality of data is not 2 and cannot be plotted! Turn off plot_data in the config.')
            sample_batch = next(iter(batched_train_data))
            plot_samples_2d(sample_batch, path=config['train']['log_path']+f"class_{class_index}/plots/", name='dataset')

        # GMM train:
        gmm = GaussianMixture(n_components=config['train']['n_components'], random_state=0).fit(X=train_data)

        # save the gmm model:
        path_save = config['train']['log_path']+f"class_{class_index}/"
        if not os.path.exists(path_save): os.makedirs(path_save)
        with open(path_save+'gmm.pickle', 'wb') as handle:
            pickle.dump(gmm, handle, protocol=pickle.HIGHEST_PROTOCOL)

def eval(config: Dict) -> Tuple[List[int], List[int], np.ndarray, List[float], List[float]]:
    # get the number of classes:
    if config['train']['data_type'] == 'toy_data':
        n_classes = config['train']['toy_data']['n_classes']
    elif config['train']['data_type'] == 'real_data':
        n_classes = len([f.path for f in os.scandir(config['train']['log_path']) if f.is_dir() and 'class_' in f.path])

    # load training data:
    if config['eval']['use_posterior']:
        population_of_classes = []
        for class_index in range(n_classes):
            train_data = np.load(config['train']['log_path']+f"class_{class_index}/data/train_data.npy")
            population_of_classes.append(train_data.shape[0])
        priors_of_classes = [population_of_class/np.sum(population_of_classes) for population_of_class in population_of_classes]

    # load test data:
    for class_index in range(n_classes):
        test_data = np.load(config['train']['log_path']+f"class_{class_index}/data/test_data.npy")
        test_label = [class_index for i in range(test_data.shape[0])]
        if class_index == 0: 
            X_test = test_data.copy()
            y_test = test_label.copy()
        else:
            X_test = np.vstack((X_test, test_data))
            y_test.extend(test_label)

    pred_prob = np.zeros((X_test.shape[0], n_classes))
    for class_index in tqdm(range(n_classes), desc='Classes'):
        # load the model for this class:
        path_save = config['train']['log_path']+f"class_{class_index}/"
        with open(path_save+'gmm.pickle', 'rb') as handle:
            gmm = pickle.load(handle)

        # calculate the predicted probabilities:
        prob = np.exp(gmm.score_samples(X_test))
        if config['eval']['use_posterior']:
            prob = prob * priors_of_classes[class_index]
        pred_prob[:, class_index] = prob

    # calculate the predicted classes:
    y_pred = np.argmax(pred_prob, axis=1).tolist()
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    f1score = f1_score(y_true=y_test, y_pred=y_pred)
    print(f'accuracy: {accuracy}, f1 score: {f1score}')

    # save the predicted labels:
    df = pd.DataFrame()
    df['y_true'] = y_test
    df['y_pred'] = y_pred
    df.loc[0, 'accuracy'] = accuracy
    df.loc[0, 'f1score'] = f1score
    df_X = pd.DataFrame(X_test)
    if not os.path.exists(config['eval']['log_path']): os.makedirs(config['eval']['log_path'])
    df.to_csv(config['eval']['log_path']+'df_predicted.csv', index=False)
    df_X.to_csv(config['eval']['log_path']+'df_X.csv', index=False)

    return y_pred, y_test, X_test, accuracy, f1score

def eval_mesh(config: Dict) -> Tuple[List[int], np.ndarray, int]:
    # load test data:
    xmin, xmax, ymin, ymax, mesh_count = -4.0, 4.0, -4.0, 4.0, 200
    x = tf.linspace(xmin, xmax, mesh_count)
    y = tf.linspace(ymin, ymax, mesh_count)
    X, Y = tf.meshgrid(x, y)
    concatenated_mesh_coordinates = tf.transpose(tf.stack([tf.reshape(Y, [-1]), tf.reshape(X, [-1])]))
    X_test = concatenated_mesh_coordinates

    pred_prob = np.zeros((X_test.shape[0], config['train']['n_classes']))
    for class_index in tqdm(range(config['train']['n_classes']), desc='Classes'):
        # load the model for this class:
        path_save = config['train']['log_path']+f"{config['train']['dataset_name']}/class_{class_index}/"
        with open(path_save+'gmm.pickle', 'rb') as handle:
            gmm = pickle.load(handle)

        # calculate the predicted probabilities:
        prob = np.exp(gmm.score_samples(X_test))
        pred_prob[:, class_index] = prob

    # calculate the predicted classes:
    not_classified_indices = np.where(pred_prob.sum(axis=1)==0)[0].tolist()
    y_pred = np.argmax(pred_prob, axis=1).tolist()
    y_pred_final = [y_pred[i] if (i not in not_classified_indices) else np.max(y_pred)+1 for i in range(len(y_pred))]

    # plot the predicted labels on the mesh:
    plt.close()
    if config['train']['n_classes'] == 2:
        # color_map = 'brg'
        color_map = 'bwr'
    else:
        color_map = 'Spectral'
    plt.imshow(tf.transpose(tf.reshape(y_pred_final, (mesh_count, mesh_count))), origin="lower", cmap=color_map)
    if not os.path.exists(config['eval']['log_path']+f"{config['train']['dataset_name']}/plots/"): os.makedirs(config['eval']['log_path']+f"{config['train']['dataset_name']}/plots/")
    plt.savefig(config['eval']['log_path'] + f"{config['train']['dataset_name']}/plots/" + f"{config['train']['dataset_name']}_predicted.png", format="png", dpi=300)

    return y_pred_final, X_test, mesh_count

if __name__ == '__main__':
    with open('./config/config_gmm.json', 'r') as f:
        config = json.load(f)
    if config['stage'] == 'train':
        train(config=config)
    elif config['stage'] == 'eval':
        eval(config=config)
    elif config['stage'] == 'eval_mesh':
        eval_mesh(config=config)