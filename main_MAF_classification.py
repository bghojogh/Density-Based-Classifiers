import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import os, random, json
from data.visu_density import plot_heatmap_2d
from data.plot_samples import plot_samples_2d
from utils.train_utils import sanity_check, train_density_estimation_noTfFunction, nll
from normalizingflows.flow_catalog import Made
from data.data_manager import Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List, Tuple
tf.random.set_seed(1234)


def load_dataset(config: Dict):
    # read data:
    path = config['train']['log_path']+f"data/"
    if config['train']['real_data']['split_data_again']:
        if config['train']['real_data']['test_data_path'] is not None:
            # train:
            df_train = pd.read_csv(config['train']['real_data']['train_data_path'])
            X_train = df_train[config['train']['real_data']['features']]
            y_train = df_train[config['train']['real_data']['label_feature']]
            # test:
            df_test = pd.read_csv(config['train']['real_data']['test_data_path'])
            X_test = df_test[config['train']['real_data']['features']]
            y_test = df_test[config['train']['real_data']['label_feature']]
            # val:
            df_val = pd.read_csv(config['train']['real_data']['val_data_path'])
            X_val = df_val[config['train']['real_data']['features']]
            y_val = df_val[config['train']['real_data']['label_feature']]
        else:
            # read data:
            df = pd.read_csv(config['train']['real_data']['train_data_path'])
            X = df[config['train']['real_data']['features']]
            y = df[config['train']['real_data']['label_feature']]
            # split data to train, test, and val:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2
        # save the train, test, and val data:
        if not os.path.exists(path): os.makedirs(path)
        np.save(path+'X_train.npy', X_train)
        np.save(path+'X_val.npy', X_val)
        np.save(path+'X_test.npy', X_test)
        np.save(path+'y_train.npy', y_train)
        np.save(path+'y_val.npy', y_val)
        np.save(path+'y_test.npy', y_test)
    else:
        X_train = np.load(path+'X_train.npy')
        X_val = np.load(path+'X_val.npy')
        X_test = np.load(path+'X_test.npy')
        y_train = np.load(path+'y_train.npy')
        y_val = np.load(path+'y_val.npy')
        y_test = np.load(path+'y_test.npy')
    
    # numerical and categorical features:
    categorical_features = config['train']['real_data']['categorical_features']
    numeric_features = [col for col in config['train']['real_data']['features'] if col not in categorical_features]
    
    # preprocess data:
    numeric_transformer = Pipeline(
        steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())]
    )
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    data_pipeline = ColumnTransformer([
        ('numerical', numeric_transformer, numeric_features),
        ('categorical', categorical_transformer, categorical_features)
    ])
    X_train_processed = data_pipeline.fit_transform(X_train)
    X_test_processed = data_pipeline.transform(X_test)
    X_val_processed = data_pipeline.transform(X_val)
    
    # convert data to float32 --> important: otherwise, the network gives error!
    X_train_processed = X_train_processed.astype(np.float32)
    X_test_processed = X_test_processed.astype(np.float32)
    X_val_processed = X_val_processed.astype(np.float32)

    # split to classes:
    batched_train_data_list, train_data_list, val_data_list, test_data_list = [], [], [], []
    n_classes = len(np.unique(y_train))
    for class_index in range(n_classes):
        X_train_processed_class = X_train_processed[y_train==class_index, :]
        X_test_processed_class = X_test_processed[y_test==class_index, :]
        X_val_processed_class = X_val_processed[y_val==class_index, :]

        # make batched train data:
        batched_train_data = tf.data.Dataset.from_tensor_slices(X_train_processed_class)
        BATCH_SIZE = config['train']['batch_size']
        SHUFFLE_BUFFER_SIZE = 100
        batched_train_data = batched_train_data.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

        # make the list:
        batched_train_data_list.append(batched_train_data)
        train_data_list.append(X_train_processed_class)
        test_data_list.append(X_test_processed_class)
        val_data_list.append(X_val_processed_class)
    
    return batched_train_data_list, train_data_list, val_data_list, test_data_list

def train(config: Dict) -> None:

    # load data:
    if config['train']['data_type'] == 'real_data':
        batched_train_data_list, train_data_list, val_data_list, test_data_list = load_dataset(config=config)

    for class_index in tqdm(range(config['train']['n_classes']), desc='Classes'):
        # set the reset flag for training tensorflow function:
        shoud_reset = True

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

        ############ Build the Normalizing Flow:

        # settings for the network:
        hidden_shape = config['train']['hidden_shape']  # hidden shape for MADE network of MAF
        layers = config['train']['layers']  # number of layers of the flow

        tfd = tfp.distributions
        tfb = tfp.bijectors

        # specify base distribution:
        base_dist = tfd.Normal(loc=0.0, scale=1.0)

        # build the network:
        bijectors = []
        for i in range(0, layers):
            bijectors.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn = Made(params=2, hidden_units=hidden_shape, activation="relu")))
            bijectors.append(tfb.Permute(permutation=[i for i in range(n_dimensions)][::-1]))  # data permutation after layers of MAF

        bijector = tfb.Chain(bijectors=list(reversed(bijectors)), name='chain_of_maf')

        maf = tfd.TransformedDistribution(
            distribution=tfd.Sample(base_dist, sample_shape=[n_dimensions]),
            bijector=bijector,
        )

        # initialize flow
        samples = maf.sample()

        ############ Train the Normalizing Flow:

        # settings of training:
        base_lr = config['train']['base_lr']
        end_lr = config['train']['end_lr']
        max_epochs = int(config['train']['max_epochs'])  # maximum number of epochs of the training
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(base_lr, max_epochs, end_lr, power=0.5)

        # initialize checkpoints:
        checkpoint_directory = config['train']['log_path']+"class_{}/checkpoint".format(class_index)
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

        # optimizer and checkpoint:
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)  # optimizer
        checkpoint = tf.train.Checkpoint(optimizer=opt, model=maf)

        global_step = []
        train_losses = []
        val_losses = []
        min_val_loss = tf.convert_to_tensor(np.inf, dtype=tf.float32)  # high value to ensure that first loss < min_loss
        min_train_loss = tf.convert_to_tensor(np.inf, dtype=tf.float32)
        min_val_epoch = 0
        min_train_epoch = 0
        delta_stop = config['train']['delta_stop_in_early_stopping']  # threshold for early stopping

        t_start = time.time()  # start time

        # start training
        for i in tqdm(range(max_epochs), desc='Epochs', leave=False):
            for batch in batched_train_data:
                if shoud_reset:
                    train_density_estimation_ = tf.function(train_density_estimation_noTfFunction)
                    train_loss = train_density_estimation_(maf, opt, batch)
                    shoud_reset = False
                else:
                    train_loss = train_density_estimation_(maf, opt, batch)

            if i % int(config['train']['frequency_validation']) == 0:
                val_loss = nll(maf, val_data)
                global_step.append(i)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"{i}, train_loss: {train_loss}, val_loss: {val_loss}")

                if train_loss < min_train_loss:
                    min_train_loss = train_loss
                    min_train_epoch = i

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    min_val_epoch = i
                    checkpoint.write(file_prefix=checkpoint_prefix)  # overwrite best val model

                elif i - min_val_epoch > delta_stop:  # no decrease in min_val_loss for "delta_stop epochs"
                    break

            if config['train']['plot_data'] and (i % int(config['train']['frequency_plot']) == 0):
                # plot heatmap every multiple epochs
                plot_heatmap_2d(maf, -4.0, 4.0, -4.0, 4.0, mesh_count=200, path=config['train']['log_path']+f"class_{class_index}/plots/heatmap/", name=f'epoch_{i}')

        train_time = time.time() - t_start

        ############ evaluate:
        # load best model with min validation loss
        checkpoint.restore(checkpoint_prefix)

        # perform on test dataset
        t_start = time.time()
        test_loss = nll(maf, test_data)
        test_time = time.time() - t_start

        # plot density estimation of the best model
        if config['train']['plot_data']:
            plot_heatmap_2d(maf, -4.0, 4.0, -4.0, 4.0, mesh_count=200, path=config['train']['log_path']+f"class_{class_index}/plots/", name='evaluate1') 

        # plot samples of the best model
        if config['train']['plot_data']:
            plot_samples_2d(maf.sample(1000), path=config['train']['log_path']+f"class_{class_index}/plots/", name='evaluate2')

def load_checkpoint(config: Dict, class_index: int) -> tfp.distributions.TransformedDistribution:
    # settings for the network:
    hidden_shape = config['train']['hidden_shape']  # hidden shape for MADE network of MAF
    layers = config['train']['layers']  # number of layers of the flow

    tfd = tfp.distributions
    tfb = tfp.bijectors

    # specify base distribution:
    base_dist = tfd.Normal(loc=0.0, scale=1.0)  

    # build the network:
    bijectors = []
    for i in range(0, layers):
        bijectors.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn = Made(params=2, hidden_units=hidden_shape, activation="relu")))
        bijectors.append(tfb.Permute(permutation=[1, 0]))  # data permutation after layers of MAF

    bijector = tfb.Chain(bijectors=list(reversed(bijectors)), name='chain_of_maf')

    maf = tfd.TransformedDistribution(
            distribution=tfd.Sample(base_dist, sample_shape=[2]),
            bijector=bijector,
        )

    # initialize flow
    samples = maf.sample()
    
    # settings of training:
    base_lr = config['train']['base_lr']
    end_lr = config['train']['end_lr']
    max_epochs = int(config['train']['max_epochs'])  # maximum number of epochs of the training
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(base_lr, max_epochs, end_lr, power=0.5)

    # initialize checkpoints:
    checkpoint_directory = config['train']['log_path']+"class_{}/checkpoint".format(class_index)
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    # optimizer and checkpoint:
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)  # optimizer
    checkpoint = tf.train.Checkpoint(optimizer=opt, model=maf)

    # load best model with min validation loss
    checkpoint.restore(checkpoint_prefix)

    return maf

def eval(config: Dict) -> Tuple[List[int], List[int], np.ndarray, List[float], List[float]]:
    # load test data:
    for class_index in range(config['train']['n_classes']):
        test_data = np.load(config['train']['log_path']+f"class_{class_index}/data/test_data.npy")
        test_label = [class_index for i in range(test_data.shape[0])]
        if class_index == 0: 
            X_test = test_data.copy()
            y_test = test_label.copy()
        else:
            X_test = np.vstack((X_test, test_data))
            y_test.extend(test_label)

    pred_prob = np.zeros((X_test.shape[0], config['train']['n_classes']))
    for class_index in tqdm(range(config['train']['n_classes']), desc='Classes'):
        # load the checkpoint for this class:
        maf = load_checkpoint(config=config, class_index=class_index)

        # perform on test dataset
        t_start = time.time()
        test_loss = nll(maf, X_test)
        test_time = time.time() - t_start

        # calculate the predicted probabilities:
        prob = maf.prob(X_test)
        prob = prob.numpy()
        pred_prob[:, class_index] = prob

        # plot density estimation of the best model
        if config['train']['plot_data']:
            plot_heatmap_2d(maf, -4.0, 4.0, -4.0, 4.0, mesh_count=200, path=config['eval']['log_path']+f"class_{class_index}/plots/", name='evaluate1') 

        # plot samples of the best model
        if config['train']['plot_data']:
            plot_samples_2d(maf.sample(1000), path=config['eval']['log_path']+f"class_{class_index}/plots/", name='evaluate2') 

    # calculate the predicted classes:
    y_pred = np.argmax(pred_prob, axis=1).tolist()
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    f1score = f1_score(y_true=y_test, y_pred=y_pred)
    print(f'accuracy: {accuracy}, f1 score: {f1score}')
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
        # load the checkpoint for this class:
        maf = load_checkpoint(config=config, class_index=class_index)

        # perform on test dataset
        t_start = time.time()
        test_loss = nll(maf, X_test)
        test_time = time.time() - t_start

        # calculate the predicted probabilities:
        prob = maf.prob(X_test)
        prob = prob.numpy()
        pred_prob[:, class_index] = prob

        # plot density estimation of the best model
        if config['train']['plot_data']:
            plot_heatmap_2d(maf, -4.0, 4.0, -4.0, 4.0, mesh_count=200, path=config['eval']['log_path']+f"class_{class_index}/plots/", name='evaluate1') 

        # plot samples of the best model
        if config['train']['plot_data']:
            plot_samples_2d(maf.sample(1000), path=config['eval']['log_path']+f"class_{class_index}/plots/", name='evaluate2') 

    # calculate the predicted classes:
    not_classified_indices = np.where(pred_prob.sum(axis=1)==0)[0].tolist()
    y_pred = np.argmax(pred_prob, axis=1).tolist()
    y_pred_final = [y_pred[i] if (i not in not_classified_indices) else np.max(y_pred)+1 for i in range(len(y_pred))]

    # plot the predicted labels on the mesh:
    plt.close()
    if config['train']['n_classes'] == 2:
        color_map = 'brg'
    else:
        color_map = 'Spectral'
    plt.imshow(tf.transpose(tf.reshape(y_pred_final, (mesh_count, mesh_count))), origin="lower", cmap=color_map)
    if not os.path.exists(config['eval']['log_path']+f"plots/"): os.makedirs(config['eval']['log_path']+f"plots/")
    plt.savefig(config['eval']['log_path'] + f"plots/" + f"mesh_predicted.png", format="png", dpi=300)

    return y_pred_final, X_test, mesh_count

if __name__ == '__main__':
    with open('./config/config_maf.json', 'r') as f:
        config = json.load(f)
    if config['stage'] == 'train':
        train(config=config)
    elif config['stage'] == 'eval':
        eval(config=config)
    elif config['stage'] == 'eval_mesh':
        eval_mesh(config=config)