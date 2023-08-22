import numpy as np
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
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict
tf.random.set_seed(1234)


def train(config: Dict) -> None:

    for class_index in tqdm(range(config['train']['n_classes']), desc='Classes'):
        # set the reset flag for training tensorflow function:
        shoud_reset = True

        # dataset:
        dataset = Dataset(dataset_name=config['train']['dataset_name'], batch_size=config['train']['batch_size'], data_size=config['train']['dataset_size'], classification=True, category=class_index)
        batched_train_data, train_data, val_data, test_data = dataset.get_data2()
        if not os.path.exists(config['train']['log_path']+f"{config['train']['dataset_name']}/class_{class_index}/data/"): os.makedirs(config['train']['log_path']+f"{config['train']['dataset_name']}/class_{class_index}/data/")
        np.save(config['train']['log_path']+f"{config['train']['dataset_name']}/class_{class_index}/data/train_data.npy", train_data)
        np.save(config['train']['log_path']+f"{config['train']['dataset_name']}/class_{class_index}/data/val_data.npy", val_data)
        np.save(config['train']['log_path']+f"{config['train']['dataset_name']}/class_{class_index}/data/test_data.npy", test_data)
        sample_batch = next(iter(batched_train_data))
        if config['train']['plot_data']:
            plot_samples_2d(sample_batch, path=config['train']['log_path']+f"{config['train']['dataset_name']}/class_{class_index}/plots/", name='dataset')

        ############ Build the Normalizing Flow:

        # settings for the network:
        hidden_shape = [200, 200]  # hidden shape for MADE network of MAF
        layers = 12  # number of layers of the flow

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

        ############ Train the Normalizing Flow:

        # settings of training:
        base_lr = config['train']['base_lr']
        end_lr = config['train']['end_lr']
        max_epochs = int(config['train']['max_epochs'])  # maximum number of epochs of the training
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(base_lr, max_epochs, end_lr, power=0.5)

        # initialize checkpoints:
        checkpoint_directory = config['train']['log_path']+"{}/class_{}/checkpoint".format(config['train']['dataset_name'], class_index)
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
                plot_heatmap_2d(maf, -4.0, 4.0, -4.0, 4.0, mesh_count=200, path=config['train']['log_path']+f"{config['train']['dataset_name']}/class_{class_index}/plots/heatmap/", name=f'epoch_{i}')

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
            plot_heatmap_2d(maf, -4.0, 4.0, -4.0, 4.0, mesh_count=200, path=config['train']['log_path']+f"{config['train']['dataset_name']}/class_{class_index}/plots/", name='evaluate1') 

        # plot samples of the best model
        if config['train']['plot_data']:
            plot_samples_2d(maf.sample(1000), path=config['train']['log_path']+f"{config['train']['dataset_name']}/class_{class_index}/plots/", name='evaluate2')

def load_checkpoint(config: Dict, class_index: int) -> tfp.distributions.TransformedDistribution:
    # settings for the network:
    hidden_shape = [200, 200]  # hidden shape for MADE network of MAF
    layers = 12  # number of layers of the flow

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
    checkpoint_directory = config['train']['log_path']+"{}/class_{}/checkpoint".format(config['train']['dataset_name'], class_index)
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    # optimizer and checkpoint:
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)  # optimizer
    checkpoint = tf.train.Checkpoint(optimizer=opt, model=maf)

    # load best model with min validation loss
    checkpoint.restore(checkpoint_prefix)

    return maf

def eval(config: Dict) -> None:
    # load test data:
    for class_index in range(config['train']['n_classes']):
        test_data = np.load(config['train']['log_path']+f"{config['train']['dataset_name']}/class_{class_index}/data/test_data.npy")
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
            plot_heatmap_2d(maf, -4.0, 4.0, -4.0, 4.0, mesh_count=200, path=config['eval']['log_path']+f"{config['train']['dataset_name']}/class_{class_index}/plots/", name='evaluate1') 

        # plot samples of the best model
        if config['train']['plot_data']:
            plot_samples_2d(maf.sample(1000), path=config['eval']['log_path']+f"{config['train']['dataset_name']}/class_{class_index}/plots/", name='evaluate2') 

    # calculate the predicted classes:
    y_pred = np.argmax(pred_prob, axis=1).tolist()
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    f1score = f1_score(y_true=y_test, y_pred=y_pred)
    print(f'accuracy: {accuracy}, f1 score: {f1score}')
    return y_pred, y_test, X_test, accuracy, f1score

if __name__ == '__main__':
    with open('./config/config.json', 'r') as f:
        config = json.load(f)
    if config['stage'] == 'train':
        train(config=config)
    elif config['stage'] == 'eval':
        eval(config=config)