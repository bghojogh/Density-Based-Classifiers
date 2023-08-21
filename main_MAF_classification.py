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
from typing import Dict
tf.random.set_seed(1234)


def train(config: Dict) -> None:

    for class_index in tqdm(range(config['train']['n_classes']), desc='Classes'):
        # set the reset flag for training tensorflow function:
        shoud_reset = True

        # dataset:
        dataset = Dataset(dataset_name=config['train']['dataset_name'], batch_size=config['train']['batch_size'], data_size=config['train']['dataset_size'], classification=True, category=class_index)
        batched_train_data, val_data, test_data = dataset.get_data()
        sample_batch = next(iter(batched_train_data))
        if config['train']['plot_data']:
            plot_samples_2d(sample_batch, path=f"log/{config['train']['dataset_name']}/class_{class_index}/plots/", name='dataset')

        ############ Build the Normalizing Flow:

        # settings for the network:
        hidden_shape = [200, 200]  # hidden shape for MADE network of MAF
        layers = 12  # number of layers of the flow

        if class_index == 1:
            tf.keras.backend.clear_session()

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
        checkpoint_directory = "log/{}/class_{}/tmp_{}".format(config['train']['dataset_name'], class_index, str(hex(random.getrandbits(32))))
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
                plot_heatmap_2d(maf, -4.0, 4.0, -4.0, 4.0, mesh_count=200, path=f"log/{config['train']['dataset_name']}/class_{class_index}/plots/heatmap/", name=f'epoch_{i}')

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
            plot_heatmap_2d(maf, -4.0, 4.0, -4.0, 4.0, mesh_count=200, path=f"log/{config['train']['dataset_name']}/class_{class_index}/plots/", name='evaluate1') 

        # plot samples of the best model
        if config['train']['plot_data']:
            plot_samples_2d(maf.sample(1000), path=f"log/{config['train']['dataset_name']}/class_{class_index}/plots/", name='evaluate2') 

if __name__ == '__main__':
    with open('./config/config.json', 'r') as f:
        config = json.load(f)
    train(config=config)