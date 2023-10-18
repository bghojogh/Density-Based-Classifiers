# Density Based Classifiers (GMM and MAF)

This is the code for the following paper:
- Benyamin Ghojogh, Milad Amir Toutounchian, "Probabilistic Classification by Density Estimation Using Gaussian Mixture Model and Masked Autoregressive Flow", arXiv, 2023.
- Link of paper: https://arxiv.org/abs/2310.10843

The classifiers in this paper and code are:
- Gaussian Mixture Model (GMM)
- Masked Autoregressive Flow (MAF), containing normalizing flows and Masked Autoencoder for Distribution Estimation (MADE)

# Related repositories

- `https://github.com/LukasRinder/normalizing-flows/tree/master`
- `https://github.com/LukasRinder/normalizing-flows/blob/master/example_training.ipynb`

# Packages

```bash
pip install tensorflow==2.11.1
pip install tensorflow-probability==0.15.0
pip install tensorflow-datasets==4.4.0
conda install -c anaconda numpy
conda install -c conda-forge matplotlib
conda install -c anaconda pandas
conda install -c anaconda scikit-learn
```

# Config of MAF classifier for the datasets:

- Toy data:

```json
{
    "stage": "train",
    "train": {
        "data_type": "toy_data",
        "real_data": {
            "train_data_path": "./dataset/SAHeart/SAHeart.csv",
            "test_data_path": null,
            "val_data_path": null,
            "split_data_again": true,
            "features": ["sbp", "tobacco", "ldl", "adiposity", "famhist", "typea", "obesity", "alcohol", "age"],
            "label_feature": "chd",
            "categorical_features": ["famhist"]
        },
        "toy_data": {
            "dataset_name": "circles",
            "dataset_size": 2000,
            "n_classes": 2
        },
        "log_path": "./log_train/",
        "batch_size": 800,
        "hidden_shape": [200, 200],
        "layers": 12,
        "base_lr": 1e-3,
        "end_lr": 1e-4,
        "max_epochs": 5e3,
        "delta_stop_in_early_stopping": 1000,
        "frequency_validation": 100,
        "frequency_plot": 1000,
        "plot_data": false
    },
    "eval": {
        "log_path": "./log_eval/",
        "use_posterior": true
    }
}
```

- SA-Heart dataset:

```json
{
    "stage": "train",
    "train": {
        "data_type": "real_data",
        "real_data": {
            "train_data_path": "./dataset/SAHeart/SAHeart.csv",
            "test_data_path": null,
            "val_data_path": null,
            "split_data_again": true,
            "features": ["sbp", "tobacco", "ldl", "adiposity", "famhist", "typea", "obesity", "alcohol", "age"],
            "label_feature": "chd",
            "categorical_features": ["famhist"]
        },
        "toy_data": {
            "dataset_name": "circles",
            "dataset_size": 2000,
            "n_classes": 2
        },
        "log_path": "./log_train/",
        "batch_size": 800,
        "hidden_shape": [200, 200],
        "layers": 12,
        "base_lr": 1e-5,
        "end_lr": 1e-6,
        "max_epochs": 5e3,
        "delta_stop_in_early_stopping": 1000,
        "frequency_validation": 100,
        "frequency_plot": 1000,
        "plot_data": false
    },
    "eval": {
        "log_path": "./log_eval/",
        "use_posterior": true
    }
}
```

# Config of GMM classifier for the datasets:

- Toy data:

```json
{
    "stage": "train",
    "train": {
        "data_type": "toy_data",
        "real_data": {
            "train_data_path": "./dataset/SAHeart/SAHeart.csv",
            "test_data_path": null,
            "val_data_path": null,
            "split_data_again": true,
            "features": ["sbp", "tobacco", "ldl", "adiposity", "famhist", "typea", "obesity", "alcohol", "age"],
            "label_feature": "chd",
            "categorical_features": ["famhist"]
        },
        "toy_data": {
            "dataset_name": "circles",
            "dataset_size": 2000,
            "n_classes": 2
        },
        "log_path": "./log_train/",
        "batch_size": 800,
        "n_components": 5, 
        "plot_data": false
    },
    "eval": {
        "log_path": "./log_eval/",
        "use_posterior": true
    }
}
```

- SA-Heart dataset:

```json
{
    "stage": "train",
    "train": {
        "data_type": "real_data",
        "real_data": {
            "train_data_path": "./dataset/SAHeart/SAHeart.csv",
            "test_data_path": null,
            "val_data_path": null,
            "split_data_again": true,
            "features": ["sbp", "tobacco", "ldl", "adiposity", "famhist", "typea", "obesity", "alcohol", "age"],
            "label_feature": "chd",
            "categorical_features": ["famhist"]
        },
        "toy_data": {
            "dataset_name": "circles",
            "dataset_size": 2000,
            "n_classes": 2
        },
        "log_path": "./log_train/",
        "batch_size": 800,
        "n_components": 3, 
        "plot_data": false
    },
    "eval": {
        "log_path": "./log_eval/",
        "use_posterior": true
    }
}
```
