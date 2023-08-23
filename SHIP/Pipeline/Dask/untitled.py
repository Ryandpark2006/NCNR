
#%conda install optuna dask dask-jobqueue

"""
Optuna example that runs optimization trials in parallel using Dask.

In this example, we perform hyperparameter optimization on a
RandomForestClassifier which is trained using the handwritten digits
dataset. Trials are run in parallel on a Dask cluster using Optuna's
DaskStorage integration.

To run this example:

    $ python dask_simple.py
"""

STORAGE = "sqlite:///dasktest.sqlite"
DEBUG_FILE = "debug_dask.txt"

import socket
import time
import random

import optuna

from dask.distributed import Client
from dask.distributed import wait
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import numpy as np
import optuna
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler

from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.metrics import smape
from darts.models import TCNModel
from darts.utils.likelihood_models import GaussianLikelihood

# scale
# scaler = Scaler(MaxAbsScaler())
# train = scaler.fit_transform(train)
# val = scaler.transform(val)

# define objective function
def objective(trial):
    # select input and output chunk lengths
    in_len = trial.suggest_int("in_len", 12, 36)
    out_len = trial.suggest_int("out_len", 1, in_len-1)

    # Other hyperparameters
    kernel_size = trial.suggest_int("kernel_size", 2, 5)
    num_filters = trial.suggest_int("num_filters", 1, 5)
    weight_norm = trial.suggest_categorical("weight_norm", [False, True])
    dilation_base = trial.suggest_int("dilation_base", 2, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    # include_year = trial.suggest_categorical("year", [False, True])

    # throughout training we'll monitor the validation loss for both pruning and early stopping
    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=3, verbose=True)
    callbacks = [pruner, early_stopper]
    
    num_workers = 4
        
    pl_trainer_kwargs = {
        "accelerator": "gpu",
        "devices": [0],
        "callbacks": callbacks,
    }

    # optionally also add the (scaled) year value as a past covariate
    # if include_year:
    #     encoders = {"datetime_attribute": {"past": ["year"]},
    #                 "transformer": Scaler()}
    # else:
    encoders = None #look into add encoders

    # reproducibility
    torch.manual_seed(42)
    
    # build the TCN model
    model = TCNModel(
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        batch_size=32,
        n_epochs=4,
        nr_epochs_val_period=1,
        kernel_size=kernel_size,
        num_filters=num_filters,
        weight_norm=weight_norm,
        dilation_base=dilation_base,
        dropout=dropout,
        optimizer_kwargs={"lr": lr},
        add_encoders=encoders,
        likelihood=GaussianLikelihood(),
        pl_trainer_kwargs=pl_trainer_kwargs,
        model_name="tcn_model_1",
        force_reset=True,
        save_checkpoints=True,
    )
    
    
    # when validating during training, we can use a slightly longer validation
    # set which also contains the first input_chunk_length time steps
    # model_val_set = scaler.transform(series[-(VAL_LEN + in_len) :])
    model_val_set = []
    for time_series in time_final:
        series = time_series.astype(np.float32)
        model_val_set.append(series[-(VAL_LEN + in_len) :])

    # train the model
    model.fit(
        series=train,
        val_series=model_val_set,
        num_loader_workers=num_workers,
        # past_covariates = covariates,
    )

    # reload best model over course of training
    model = TCNModel.load_from_checkpoint("tcn_model")
    
    # Evaluate how good it is on the validation set, using sMAPE
    preds = model.predict(series=train, n=VAL_LEN)
    smapes = smape(val, preds, n_jobs=-1, verbose=True)
    smape_val = np.mean(smapes)
    return smape_val if smape_val != np.nan else float("inf")


# for convenience, print some optimization trials information
def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


# def objective(trial):
#     #raise Hell
#     X, y = load_digits(return_X_y=True)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

#     max_depth = trial.suggest_int("max_depth", 2, 10)
#     n_estimators = trial.suggest_int("n_estimators", 1, 100)

#     t0 = time.time()
#     with open(DEBUG_FILE, 'a+') as fd: print(f"{socket.gethostname()} [{max_depth},{n_estimators}] start", file=fd)

#     clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     score = accuracy_score(y_test, y_pred)
#     time.sleep(5) # Just to slow things down and verify parallelism
#     with open(DEBUG_FILE, 'a+') as fd: print(f"{socket.gethostname()} [{max_depth},{n_estimators}] stop t={time.time()-t0:.3f}s", file=fd)
#     return score


def run_trial(storage=STORAGE, cluster=None, name=None):
    with open(DEBUG_FILE, 'w') as fd: ...  # create an empty file
    with Client(address=cluster) as client:
        print(f"Dask dashboard is available at {client.dashboard_link}")
        storage = optuna.integration.dask.DaskStorage(storage=storage)
        study = optuna.create_study(
            storage=storage, direction="minimize", study_name=name,
        )
        
    #         study = optuna.create_study(direction="minimize")
    # study.optimize(objective, n_trials=100, callbacks=[print_callback])
        # Submit 10 different optimization tasks, where each task runs 7 optimization trials
        # for a total of 70 trials in all
        futures = [
            client.submit(study.optimize, objective, n_trials=7, pure=False, callbacks = [print_callback]) for _ in range(10)
        ]
        wait(futures)
        print(f"Best params: {study.best_params}")
    with open(DEBUG_FILE, 'a+') as fd: print("trial complete\n", file=fd)

def slurm_trial(name=None):
    from dask_jobqueue import SLURMCluster
    # https://jobqueue.dask.org/en/latest/configuration-setup.html
    cluster = SLURMCluster(
        cores=1,
        memory='4GB',
        processes=1,
        queue='idle',
        local_directory='/tmp',
        job_extra_directives=['--gres=gpu:1'])
    cluster.adapt(maximum_jobs=4)
    print(cluster.job_script())
    run_trial(storage=STORAGE, cluster=cluster, name=name)

if __name__ == "__main__":
    #run_trial(storage=STORAGE)
    slurm_trial(name=f"slurm trial 10 [{random.randint(1, 999):03d}]")