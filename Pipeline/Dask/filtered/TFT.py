
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

STORAGE = "sqlite:///optuna.sqlite"
DEBUG_FILE = "debug_optuna.txt"
# VAL_LEN = 300

import os
import socket
import time
import random
from pathlib import Path

import numpy as np
import torch
import pickle
from darts import TimeSeries
from darts import concatenate

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

from darts.dataprocessing.transformers import Scaler
from darts.models import TransformerModel
from darts.models import TCNModel
from darts.models import TFTModel
from darts.metrics import smape
from darts.utils.likelihood_models import GaussianLikelihood
from darts import TimeSeries

from dask.distributed import Client
from dask.distributed import wait

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.strategies import SingleDeviceStrategy

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

VAL_LEN = 500

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def prepare_data(trim=None):
    time_series_loaded = None
    with open('timeseries_4000_filtered','rb') as f: time_series_loaded = pickle.load(f)
    time_series_list = time_series_loaded
    
    time_series_list_scaled = []

    for time_series in time_series_list:
        scaler = Scaler()
        time_series_list_scaled.append(scaler.fit_transform(time_series))
        
    # print(time_series_list_scaled[0].columns)
    temp = []
    # Assuming you have a Darts TimeSeries object named 'ts'

    # Convert the TimeSeries to a pandas DataFrame
    for ts in time_series_list_scaled:
        df = ts.pd_dataframe()

        # Rename the columns in the pandas DataFrame
        new_column_names = ['Normalized Elapsed time (minutes)', 'Normalized Set B field (T)', 'Normalized B field (T)',
           'Normalized Ramp speed (T/min)', 'Normalized Voltage (V)', 'Normalized Inner Lower Temperature',
           'Normalized Outer Lower Temperature', 'Normalized Inner Upper Temperature',
           'Normalized Outer Upper Temperature', 'Normalized 1st Stage Temperature',
           'Normalized Shield Temperature', 'Normalized 2nd Stage Temperature']  # Replace with the new column names you want
        df.columns = new_column_names

        # Convert the modified pandas DataFrame back to a Darts TimeSeries
        temp.append(TimeSeries.from_dataframe(df))
    time_series_list_scaled = temp
    
    # from darts import TimeSeries

    covariates = []
    for i in range(len(time_series_list)):
        ts_df = (time_series_list[i]).pd_dataframe()
        columns_to_keep = ['Set B field (T)']
        covariate_ts_df = ts_df[columns_to_keep]
        covariates.append(TimeSeries.from_dataframe(covariate_ts_df))

    temp1 = []
    from darts import concatenate
    for df in time_series_list_scaled:
        storage = []
        for col in df.columns:
            storage.append(df[col])
        temp1.append(concatenate([timeSeries for timeSeries in storage], axis=1))

    past_covariates = temp1
    
    time_final = []
    for i in range(len(time_series_list)):
        ts_df = (time_series_list[i]).pd_dataframe()
        columns_to_keep = ['B field (T)', 'Voltage (V)', 'Ramp speed (T/min)', 'Inner Lower Temperature', 
                'Outer Lower Temperature', 'Inner Upper Temperature',
                'Outer Upper Temperature', '1st Stage Temperature',
                'Shield Temperature', '2nd Stage Temperature']
        covariate_ts_df = ts_df[columns_to_keep]
        time_final.append(TimeSeries.from_dataframe(covariate_ts_df))
    
    # load data
    train, val = [], []
    # VAL_LEN = 300
    temp1, temp2 = [], []


    for i in range(len(time_final)):
        series = time_final[i]
        train.append(series)
        # val.append(series[-VAL_LEN:])
        temp1.append(covariates[i])
        temp2.append(covariates[i])

    covariates = temp2

    val = train[60:]
    train = train[:60]

    future_covariates = covariates[60:]
    covariates = covariates[:60]

    val_past = past_covariates[60:]
    past_covariates = past_covariates[:60]
    
    return val, train, future_covariates, covariates, val_past, past_covariates






# define objective function
def objective(trial):
    epochs = 100
    verbose = False
    model_name = f"{trial.study.study_name}/T{trial.number:04d}"
    (Path("darts_logs")/model_name).mkdir(parents=True, exist_ok=True)

    # select input and output chunk lengths
    # select input and output chunk lengths


    activation_temp =  trial.suggest_categorical("activation_temp", ["GatedResidualNetwork", "GLU", "Bilinear", "ReGLU", "GEGLU", "SwiGLU", "ReLU", "GELU"])
    
    kernel_size = trial.suggest_int("kernel_size", 4, 7)
    num_filters = trial.suggest_int("num_filters", 2, 4)
    weight_norm = trial.suggest_categorical("weight_norm", [False, True])
    dilation_base = trial.suggest_int("dilation_base", 3, 6)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    lr = trial.suggest_float("lr", 5e-6, 1e-3, log=True)
    hidden_size_temp = trial.suggest_int("hidden_size", 12, 20)
    
    in_len = trial.suggest_int("in_len", 20, 140)
    out_len = trial.suggest_int("out_len", 5, in_len-1)
    
    kernel_size = trial.suggest_int("kernel_size", 4, 7)
    num_filters = trial.suggest_int("num_filters", 2, 4)
    weight_norm = trial.suggest_categorical("weight_norm", [False, True])
    dilation_base = trial.suggest_int("dilation_base", 3, 6)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    lr = trial.suggest_float("lr", 5e-6, 1e-3, log=True)
    # include_year = trial.suggest_categorical("year", [False, True])

    # throughout training we'll monitor the validation loss for both pruning and early stopping
    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=3, verbose=True)
    callbacks = [pruner, early_stopper]
    # callbacks = [early_stopper]
   
    pl_trainer_kwargs = {
        "accelerator": "gpu",
        #"devices": [int(os.environ["CUDA_VISIBLE_DEVICES"])],
        #"strategy": SingleDeviceStrategy(device=torch.cuda.current_device()),
        #"strategy": "single_device",
        #"strategy": "ddp",
        "callbacks": callbacks,
    }

    encoders = None #look into add encoders

    set_seed(1)
    
    # kernel_size', 'num_filters', 'weight_norm', 'dilation_base']`
    # build the TCN model
    model = TFTModel(
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        batch_size=32,
        n_epochs=epochs,
        nr_epochs_val_period=1,
        feed_forward = activation_temp,
        hidden_size = hidden_size_temp,
        dropout=dropout,
        optimizer_kwargs={"lr": lr},
        add_encoders=encoders,
        likelihood=GaussianLikelihood(),
        pl_trainer_kwargs=pl_trainer_kwargs,
        model_name=model_name,
        force_reset=True,
        save_checkpoints=True,
    )
    
    
    # train, val, covariates, val_past_covariates_past = prepare_data(trim=2)
    val, train, future_covariates, covariates, val_past, past_covariates = prepare_data()

       # train the model
    model.fit(
        series=train,
        val_series= val,
        # num_loader_workers=1,
        future_covariates = covariates,
        val_future_covariates= future_covariates,
        past_covariates = past_covariates,
        val_past_covariates = val_past
    )
    

    # reload best model over course of training
    model = TFTModel.load_from_checkpoint(model_name)
    
    # Evaluate how good it is on the validation set, using sMAPE
    # preds = model.predict(series=train[0], n=VAL_LEN)
    smapes = 0
    for i in range(len(train)):
        preds = model.predict(
            series=train[i][:2000],
            future_covariates=covariates[i],
            past_covariates = past_covariates[i],
            n=VAL_LEN,
        )
        smapes += smape(preds, train[i][2000:2000+VAL_LEN])
    smapes = smapes/len(train)
    smape_val = np.mean(smapes)
    # if smape_val== np.nan:
    #     print("nan")
    return smape_val if smape_val != np.nan else float("inf")


# for convenience, print some optimization trials information
def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


def run_trial(storage=STORAGE, cluster=None, name=None, trials=4):
    # Create a unique study name
    if name is None: name = "study"
    name = f"{name}-{random.randint(1, 9999):04d}"
    print(f"Running study {name} with {trials} trials.")
    with open(DEBUG_FILE, 'w') as fd: ...  # create an empty file
    with Client(address=cluster) as client:
        print(f"Dask dashboard is available at {client.dashboard_link}")
        storage = optuna.integration.dask.DaskStorage(storage=storage)
        study = optuna.create_study(
            storage=storage, direction="minimize", study_name=name,
        )
        
        futures = [
            client.submit(study.optimize, objective, n_trials=5, pure=False, callbacks = [print_callback]) for _ in range(trials)
        ]
        wait(futures)
    with open(DEBUG_FILE, 'a+') as fd: 
        print("trial complete\n", file=fd)
        print(f"Best params: {study.best_params}")

def slurm_trial(jobs=None, **kwargs):
    from dask_jobqueue import SLURMCluster
    # https://jobqueue.dask.org/en/latest/configuration-setup.html
    prefix = "slurm"
    job_args = [
        '--gres=gpu:1',
        #f'-e {prefix}-%j.err',
        #f'-o {prefix}-%j.out',
    ]
    cluster = SLURMCluster(
        cores=1,
        memory='4GB',
        processes=1,
        # queue='idle',
        walltime='96:00:00',
        local_directory='/tmp',
        job_extra_directives=job_args,
        queue="cuda+",
    )
    # 27 gpus available on the cluster as of this writing:
    #   1x980, 14*1080, 12x3080
    if jobs is None: jobs = 12+14+1
    cluster.adapt(maximum_jobs=jobs)
    print(cluster.job_script())
    run_trial(storage=STORAGE, cluster=cluster, **kwargs)

if __name__ == "__main__":
    #run_trial(storage=STORAGE)
    slurm_trial(name="transformer", jobs=27, trials=20)

