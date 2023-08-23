
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

import pandas as pd
from darts import TimeSeries

# df = pd.read_csv("magnetism.csv")
# float64_cols = list(df.select_dtypes(include='float64'))
# df[float64_cols] = df[float64_cols].astype('float32')

import pandas as pd
import pickle

with open('timeseries','rb') as f: time_series_loaded = pickle.load(f)
time_series_list = time_series_loaded

# print("hello")
    
# fix python path if working locally
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.utils.timeseries_generation import (
    gaussian_timeseries,
    linear_timeseries,
    sine_timeseries,
)
from darts.models import (
    RNNModel,
    TCNModel,
    TransformerModel,
    NBEATSModel,
    BlockRNNModel,
    VARIMA,
)
from darts.metrics import mape, smape, mae
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.datasets import AirPassengersDataset, MonthlyMilkDataset, ElectricityDataset

import logging

# logging.disable(logging.CRITICAL)

import warnings

# warnings.filterwarnings("ignore")

# %matplotlib inline

# for reproducibility
torch.manual_seed(1)
np.random.seed(1)

time_series_list_scaled = []

for time_series in time_series_list:
    scaler = Scaler()
    time_series_list_scaled.append(scaler.fit_transform(time_series))
    
# print(time_series_list_scaled[0].columns)
from darts import TimeSeries
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
# print(time_series_list_scaled[0].columns)

from darts import TimeSeries

covariates = []
for i in range(len(time_series_list)):
    ts_df = (time_series_list[i]).pd_dataframe()
    columns_to_keep = ['Set B field (T)', 'Elapsed time (minutes)', 'Ramp speed (T/min)', 'Inner Lower Temperature', 
                       'Outer Lower Temperature', 'Inner Upper Temperature',
                       'Outer Upper Temperature', '1st Stage Temperature',
                       'Shield Temperature', '2nd Stage Temperature']
    covariate_ts_df = ts_df[columns_to_keep]
    covariates.append(TimeSeries.from_dataframe(covariate_ts_df))

updated_covariates = []
for i in range(len(covariates)):
    df = covariates[i].pd_dataframe()
    # for time_series in time_series_list_scaled:
    temp = time_series_list_scaled[i].pd_dataframe()
    for col in temp.columns:
        df[col] = temp[col]
    updated_covariates.append(TimeSeries.from_dataframe(df))
    
# print(updated_covariates[0].columns)
covariates = updated_covariates

from darts import TimeSeries

# ts_df = ts.pd_dataframe()
time_final = []
for i in range(len(time_series_list)):
    ts_df = (time_series_list[i]).pd_dataframe()
    columns_to_keep = ['B field (T)', 'Voltage (V)', ]
    covariate_ts_df = ts_df[columns_to_keep]
    time_final.append(TimeSeries.from_dataframe(covariate_ts_df))
    
# print(len(time_final[0].columns))
# time_final = time_series_list

# load data
train, val = [], []
VAL_LEN = 200
temp1, temp2 = [], []

for i in range(len(time_final)):
    series = time_final[i]
    train.append(series[:-VAL_LEN])
    val.append(series[-VAL_LEN:])
    temp1.append(covariates[i][-VAL_LEN:])
    temp2.append(covariates[i][:-VAL_LEN])

# for time_series in time_final:
#     series = time_series.astype(np.float32)
#     train.append(series[:-VAL_LEN])
#     val.append(series[-VAL_LEN:])
    
    
# for time_series in covariates:
#     series = time_series.astype(np.float32)
#     temp.append(series[:-VAL_LEN])
val_past_covariates_past = temp1
covariates = temp2

# split in train / validation (note: in practice we would also need a test set)
# train, val = 

train = train[:2]
val = val[:2]
covariates = covariates[:2]
val_past_covariates_past = val_past_covariates_past[:2]

import os
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
from pytorch_lightning.strategies import SingleDeviceStrategy

# scale
# scaler = Scaler(MaxAbsScaler())
# train = scaler.fit_transform(train)
# val = scaler.transform(val)

# define objective function
def objective(trial):
    # select input and output chunk lengths
    in_len = trial.suggest_int("in_len", 15, 120)
    out_len = trial.suggest_int("out_len", 1, in_len-1)
    

    # Other hyperparameters
    d_model_length = trial.suggest_int("d_model_length", 54, 72)
    num_encoder_layers_temp = trial.suggest_int("num_encoder_layers_temp", 1, 5)
    num_decoder_layers_temp = trial.suggest_int("num_decoder_layers_temp", 1, 5)
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
    # callbacks = [early_stopper]
   
    
    num_workers = 4
        
    pl_trainer_kwargs = {
        "accelerator": "gpu",
        "devices": [int(os.environ["CUDA_VISIBLE_DEVICES"])],
        #"strategy": SingleDeviceStrategy(device=torch.cuda.current_device()),
        #"strategy": "single_device",
        #"strategy": "ddp",
        "callbacks": callbacks,
    }

    encoders = None #look into add encoders

    # reproducibility
    torch.manual_seed(42)
    # kernel_size', 'num_filters', 'weight_norm', 'dilation_base']`
    # build the TCN model
    model = TransformerModel(
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        batch_size=32,
        n_epochs=4,
        nr_epochs_val_period=1,
        num_encoder_layers = num_encoder_layers_temp,
        num_decoder_layers = num_decoder_layers_temp,
        # d_model = d_model_length,
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
    # model_val_set = []
    # for time_series in time_final:
    #     series = time_series.astype(np.float32)
    #     model_val_set.append(series[-(VAL_LEN + in_len) :])

    # train the model
    model.fit(
        series=train,
        val_series=val,
        num_loader_workers=1,
        past_covariates = covariates,
        val_past_covariates=val_past_covariates_past
    )
    
#     model.fit(
#         series=train,
#         past_covariates=covariates,
#     );

    # reload best model over course of training
    model = TransformerModel.load_from_checkpoint("tcn_model")
    
    # Evaluate how good it is on the validation set, using sMAPE
    preds = model.predict(series=train[0], n=VAL_LEN)
    smapes = smape(val[0], preds, n_jobs=-1, verbose=True)
    smape_val = np.mean(smapes)
    # if smape_val== np.nan:
    #     print("nan")
    return smape_val if smape_val != np.nan else float("inf")


# for convenience, print some optimization trials information
def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


# optimize hyperparameters by minimizing the sMAPE on the validation set
# def run_trial():
#     study = optuna.create_study(direction="minimize")
#     study.optimize(objective, n_trials=3, callbacks=[print_callback])
        
# if __name__ == "__main__":
#     run_trial()
#     # objective(Trial())



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
            client.submit(study.optimize, objective, n_trials=100, pure=False, callbacks = [print_callback]) for _ in range(10)
        ]
        wait(futures)
        # print(f"Best params: {study.best_params}")
    with open(DEBUG_FILE, 'a+') as fd: print("trial complete\n", file=fd)

def slurm_trial(name=None):
    from dask_jobqueue import SLURMCluster
    # https://jobqueue.dask.org/en/latest/configuration-setup.html
    cluster = SLURMCluster(
        cores=4,
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
    slurm_trial(name=f"transformer [{random.randint(1, 999):03d}]")
