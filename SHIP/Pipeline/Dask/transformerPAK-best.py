
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
VAL_LEN = 300

import os
import socket
import time
import random
from pathlib import Path

import numpy as np
import torch


from darts.dataprocessing.transformers import Scaler
from darts.models import TransformerModel
from darts.metrics import smape
from darts.utils.likelihood_models import GaussianLikelihood
from darts import TimeSeries

from dask.distributed import Client
from dask.distributed import wait

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.strategies import SingleDeviceStrategy

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def prepare_data(trim=None):
    import pickle
    with open('timeseries', 'rb') as f:
        time_series_list = pickle.load(f)

    time_series_list_scaled = []
    for time_series in time_series_list:
        scaler = Scaler()
        time_series_list_scaled.append(scaler.fit_transform(time_series))
    
    # Convert the TimeSeries to a pandas DataFrame
    temp = []
    for ts in time_series_list_scaled:
        df = ts.pd_dataframe()

        # Rename the columns in the pandas DataFrame
        new_column_names = [
            'Normalized Elapsed time (minutes)', 'Normalized Set B field (T)', 'Normalized B field (T)',
            'Normalized Ramp speed (T/min)', 'Normalized Voltage (V)', 'Normalized Inner Lower Temperature',
            'Normalized Outer Lower Temperature', 'Normalized Inner Upper Temperature',
            'Normalized Outer Upper Temperature', 'Normalized 1st Stage Temperature',
            'Normalized Shield Temperature', 'Normalized 2nd Stage Temperature',
        ]  # Replace with the new column names you want
        df.columns = new_column_names

        # Convert the modified pandas DataFrame back to a Darts TimeSeries
        temp.append(TimeSeries.from_dataframe(df))
    time_series_list_scaled = temp


    covariates = []
    for i in range(len(time_series_list)):
        ts_df = (time_series_list[i]).pd_dataframe()
        columns_to_keep = [
            'Set B field (T)', 'Elapsed time (minutes)', 'Ramp speed (T/min)', 'Inner Lower Temperature', 
            'Outer Lower Temperature', 'Inner Upper Temperature',
            'Outer Upper Temperature', '1st Stage Temperature',
            'Shield Temperature', '2nd Stage Temperature',
        ]
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
    covariates = updated_covariates

    # ts_df = ts.pd_dataframe()
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
    temp1, temp2 = [], []
    for i in range(len(time_final)):
        series = time_final[i]
        train.append(series[:-VAL_LEN])
        val.append(series[-VAL_LEN:])
        temp1.append(covariates[i][-VAL_LEN:])
        temp2.append(covariates[i][:-VAL_LEN])

    val_past_covariates_past = temp1
    covariates = temp2

    if trim:
        train = train[:trim]
        val = val[:trim]
        covariates = covariates[:trim]
        val_past_covariates_past = val_past_covariates_past[:trim]

    return train, val, covariates, val_past_covariates_past



# define objective function
def objective(trial):
    epochs = 200
    verbose = False
    model_name = f"{trial.study.study_name}/T{trial.number:04d}"
    (Path("darts_logs")/model_name).mkdir(parents=True, exist_ok=True)

    # select input and output chunk lengths
    in_len = trial.suggest_int("in_len", 60, 150)
    out_len = trial.suggest_int("out_len", 30, in_len-1)
    

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
    model = TransformerModel(
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        batch_size=64,
        n_epochs=epochs,
        nr_epochs_val_period=1,
        num_encoder_layers = num_encoder_layers_temp,
        num_decoder_layers = num_decoder_layers_temp,
        # d_model = d_model_length,
        dropout=dropout,
        optimizer_kwargs={"lr": lr},
        add_encoders=encoders,
        likelihood=GaussianLikelihood(),
        pl_trainer_kwargs=pl_trainer_kwargs,
        model_name=model_name,
        force_reset=True,
        save_checkpoints=True,
    )
    
    
    train, val, covariates, val_past_covariates_past = prepare_data(trim=2)
    # train, val, covariates, val_past_covariates_past = prepare_data()

    # train the model
    model.fit(
        series=train,
        val_series=val,
        #num_loader_workers=0,
        #past_covariates = covariates,
        #val_past_covariates=val_past_covariates_past,
        # verbose=verbose,
    )
    
    # reload best model over course of training
    model = TransformerModel.load_from_checkpoint(model_name)
    
    # Evaluate how good it is on the validation set, using sMAPE
    smapes = 0
    for i in range(len(train)):
        preds = model.predict(
            series=train[i],
            # past_covariates=covariates[i],
            n=VAL_LEN,
            # verbose=verbose,
        )
        smapes += smape(preds, val[i])
    smapes = smapes/len(train)
    smape_val = np.mean(smapes)
    # Note: (x == nan) is always false, even if x is nan, so use np.isnan(x) instead
    return smape_val if not np.isnan(smape_val) else np.inf


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
            client.submit(study.optimize, objective, n_trials=100, pure=False, callbacks = [print_callback]) for _ in range(trials)
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
        queue='idle',
        walltime='96:00:00',
        local_directory='/tmp',
        job_extra_directives=job_args,
    )
    # 27 gpus available on the cluster as of this writing:
    #   1x980, 14*1080, 12x3080
    if jobs is None: jobs = 12+14+1
    cluster.adapt(maximum_jobs=jobs)
    print(cluster.job_script())
    run_trial(storage=STORAGE, cluster=cluster, **kwargs)

if __name__ == "__main__":
    #run_trial(storage=STORAGE)
    slurm_trial(name="transformer", jobs=27, trials=5)

