#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: run_regressor_input_search.py
Description:
    This script demonstrates how to train and evaluate a Multi-Layer Perceptron (MLP)
    and a Sparse Autoencoder (SAE) on meteorological data from our observational dataset.
    It sets deterministic operations for reproducibility, splits data into 
    train/test sets, and performs correlation analyses between learned 
    feature representations and the original predictors to find interesting cases.
    This version is for the snowfall rate regressor example problem.

Usage:
    python run_regressor_input_search.py

Author: Fraser King (kingfr@umich.edu)
Date: 2025-01-29
"""


### Set random seeding and deterministic globals ENVs
import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import random
import numpy as np
import tensorflow as tf
tf.config.experimental.enable_op_determinism()
print(tf.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


### Remaining imports

import gc
import math
import warnings
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from scipy.stats import pearsonr
from keras import regularizers

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.rcParams.update({'font.size': 26})


### GLOBAL VARIABLE DEFINITIONS

palette = ['black',
            '#3cb44b', '#4363d8','#bfef45','#42d4f4','#911eb4',
            '#ffe119','#e6194B', '#f032e6','#f58231','gray']
columns_to_exclude = ['UMAP', 'PCA', 'phase_thresh', 'phase_umap', 'phase_pca', 'phase_temp', 'phase_pip_rate', 'Rr', 'Nt', 'Dm', 'Near Surface Reflectivity']#, 'n0', 'lambda', 'Rho', 'Nt', 'Fs', 'Rr', 'Dm']
response = 'Sr'
N_INPUTS_TO_SEARCH = 5
datetime = 'Date'
data_path = '../data/MQT_regression2.csv'
STATE = 42

# RF CONSTANTS
N_TREES = 100
MAX_DEPTH = 6

# MLP CONSTANTS
N_HIDDEN = 6
ACTIVATION = 'relu'
LR = 0.0005
LOSS_FUNC = 'mse'
N_EPOCHS = 32
BATCH_SIZE = 32
L1 = 0.01
OPTIMIZER = 'ADAM'

### Reset Keras/TensorFlow session to clear memory
def reset_keras_backend():
    tf.keras.backend.clear_session()
    gc.collect()

### Main runloop
def run_all(predictors):
    print()
    predictors = ['reflectivity_3', 'reflectivity_29', 'dv_3', 'dv_29', 'n0']
    def load_and_scale(path):
        def print_var_ranges():
            print("Variable Statistics:\n")
            for col in predictors:
                if col in df.columns:
                    print(f"{col}")
                    print(f"Max: {df[col].max()}")
                    print(f"Min: {df[col].min()}")
                    print(f"Mean: {df[col].mean()}")
                    print(f"Median: {df[col].median()}")
                    print("-" * 40)

        df = pd.read_csv(path)
        print("Data loaded!\n")

        # print_var_ranges()
        df['n0'] = np.log10(df['n0'])
        df['Nt'] = np.log10(df['Nt'])
        df['lambda'] = np.log10(df['lambda'])
        df['Rho'] = np.log10(df['Rho'])
        df['Dm'] = np.log10(df['Dm'])

        print(f'\n{len(df)} total rows')
        print(f'{len(df.columns)-1} input variables')
        print(f'{len(df)*(len(df.columns)-1)} value combinations')
        return df


    df = load_and_scale(data_path)

    def get_X_Y_Z(df, predictors, response):
        df_loc = df.copy()

        date_df = df_loc[['Date']]
        df_loc = df_loc.drop(columns=['Date'])

        X = df_loc[predictors]
        y = df_loc[response]

        data = pd.concat([X, y], axis=1)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_cleaned = data.dropna()
        X_cleaned = data_cleaned[predictors]
        y_cleaned = data_cleaned[response]


        # Split the data into X_data (60%) and temp_data (40%)
        X_data, temp_data, y_data, temp_labels, data_indices, temp_indices = train_test_split(
            X_cleaned, y_cleaned, data_cleaned.index, test_size=0.4, shuffle=False, random_state=STATE
        )

        # Split temp_data into Y_data (20%) and Z_data (20%)
        Y_data, Z_data, y_Y, y_Z, indices_Y, indices_Z = train_test_split(
            temp_data, temp_labels, temp_indices, test_size=0.5, shuffle=False, random_state=STATE
        )

        X_data = X_data.to_frame() if isinstance(X_data, pd.Series) else X_data
        Y_data = Y_data.to_frame() if isinstance(Y_data, pd.Series) else Y_data
        Z_data = Z_data.to_frame() if isinstance(Z_data, pd.Series) else Z_data

        # Scaling
        scaler = StandardScaler()
        X_data_scaled = scaler.fit_transform(X_data)
        Y_data_scaled = scaler.transform(Y_data)
        Z_data_scaled = scaler.transform(Z_data)

        scaling_info = {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist()
        }

        return (X_data_scaled, Y_data_scaled, Z_data_scaled,
                y_data, y_Y, y_Z,
                data_indices, indices_Y, indices_Z,
                scaler, scaling_info)

    ### PERFORM DATA SPLIT
    mlp_train_X, sae_train_X, sae_test_X, mlp_train_y, sae_train_y, sae_test_y, data_indices, indices_Y, indices_Z, scaler, scaling_info = get_X_Y_Z(df, predictors, response)

    print("\nTraining shapes:")
    print("X:", mlp_train_X.shape)
    print("Y:", sae_train_X.shape)
    print("Y:", sae_test_X.shape)

    print("\nTesting shapes:")
    print("X:", mlp_train_y.shape)
    print("Y:", sae_train_y.shape)
    print("Z:", sae_test_y.shape)


    # Split mlp_train_X and mlp_train_y into training and validation sets for MLP
    mlp_X_train, mlp_X_val, mlp_y_train, mlp_y_val = train_test_split(
        mlp_train_X, mlp_train_y, test_size=0.2, random_state=STATE, shuffle=False
    )

    print(mlp_X_train)

    model = Sequential()
    input_shape = mlp_train_X.shape[1]
    model.add(Dense(N_HIDDEN, activation=ACTIVATION, input_shape=(input_shape,)))
    # model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(optimizer=optimizer, 
                loss=LOSS_FUNC, 
                metrics=['mean_absolute_error'])

    history = model.fit(mlp_X_train, mlp_y_train, 
                        validation_data=(mlp_X_val, mlp_y_val), 
                        epochs=N_EPOCHS, 
                        batch_size=BATCH_SIZE)

    test_loss_mlp, test_mse_mlp = model.evaluate(sae_test_X, sae_test_y)

    y_test_pred = model.predict(sae_test_X).flatten()
    y_test_true = sae_test_y.to_numpy().flatten()

    mse_test = mean_squared_error(y_test_true, y_test_pred)
    r2_test = r2_score(y_test_true, y_test_pred)
    corr_test, _ = pearsonr(y_test_true, y_test_pred)

    print("Test Set Performance:")
    print(f"Mean Squared Error (MSE): {mse_test:.4f}")
    print(f"Correlation Coefficient: {corr_test:.4f}")
    print(f"R^2 Score: {r2_test:.4f}")

    model.summary()

    reset_keras_backend()
    gc.collect()

    # Create a model to output activations from the hidden layer
    activation_model = Model(inputs=model.layers[0].input, outputs=model.layers[0].output)

    sae_train_activations = activation_model.predict(sae_train_X)
    input_dim = sae_train_activations.shape[1]
    encoding_dim = input_dim * 2  # More neurons than the input activations

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu',
                    activity_regularizer=regularizers.l1(L1))(input_layer)
    decoded = Dense(input_dim, activation='linear')(encoded)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    sae_model = Model(inputs=input_layer, outputs=decoded)
    sae_model.compile(optimizer=optimizer, loss='mse')

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                              patience=8)
    sae_history = sae_model.fit(sae_train_activations, sae_train_activations,
                                epochs=128,
                                batch_size=8,
                                callbacks=[callback],
                                shuffle=False)
    sae_test_activations = activation_model.predict(sae_test_X)

    encoder_model = Model(inputs=sae_model.input, outputs=sae_model.layers[1].output)

    def plot_mlp_and_sae_correlations(model, activation_model, encoder_model, sae_test_X, predictors, r2):
        # 1. Compute MLP Hidden Activations
        hidden_layer = model.layers[0]
        hidden_activation_model = Model(inputs=model.layers[0].input, outputs=model.layers[0].output)
        mlp_hidden_activations = hidden_activation_model.predict(sae_test_X)

        n_hidden_units = mlp_hidden_activations.shape[1]
        n_predictors = len(predictors)
        mlp_corr_matrix = np.zeros((n_hidden_units, n_predictors))
        
        for i in range(n_hidden_units):
            for j in range(n_predictors):
                r, _ = pearsonr(mlp_hidden_activations[:, i], sae_test_X[:, j])
                mlp_corr_matrix[i, j] = r

        mlp_unit_names = [f"{i}" for i in range(n_hidden_units)]
        mlp_corr_df = pd.DataFrame(mlp_corr_matrix, index=mlp_unit_names, columns=predictors)

        # 2. Compute SAE Encoded Activations
        sae_encoded_activations = encoder_model.predict(mlp_hidden_activations)
        n_sae_units = sae_encoded_activations.shape[1]
        sae_corr_matrix = np.zeros((n_sae_units, n_predictors))

        for i in range(n_sae_units):
            for j in range(n_predictors):
                r, _ = pearsonr(sae_encoded_activations[:, i], sae_test_X[:, j])
                sae_corr_matrix[i, j] = r

        sae_unit_names = [f"{i}" for i in range(n_sae_units)]
        sae_corr_df = pd.DataFrame(sae_corr_matrix, index=sae_unit_names, columns=predictors)

        # 3. Plot both side-by-side
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(42, 24))
        vmin, vmax = -1, 1
        fsize = 50

        im1 = sns.heatmap(
            mlp_corr_df,
            ax=axes[0],
            annot=False,
            fmt=".2f",
            cmap='bwr',
            vmin=vmin,
            vmax=vmax,
            center=0,
            cbar=False
        )
        axes[0].set_title("MLP Hidden Activation Correlations", fontsize=fsize+10)
        axes[0].set_xlabel("Predictors", fontsize=fsize)
        axes[0].set_ylabel("Hidden Neuron", fontsize=fsize)
        axes[0].set_yticklabels(axes[0].get_ymajorticklabels(), rotation=0, fontsize=fsize)
        axes[0].set_xticklabels(axes[0].get_xmajorticklabels(), rotation=0, fontsize=fsize)
        
        im2 = sns.heatmap(
            sae_corr_df,
            ax=axes[1],
            annot=False,
            fmt=".2f",
            cmap='bwr',
            vmin=vmin,
            vmax=vmax,
            center=0,
            cbar=False
        )
        axes[1].set_title("SAE Activation Correlations", fontsize=fsize+10)
        axes[1].set_xlabel("Predictors", fontsize=fsize)
        axes[1].set_ylabel("")
        axes[1].set_yticklabels(axes[1].get_ymajorticklabels(), rotation=0, fontsize=fsize)
        axes[1].set_xticklabels(axes[1].get_xmajorticklabels(), rotation=0, fontsize=fsize)

        for i in range(n_hidden_units):
            for j in range(n_predictors):
                val = mlp_corr_df.iloc[i, j]
                if -0.4 <= val <= 0.4:
                    rect = patches.Rectangle((j, i), 1, 1, fill=False, hatch='/')
                    axes[0].add_patch(rect)
                else:
                    rect = patches.Rectangle((j, i), 1, 1, fill=False, linewidth=3.5)
                    axes[0].add_patch(rect)

        for i in range(n_sae_units):
            for j in range(n_predictors):
                val = sae_corr_df.iloc[i, j]
                if -0.4 <= val <= 0.4 or math.isnan(val):
                    rect = patches.Rectangle((j, i), 1, 1, fill=False, hatch='/')
                    axes[1].add_patch(rect)
                else:
                    rect = patches.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', linewidth=3.5)
                    axes[1].add_patch(rect)

        cbar = fig.colorbar(
            im1.collections[0],
            ax=axes,
            orientation="horizontal",
            fraction=0.05,
            pad=-0.25,
            aspect=50
        )

        cbar.set_label("Correlation", fontsize=fsize)
        cbar.ax.tick_params(labelsize=fsize)

        plt.suptitle(str(r2))
        plt.tight_layout(rect=[0,0,1,0.95])


        above_or_below_threshold_counts = mlp_corr_df.gt(0.4).sum(axis=1) + mlp_corr_df.lt(-0.4).sum(axis=1)
        filtered_counts = above_or_below_threshold_counts[above_or_below_threshold_counts > 0]
        mlp_avg = filtered_counts.mean()
        above_or_below_threshold_counts = sae_corr_df.gt(0.4).sum(axis=1) + sae_corr_df.lt(-0.4).sum(axis=1)
        filtered_counts = above_or_below_threshold_counts[above_or_below_threshold_counts > 0]
        sae_avg = filtered_counts.mean()


        plt.savefig('../figures/' + str(len(filtered_counts)) + '_' + str(round((sae_avg - mlp_avg), 4)) + '_' + str(round(r2, 4)) + '_' + ''.join(predictors) + '.png')


    # mlp_corr_df, sae_corr_df, mlp_hidden_activations, sae_encoded_activations = plot_mlp_and_sae_correlations(model, activation_model, encoder_model, sae_test_X, ['refl$_{29}$', '$\lambda$', 'Fs', 't$_{15}$', 'rh$_{15}$'])
    plot_mlp_and_sae_correlations(model, activation_model, encoder_model, sae_test_X, predictors, r2_test)


test_variables = [
    'reflectivity_3', 'reflectivity_15', 'reflectivity_29', 'dv_3', 'dv_15', 'dv_29',
    'n0', 'Nt', 'lambda', 'Fs', 'Rho', 'Dm', 't_15', 'rh_15',
]

### How many sample combinations do we want to look at?
N_SAMPLES = 250

all_combinations = list(itertools.combinations(test_variables, N_INPUTS_TO_SEARCH))

# Randomly pick some number of them (n_samples)
chosen_combos = random.sample(all_combinations, N_SAMPLES)

results = []
# Note that this can take a while if running with many samples
for combo in chosen_combos:
    predictors = list(combo)
    
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Clear session
    tf.keras.backend.clear_session()
    gc.collect()

    run_all(predictors)
    # break