#!/usr/bin/env python3

# Import specific functions or classes from other modules
from .utils import create_out_dirs_for_optimisation, save_config
from .preprocess import prepare_the_necessary_directories_and_raw_files, prepare_the_data_for_training, get_shapes_of_inputs, dataset_to_numpy

# Import external libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Don't show TF debug info while training
import sys
import warnings
import numpy as np
import seaborn as sns
import pandas as pd
import re
import json
import optuna
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.cluster import AgglomerativeClustering
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Input, Conv1D, LSTM, Dense, Bidirectional, BatchNormalization, Activation, LayerNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate, GRU
from tensorflow.keras import regularizers, Input, Model



def build_model(config, input_shape, output_shape):
    """
    Build the model based on the hyperparameters in a config file. The architecture can be partially customised from the custom_config file. 
    For advanced use this can also be adapted, but it should serve as a good starting point for sequence classification tasks as is.
    """
    model = Sequential()
    model.add(Input(shape=(input_shape[1], input_shape[2])))

    for block in range(config['num_blocks']):
        model.add(Conv1D(filters=int(config['Final_CNN_units'] * ((1-config['CNN_Units_Increase_By_Percent']) ** (config['num_blocks']-1-block))), 
                         kernel_size=config['kernel_size']*((config['Increase_Kernel_By']) ** block), 
                         padding="same", 
                         kernel_regularizer=regularizers.l1(l1=config['l1_lambda']), 
                         dilation_rate=((config['Increase_Dilation_By']) ** block)))
        
        if config['after_cnn_normalization'] == 'BatchNormalization':
            model.add(BatchNormalization())
        elif config['after_cnn_normalization'] == 'LayerNormalization':
            model.add(LayerNormalization())
               
        model.add(Activation('relu'))
        model.add(Dropout(rate=config['dropout_prob']))
        model.add(MaxPooling1D(pool_size=config['reduce_by']))
    
    if config['rnn_type'] == 'LSTM':
        model.add(Bidirectional(LSTM(units=config['LSTM_units'], 
                                     return_sequences=False, 
                                     kernel_regularizer=regularizers.l1(l1=config['l1_lambda']))))
    elif config['rnn_type'] == 'GRU':
        model.add(Bidirectional(GRU(units=config['LSTM_units'], 
                                    return_sequences=False, 
                                    kernel_regularizer=regularizers.l1(l1=config['l1_lambda']))))
    
    if config['after_lstm_normalization'] == 'BatchNormalization':
        model.add(BatchNormalization())
    elif config['after_lstm_normalization'] == 'LayerNormalization':
        model.add(LayerNormalization())
        
    model.add(Dropout(rate=config['dropout_prob']))
    
    for rep in range(2):
        model.add(Dense(units=config['Dense_units']//(rep+1),
                        activation='relu', 
                        kernel_regularizer=regularizers.l1(l1=config['l1_lambda'])))
        
        if config['after_lstm_normalization'] == 'BatchNormalization':
            model.add(BatchNormalization())
        elif config['after_lstm_normalization'] == 'LayerNormalization':
            model.add(LayerNormalization())
            
        model.add(Dropout(rate=config['dropout_prob']))
    
    model.add(Dense(units=output_shape[1], activation='softmax'))
    
    return model

# Validate the model size to make sure the memory usage does not excede the gpu vram limit. This is very much an estimation and should not be treated as a fact.
def estimate_model_memory_usage_from_layers(config, input_shape, output_shape):
    
    # Assuming float32 uses 4 bytes
    bytes_per_param = 4
    
    # Calculate input size assuming batch size of 1
    input_size = np.prod(input_shape) * bytes_per_param
    
    total_memory = 0  # Start with input size
    total_memory += input_size
    
    current_shape = input_shape
    
    for block in range(config['num_blocks']):
        filters = int(config['Final_CNN_units'] * ((1 - config['CNN_Units_Increase_By_Percent']) ** (config['num_blocks'] - 1 - block)))
        kernel_size = config['kernel_size'] * (config['Increase_Kernel_By'] ** block)
        output_shape = (current_shape[0], filters)
        
        # Memory for parameters in the Conv1D layer (kernel + bias)
        num_params = filters * current_shape[1] * kernel_size + filters
        total_memory += num_params * bytes_per_param
        
        # Memory for output of this layer (feature map)
        output_size = np.prod(output_shape) * bytes_per_param
        total_memory += output_size
        
        # Update current shape for next layer calculations
        current_shape = output_shape
    
    # Additional LSTM and Dense layers calculations here
    # LSTM parameters
    lstm_units = config['LSTM_units']
    num_lstm_params = 4 * (lstm_units * current_shape[1] + lstm_units * lstm_units + lstm_units)
    total_memory += num_lstm_params * bytes_per_param
    
    # Output from LSTM (assuming concatenation does not increase size)
    lstm_output_shape = (lstm_units,)
    lstm_output_size = np.prod(lstm_output_shape) * bytes_per_param
    total_memory += lstm_output_size
    
    # Dense layers
    for i in range(2):
        dense_units = config['Dense_units'] // (2 ** i)
        num_dense_params = dense_units * np.prod(lstm_output_shape) + dense_units
        total_memory += num_dense_params * bytes_per_param
        
        # Update output shape
        lstm_output_shape = (dense_units,)
    
    # Final output layer
    num_output_params = output_shape[1] * np.prod(lstm_output_shape) + output_shape[1]
    total_memory += num_output_params * bytes_per_param

    total_memory_mb = total_memory / (1024 ** 2)  # Convert to MB

    print("Memory usage estimated.")
    return total_memory_mb, total_memory_mb / 1024  # MB and GB

def validate_memory_usage_before_building_the_model(config, features_shape, labels_shape):

    # Estimate memory usage and dynamically decide the unit for display
    memory_usage_mb, memory_usage_gb = estimate_model_memory_usage_from_layers(config, features_shape, labels_shape)
    if memory_usage_mb > 1024:  # More than 1 GB
        print(f"Estimated memory usage for a batch: {memory_usage_gb:.2f} GB")
    else:
        print(f"Estimated memory usage for a batch: {int(memory_usage_mb)} MB")
        
    if memory_usage_gb > 0.5:
        raise MemoryError(f"Estimated memory usage exceeds 0.5 GB ({memory_usage_gb:.2f} GB). Halting execution to prevent system overload.")

def validate_parameter_number(model):
    
    total_params = model.count_params()
    if total_params >= 1e6:
        MemoryError(f"The model has has over 1 milion parameters ({total_params}). Halting execution to prevent system overload.")
    elif total_params > 2e5:
        warnings.warn("Serious Warning: The model has over 200.000 parameters. This is too much for most cases.", RuntimeWarning)
    elif total_params > 1e5:
        warnings.warn("Warning: The model has over 5 million parameters. Consider simplifying the model if training is inefficient.", RuntimeWarning)


def train_the_model(config, training_dataset, validation_dataset, training_steps, validation_steps, features_shape, labels_shape):
    """
    Train the model based on the config file and save the training history.
    """
    features_shape, labels_shape = get_shapes_of_inputs(validation_dataset)
    print("features_shape, labels_shape", features_shape, labels_shape)
    validate_memory_usage_before_building_the_model(config, features_shape, labels_shape)

    model = build_model(config, input_shape=features_shape, output_shape=labels_shape)
    
    optimizer = Adam(learning_rate=config['learning_rate'])

    # Compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc'), 'accuracy'])
    
    if not config['OPTIMISE']:
        model.summary()
        
    validate_parameter_number(model)
    
    # Early stopping for both training and validation losses
    early_stopping = EarlyStopping(monitor='val_auc', patience=config['patience'], restore_best_weights=True)    
    
    verbose = 0 if config['OPTIMISE'] else 1

    # Use this dictionary in the fit method
    history = model.fit(training_dataset,
                        epochs=config['epochs'],
                        verbose=verbose,
                        steps_per_epoch=training_steps,
                        validation_data=validation_dataset,
                        validation_steps=validation_steps,
                        callbacks=[early_stopping])

    return model, history


def predict_validation_dataset(model, X_val, y_val, batch_size):
    """
    Predict the validation dataset.
    """
    y_pred = np.vstack([model.predict(X_val[i:i + batch_size]) for i in range(0, len(X_val), batch_size)])
    # Get the predicted classes
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_val_classes = np.argmax(y_val, axis=1)
    return y_pred, y_pred_classes, y_val_classes

def auroc_score(y_val, y_pred, average='standard'):
    """
    Calculate ROC AUC for each class.
    """
    auc_scores = []
    for i in range(y_val.shape[1]):  # iterate over each class
        # Compute ROC AUC for the i-th class
        auc = roc_auc_score(y_val[:, i], y_pred[:, i])
        auc_scores.append(auc)
    if average == 'weighted':
        # Optionally, calculate a weighted average AUROC if classes are imbalanced
        weights = y_val.mean(axis=0)
        weighted_average_auc = np.average(auc_scores, weights=weights)
        return weighted_average_auc
    else:
        return np.average(auc_scores)

def save_evaluation_metrics(y_val_classes, y_pred_classes, y_val, y_pred, directory_name):
    """
    Save the model's accuracy, precision, recall, F1-score, and AUROC.
    """
    metrics = {
        "Accuracy": accuracy_score(y_val_classes, y_pred_classes),
        "Precision": precision_score(y_val_classes, y_pred_classes, average='weighted', zero_division=0),
        "Recall": recall_score(y_val_classes, y_pred_classes, average='weighted'),
        "F1-score": f1_score(y_val_classes, y_pred_classes, average='weighted'),
        "AUROC": auroc_score(y_val, y_pred, average='weighted')
    }
    with open(os.path.join(directory_name, "evaluation_metrics.txt"), "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

def save_confusion_matrix(y_val_classes, y_pred_classes, directory_name, categories_list):
    """
    Plot and save the confusion matrix.
    """
    # Assume y_val_classes and y_pred_classes are defined elsewhere
    cm = confusion_matrix(y_val_classes, y_pred_classes)

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(7, 7))
    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues", cbar=False, vmax=1, vmin=0, xticklabels=categories_list, yticklabels=categories_list)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix in Percentages')
    plt.savefig(os.path.join(directory_name, "confusion_matrix_validation_dataset.png"))  # Save the plot as PNG
    plt.close()  # Close the plot

def save_training_curves(history, directory_name):
    """
    Get the training and validation loss and accuracy values from history.
    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_auc = history.history['auc']
    val_auc = history.history['val_auc']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    # Create a figure with two subplots for loss and accuracy
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 16))

    # Plot the loss curves on the first subplot
    ax1.plot(train_loss, label='Training Loss')
    ax1.plot(val_loss, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot the accuracy curves on the second subplot
    ax2.plot(train_auc, label='Training AUC')
    ax2.plot(val_auc, label='Validation AUC')
    ax2.set_title('Training and Validation AUC')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('AUC')
    ax2.legend()
    
    # Plot the accuracy curves on the second subplot
    ax3.plot(train_accuracy, label='Training Accuracy')
    ax3.plot(val_accuracy, label='Validation Accuracy')
    ax3.set_title('Training and Validation Accuracy')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy')
    ax3.legend()

    # Save the figure containing both subplots
    plt.savefig(os.path.join(directory_name, "training_validation_curves.png"))
    plt.close(fig)  # Close the figure to free up memory


# Functions to save the model, configuration files, model summary, weights and architecture, and hyperparameters.

def save_training_history(history, directory_name):
    with open(os.path.join(directory_name, "training_history.json"), "w") as f:
        # Convert possible NumPy types in the history to native Python types
        history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
        json.dump(history_dict, f, indent=4)


def save_model_summary(model, directory_name):
    with open(os.path.join(directory_name, "model_summary.txt"), "w") as f:
        # Redirect the default standard output to the file
        original_stdout = sys.stdout  # Save the original standard output
        sys.stdout = f  # Redirect to the file
        model.summary()  # This will write to the file instead of the console
        sys.stdout = original_stdout  # Reset standard output to its original value


def save_weights_and_architecture(model, directory_name):
    model.save_weights(os.path.join(directory_name, "model_weights.h5"))
    with open(os.path.join(directory_name, "model_architecture.json"), "w") as f:
        f.write(model.to_json())


def save_models_hyperparameters(model, batch_size, directory_name):
    with open(os.path.join(directory_name, "hyperparameters.txt"), "w") as f:
        f.write(str(model.get_config()))
        f.write(f"Batch Size: {batch_size}\n")
        f.write(str(model.get_config()))


def save_full_model(model, directory_name):
    model.save(os.path.join(directory_name, "full_model.h5"))
    optimizer_config = model.optimizer.get_config()
    with open(os.path.join(directory_name, "config.txt"), "w") as f:
        for key, value in optimizer_config.items():
            f.write(f"{key}: {value}\n")

def save_optimization_details(details, directory_name, config):
    with open(os.path.join(directory_name, 'optimization_results.txt'), 'w') as f:
        f.write("Optimization Results\n")
        f.write("====================\n")
        f.write(f"Best Parameters: {details['best_params']}\n")
        f.write(f"Best Accuracy: {details['best_value']}\n")
    with open(os.path.join(directory_name, 'optimised_config.txt'), 'w') as f:    
        f.write("Optimised Config:\n")
        f.write("====================\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(directory_name, 'optimization_history.txt'), 'w') as f:
        f.write("Optimization History:\n")
        f.write("====================\n")
        for entry in details['history']:
            f.write(f"Trial {entry['trial_number']}: Value: {entry['value']}, Parameters: {entry['params']}\n")  
            
def plot_optimization_history(directory_name, config):
    # File path to the optimization history
    file_path = os.path.join(directory_name, 'optimization_history.txt')

    # Read the file content
    with open(file_path, 'r') as file:
        data = file.read()

    # Extracting the values using regex
    values = [float(x) for x in re.findall(r'Value: ([\d\.\-]+)', data)]

    # Create a DataFrame
    df = pd.DataFrame({'Trial': range(len(values)), 'Value': values})

    # Filter out the -1 values for the rolling average calculation
    df_filtered = df[df['Value'] != -1]

    # Calculate the rolling average with a window of 5
    df_filtered['Rolling_Avg'] = df_filtered['Value'].rolling(window=10, min_periods=1).mean()

    # Finding the maximum value and its index
    max_value = df_filtered['Value'].max()
    max_index = df_filtered['Trial'][df_filtered['Value'].idxmax()]

    plt.figure(figsize=(14, 4))

    # Plotting the training progression as points
    plt.plot(df['Trial'], df['Value'], marker='.', linestyle='', alpha=1, color='royalblue', label='Individual Trials')

    # Plotting the smoothed average with lower alpha
    plt.plot(df_filtered['Trial'], df_filtered['Rolling_Avg'], linestyle='--', color='red', alpha=1,  label='Smoothed Average (5 points)', linewidth=1)

    plt.xlabel('Trial', fontsize=14)
    plt.ylabel('AUC Score', fontsize=14)
    plt.title('Optimisation Progression', fontsize=16)

    # Labeling the maximum value
    plt.annotate(f'Max AUC Score: {max_value}', xy=(max_index, max_value), xytext=(max_index, max_value + 0.05),
                arrowprops=dict(facecolor='red', shrink=0.05, width=3, headwidth=8))

    plt.ylim(0.5, 1)
    plt.xlim(0, config['n_trials'])
    plt.savefig(os.path.join(directory_name, "optimisation_progression.png"))

def save_optimisation(directory_name, study, config):
    # Now, save the optimization history and best parameters
    optimization_details = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'history': [{'trial_number': trial.number, 'value': trial.value, 'params': trial.params} for trial in study.trials]
    }
    
    save_optimization_details(optimization_details, directory_name, config)
    plot_optimization_history(directory_name, config)

def evaluate_and_save_the_model(config, model, history, directory_name, validation_dataset, validation_steps, categories_list, study=None):
    """
    Evaluate the trained model and save it with its configuration.
    """
    # create output directory
    model_info_out, model_eval_out, *model_optimise_out  = create_out_dirs_for_optimisation(directory_name, config["OPTIMISE"])
    # Prepare the validation dataset for evaluation
    X_val, y_val = dataset_to_numpy(validation_dataset, validation_steps)
    
     # Evaluate the model
    y_pred, y_pred_classes, y_val_classes = predict_validation_dataset(model, X_val, y_val, config['batch_size'])
    save_confusion_matrix(y_val_classes, y_pred_classes, model_eval_out, categories_list)
    save_evaluation_metrics(y_val_classes, y_pred_classes, y_val, y_pred, model_eval_out)
    save_training_curves(history, model_eval_out)   
    
    # Save the model information
    save_config(config, directory_name)
    save_training_history(history, model_info_out)
    save_model_summary(model, model_info_out)    
    save_weights_and_architecture(model, model_info_out)
    save_models_hyperparameters(model, config['batch_size'], model_info_out)
    save_full_model(model, model_info_out)
    
    if config["OPTIMISE"]:
        save_optimisation(model_optimise_out[0], study, config)

def get_optuna_config(trial, variables_to_optimize):
    config_optuna = {}
    if 'batch_size' in variables_to_optimize:
        config_optuna['batch_size'] = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    if 'num_blocks' in variables_to_optimize:
        config_optuna['num_blocks'] = trial.suggest_int('num_blocks', 2, 8)
    if 'dropout_prob' in variables_to_optimize:
        config_optuna['dropout_prob'] = trial.suggest_float('dropout_prob', 0.0, 0.4)
    if 'l1_lambda' in variables_to_optimize:
        config_optuna['l1_lambda'] = trial.suggest_loguniform('l1_lambda', 1e-8, 1e-2)
    if 'reduce_by' in variables_to_optimize:
        config_optuna['reduce_by'] = trial.suggest_categorical('reduce_by', [1, 2, 3, 4])
    if 'Final_CNN_units' in variables_to_optimize:
        config_optuna['Final_CNN_units'] = trial.suggest_categorical('Final_CNN_units', [16, 32, 64, 128, 256, 512, 1024])
    if 'CNN_Units_Increase_By_Percent' in variables_to_optimize:
        config_optuna['CNN_Units_Increase_By_Percent'] = trial.suggest_categorical('CNN_Units_Increase_By_Percent', [0.0, 0.25, 0.5, 0.75])
    if 'LSTM_units' in variables_to_optimize:
        config_optuna['LSTM_units'] = trial.suggest_categorical('LSTM_units', [16, 32, 64, 128, 256, 512, 1024])
    if 'Dense_units' in variables_to_optimize:
        config_optuna['Dense_units'] = trial.suggest_categorical('Dense_units', [16, 32, 64, 128, 256, 512, 1024])
    if 'kernel_size' in variables_to_optimize:
        config_optuna['kernel_size'] = trial.suggest_categorical('kernel_size', [3, 5, 7, 9])
    if 'Increase_Kernel_By' in variables_to_optimize:
        config_optuna['Increase_Kernel_By'] = trial.suggest_int('Increase_Kernel_By', 1, 4)
    if 'Increase_Dilation_By' in variables_to_optimize:
        config_optuna['Increase_Dilation_By'] = trial.suggest_int('Increase_Dilation_By', 1, 4)
    if 'learning_rate' in variables_to_optimize:
        config_optuna['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    if 'after_cnn_normalization' in variables_to_optimize:
        config_optuna['after_cnn_normalization'] = trial.suggest_categorical('after_cnn_normalization', [None, 'BatchNormalization', 'LayerNormalization'])
    if 'after_lstm_normalization' in variables_to_optimize:
        config_optuna['after_lstm_normalization'] = trial.suggest_categorical('after_lstm_normalization', [None, 'BatchNormalization', 'LayerNormalization'])
    if 'rnn_type' in variables_to_optimize:
        config_optuna['rnn_type'] = trial.suggest_categorical('rnn_type', ['LSTM', 'GRU'])

    return config_optuna

def objective(trial, training_dataset, validation_dataset, training_steps, validation_steps, features_shape, labels_shape, config):
    """
    Define the objective function for optuna optimization. Create the search space for each parameter.
    """
    try:
        config_optuna = get_optuna_config(trial, config["hyperparameters_to_optimise"])
        
        config_trial = {**config, **config_optuna}
        
        # Train the model with the trial's hyperparameters
        model, history = train_the_model(config_trial, training_dataset, validation_dataset, training_steps, validation_steps, features_shape, labels_shape)

        # Evaluate the model on the validation set to get the 'true' best performance
        # Note: This evaluation step is crucial if early stopping might have restored the model to a state from a previous epoch
        val_loss, val_auc, val_accuracy = model.evaluate(validation_dataset, steps=validation_steps)

        return val_auc
    
    except Exception as e:
        # Log the error or handle it as per your needs
        print(f"Error during trial: {e}")
        # print config_trial to see the error
        print(config_trial)
        # Return a very low accuracy value
        return -1.0
    

def optimise_with_optuna(config, training_dataset, validation_dataset, training_steps, validation_steps, features_shape, labels_shape):
    """
    Execute optuma optimisation of the hyperparameters.
    """
    study = optuna.create_study(direction='maximize')
    
    study.optimize(lambda trial: objective(trial, training_dataset, validation_dataset, training_steps, validation_steps, features_shape, labels_shape, config), n_trials=config["n_trials"])

    # Use the best hyperparameters
    best_params = study.best_trial.params
    best_value = study.best_trial.value
    print("Best hyperparameters:", best_params)
    print("Best AUC:", best_value)
    return study, best_params  
    
def train_or_optimise(config):
    directory_name = prepare_the_necessary_directories_and_raw_files(config)
    training_dataset, validation_dataset, training_steps, validation_steps, features_shape, labels_shape, categories_list = prepare_the_data_for_training(config, directory_name)
    
    study = None
    if config["OPTIMISE"]:
        study, best_params = optimise_with_optuna(config, training_dataset, validation_dataset, training_steps, validation_steps, features_shape, labels_shape)
        config.update(best_params)
    
    model, history = train_the_model(config, training_dataset, validation_dataset, training_steps, validation_steps, features_shape, labels_shape)
    evaluate_and_save_the_model(config, model, history, directory_name, validation_dataset, validation_steps, categories_list, study)
    
def main_function(config):
    [train_or_optimise(config) for _ in range(config["repeat_training_for"])] if not config["OPTIMISE"] else train_or_optimise(config)
    