#!/usr/bin/env python3

import os
import pandas as pd
import sys
import warnings
import json
import shutil
import datetime
import re

def validate_config_and_data(config):
    """
    Validate the input data and custom variables to make sure the file exists, that the input columns exist, that the sequences and groups are in the right format, 
    that all other custom variables are the right type and that sequences are not too long and there are not too many different groups.
    """
    # Ensure 'full_file' is defined in custom_config
    if 'full_file' not in config:
        raise ValueError("'full_file' must be defined in the custom configuration.")
    
    # Load the dataset to perform checks
    try:
        data = pd.read_csv(config['full_file'], sep=config['sep'])
    except Exception as e:
        raise ValueError(f"Failed to load the data file {config['full_file']}: {e}")
    
    # Check if the required columns are present
    if not config['group_column_name']:
        raise ValueError("The 'group_column_name' must be defined in the custom configuration.")
    
    # Check the group column for less than 10 unique values
    if data[config['group_column_name']].nunique() >= 10:
        raise ValueError(f"The group column '{config['group_column_name']}' must have less than 10 unique values.")
        
    missing_columns = []
    columns = [config['sequence_column_name'], config['group_column_name'], config['signal_column_name']]
    for col in [item for sublist in columns for item in (sublist if isinstance(sublist, list) else [sublist])]:
        if col:
            if col not in data.columns:
                missing_columns.append(col)
    if len(missing_columns) > 0:
        raise ValueError(f"The following required columns are missing from the data file: {', '.join(missing_columns)}. Please check the column names or the separator and try again.")

    # Check the sequence column for valid nucleotide strings
    if config['sequence_column_name']:
        if not data[config['sequence_column_name']].apply(lambda x: isinstance(x, str) and all(c in 'ATCGN' for c in x)).all():
            raise ValueError(f"The sequence column '{config['sequence_column_name']}' contains invalid nucleotide strings.")
        # Check for the length of the longest sequence
        max_sequence_length = data[config['sequence_column_name']].str.len().max()
        print(f"The longest sequence is {max_sequence_length} nucleotides long.")
        if max_sequence_length >= 100000:
            warnings.warn("Warning: The longest sequence is 100,000 or more nucleotides long. This is not a problem but can cause extremly high memory usage - check your available vram.", RuntimeWarning)
        elif 10000 <= max_sequence_length < 100000:
            warnings.warn("Mild Warning: The longest sequence is between 10,000 and 100,000 nucleotides long. This is not a problem but can cause extensive memory usage - check your available vram.", RuntimeWarning)
    
    if config['signal_column_name']:
        lengths_of_first_item = []
        for col in config['signal_column_name']:
            length_of_first_item = data[col].str.split(',').apply(len)
            lengths_of_first_item.append(length_of_first_item)
            # Regular expression pattern to match a string of comma-separated numbers
            pattern = re.compile(r'^(\d+(\.\d+)?)(,\d+(\.\d+)?)*$')

            # Check the column for valid comma-separated numbers
            if not data[col].apply(lambda x: isinstance(x, str) and bool(pattern.match(x.strip()))).all():
                raise ValueError(f"The column '{col}' contains invalid comma-separated numbers.")
            
        if not all(all(series[i] == lengths_of_first_item[0][i] for series in lengths_of_first_item) for i in range(len(lengths_of_first_item[0]))):
            raise ValueError("The signal columns must have the same number of values.")

    # If all checks pass, print a success message
    print("Data has been successfully loaded and validated. All necessary columns are present, and data is in the expected format.")
    

def create_main_dir(config):
    """
    Create the main out dir if not yet present
    """
    os.makedirs(config['out_dir'], exist_ok=True)



def create_out_dir_and_copy_source_file_in(config):
    """
    Create a directory inside "models" based on the current timestamp and type of run
    """
    current_time = datetime.datetime.now().strftime("%d%m%Y_%H%M%S%f")[:18]
    if config["OPTIMISE"]:
        directory_name = f"{config['out_dir']}/optimising_run_{current_time}"
    else:
        directory_name = f"{config['out_dir']}/individual_run_{current_time}"
    os.makedirs(directory_name, exist_ok=True)
    
    print(f"Out dir: {directory_name}")

    # # Copy the source file to the new directory to save as a reference
    # source_file_path = "main_guide.ipynb"
    # destination_file_path = os.path.join(directory_name, "main_guide.ipynb")
    # shutil.copy(source_file_path, destination_file_path)

    # # Also export a pip-compatible text file with module versions for reproducibility
    # session_info.show(std_lib = True, write_req_file = True, req_file_name=os.path.join(directory_name, "session_info_requirements.txt"))
    
    return directory_name


def create_out_dirs_for_optimisation(directory_name, optimise):
    # First create a general output directory
    os.makedirs(f"{directory_name}/results", exist_ok=True)
    general_out = f"{directory_name}/results"
    if optimise:
        model_info_out = f"{general_out}/saved_trained_model_with_best_parameters"
        os.makedirs(model_info_out, exist_ok=True)
        model_eval_out = f"{general_out}/evaluated_trained_model_with_best_parameters"
        os.makedirs(model_eval_out, exist_ok=True)
        optimisation_out = f"{general_out}/optimisation_process_and_results"
        os.makedirs(optimisation_out, exist_ok=True)
        return model_info_out, model_eval_out, optimisation_out
    
    model_info_out = f"{general_out}/saved_trained_model"
    os.makedirs(model_info_out, exist_ok=True)
    model_eval_out = f"{general_out}/evaluated_trained_model"
    os.makedirs(model_eval_out, exist_ok=True)
    return model_info_out, model_eval_out



def save_config(config, directory_name):

    if config['OPTIMISE']:
        with open(os.path.join(directory_name, "optimal_config.json"), "w") as f:
            json.dump(config, f, indent=4)
    else:
        with open(os.path.join(directory_name, "config.json"), "w") as f:
            json.dump(config, f, indent=4)   


def import_config(directory_name):
    config_file_path = os.path.join(directory_name, 'config.json')
    alternative_config_file_path = os.path.join(directory_name, 'optimal_config.json')

    # Check which file exists and select it
    if os.path.exists(config_file_path):
        file_to_open = config_file_path
    elif os.path.exists(alternative_config_file_path):
        file_to_open = alternative_config_file_path
    else:
        raise FileNotFoundError("Neither config.json nor optimal_config.json was found.")
    
    # Now open and load the configuration from the selected file
    with open(file_to_open) as f:
        config = json.load(f)
        
    return config


def create_model_explain_out_dir(directory_name):
    
    model_explanation_out = f"{directory_name}/results/model_explanation"
    os.makedirs(model_explanation_out, exist_ok=True)

    return model_explanation_out