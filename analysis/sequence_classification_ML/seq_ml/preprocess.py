#!/usr/bin/env python3

# Import specific functions or classes from other modules
from .utils import validate_config_and_data, create_main_dir, create_out_dir_and_copy_source_file_in

# Import any external libraries needed for preprocessing
import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.cluster import AgglomerativeClustering
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Input, Conv1D, LSTM, Dense, Flatten, Bidirectional, BatchNormalization, Activation, LayerNormalization
from tensorflow.keras import regularizers
from tensorflow.keras import Input, Model
import json
import random
import numpy as np


def prepare_train_test(config, directory_name):
    """
    Split data.
    """

    full_dataframe = pd.read_csv(config['full_file'], sep=config['sep'])
    
    # Extract the test size and compute the combined train and validation size
    test_size = config['test_and_validation_size'] / 2
    remaining_size = 1 - test_size

    # Split the full dataframe into test set and remaining data
    train_and_val, test = train_test_split(
        full_dataframe,
        test_size=test_size,
        stratify=full_dataframe[config['group_column_name']],
        random_state=42  # Fixed random state to ensure the test set is always the same
    )
    
    
    random_state = 42 if (config["fix_train_val_split"] or config["OPTIMISE"]) else None
            
    # Split the remaining data into training and validation sets
    # The test_size here is the proportion of the validation set relative to the remaining data
    train, validation = train_test_split(
        train_and_val,
        test_size=test_size / remaining_size,  # Adjust to get the correct validation size
        stratify=train_and_val[config['group_column_name']],
        random_state=random_state  # No fixed random state for randomness
    )

    os.makedirs(f"{directory_name}/created_datasets", exist_ok=True)
    
    full_csv_out = f"{directory_name}/created_datasets/original_file.tsv"
    full_dataframe.to_csv(full_csv_out, sep=config['sep'], index=True, index_label='index_column')
    
    training_csv_out = f"{directory_name}/created_datasets/training.tsv"
    train.to_csv(training_csv_out, sep=config['sep'], index=True, index_label='index_column')

    validation_csv_out = f"{directory_name}/created_datasets/validation.tsv"
    validation.to_csv(validation_csv_out, sep=config['sep'], index=True, index_label='index_column')
    
    test_csv_out = f"{directory_name}/created_datasets/testing.tsv"
    test.to_csv(test_csv_out, sep=config['sep'], index=True, index_label='index_column')

def prepare_the_necessary_directories_and_raw_files(config):
    validate_config_and_data(config)
    create_main_dir(config)
    directory_name = create_out_dir_and_copy_source_file_in(config)
    prepare_train_test(config, directory_name)
    return directory_name

# Set up the data encoding and processing functions in order to enable procedural batch generation from raw files represented as a RepeatDataset structure. The sequences and groups are proceduraly one-hot encoded and padded to the longest sequence.

def parse_categories(group, categories_list):
    # Convert the category to a one-hot encoded vector
    category_index = tf.argmax(tf.equal(categories_list, group))
    one_hot_encoded_group = tf.one_hot(category_index, depth=len(categories_list))
    return one_hot_encoded_group

def one_hot_encode(sequence, longest_sequence, encoding=['A', 'C', 'G', 'T']):
    # Define the mapping for nucleotides to integers
    nucleotide_to_index = tf.constant(encoding)
    # Split the sequence into characters
    nucleotides = tf.strings.bytes_split(sequence)
    # Find the indices of each nucleotide in the sequence
    indices = tf.argmax(tf.equal(tf.expand_dims(nucleotides, -1), nucleotide_to_index), axis=-1)
    # Perform one-hot encoding
    one_hot_encoded = tf.one_hot(indices, depth=4)
    # Pad the sequence to the desired width
    padded_sequence = tf.pad(one_hot_encoded, paddings=[[0, longest_sequence - tf.shape(one_hot_encoded)[0]], [0, 0]], constant_values=-1)
    return padded_sequence

def parse_crosslink_scores_full(scores_str, longest_sequence):
    """Parse the comma-separated crosslink scores and create a list of scores with the given sequence length."""
    
    # Split the string by commas
    scores_list = tf.strings.split(scores_str, ',')
    
    converted_scores = tf.strings.to_number(scores_list, out_type=tf.float32)

    scores = converted_scores
    
    # Padding with -1 to match the longest_sequence length
    padded_scores = tf.concat([scores, tf.fill([longest_sequence - tf.size(scores)], -1.0)], axis=0)
    return padded_scores

def parse_crosslink_scores_sparse(scores_str, longest_sequence):
    """Parse the sparse encoded crosslink scores and create a list of scores with the given sequence length."""

    def handle_empty_or_invalid():
        # Return a tensor of zeros with the given sequence length
        return tf.zeros([longest_sequence], dtype=tf.float32)

    def handle_valid_scores():
        # Process the valid scores_str
        score_pairs = tf.strings.split(scores_str, ';')
        indices_scores = tf.strings.split(score_pairs, ':')
        indices_scores = indices_scores.to_tensor()
        indices = tf.strings.to_number(indices_scores[:, 0], out_type=tf.int32)
        scores = tf.strings.to_number(indices_scores[:, 1], out_type=tf.float32)
        
        full_scores = tf.zeros([longest_sequence], dtype=tf.float32)
        full_scores = tf.tensor_scatter_nd_update(full_scores, tf.expand_dims(indices, 1), scores)
        return full_scores

    # Check if the scores_str contains a ":"
    contains_colon = tf.strings.regex_full_match(scores_str, ".*:.*")

    return tf.cond(contains_colon, handle_valid_scores, handle_empty_or_invalid)

def process_line(line, config, column_indices, categories_list, longest_sequence):
    fields = tf.io.decode_csv(line, record_defaults=[[""]] * len(column_indices), field_delim=config['sep'])
    
    if config['sequence_column_name']:
        sequence = fields[column_indices[config['sequence_column_name']]]
        
    scores_str_list = []
    if config['signal_column_name']:
        scores_str_list = [fields[column_indices[clip_column]] for clip_column in config['signal_column_name']]
        if config['sequence_column_name']:
            one_hot_encoded_sequence = one_hot_encode(sequence, longest_sequence)
            crosslink_scores_list = [parse_crosslink_scores_full(scores_str, longest_sequence) for scores_str in scores_str_list]
            crosslink_scores_expanded = [tf.expand_dims(scores, axis=-1) for scores in crosslink_scores_list]
            combined_encoded = tf.concat([one_hot_encoded_sequence] + crosslink_scores_expanded, axis=-1)
        else:
            crosslink_scores_list = [parse_crosslink_scores_full(scores_str, longest_sequence) for scores_str in scores_str_list]
            crosslink_scores_expanded = [tf.expand_dims(scores, axis=-1) for scores in crosslink_scores_list]
            combined_encoded = tf.concat(crosslink_scores_expanded, axis=-1)
    else:
        combined_encoded = one_hot_encode(sequence, longest_sequence)
        
    groups = fields[column_indices[config['group_column_name']]]    
    encoded_groups = parse_categories(groups, categories_list)

    return combined_encoded, encoded_groups

def shuffle_file(file_path, out_dir_for_scrambled_data):
    # Specify the path to your file
    shuffled_file_path = f"{out_dir_for_scrambled_data}/original_file_shuffled.tsv"
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Separate the header and the rows
    header = lines[0]
    rows = lines[1:]
    # Set the seed
    seed_value = 42
    random.seed(seed_value)
    # Shuffle the rows
    random.shuffle(rows)
    # Write the header and the shuffled rows back to a new file
    with open(shuffled_file_path, 'w') as file:
        file.write(header)  # Write the header first
        file.writelines(rows)  # Then write the shuffled rows
    file_path = shuffled_file_path
    return file_path

def encode_from_csv(file_path, config, column_indices, longest_sequence, categories_list, out_dir_for_scrambled_data=None):
        
    if out_dir_for_scrambled_data:
        file_path = shuffle_file(file_path, out_dir_for_scrambled_data)
            
    dataset = tf.data.TextLineDataset(file_path).skip(1)  # Skip header
    
    dataset = dataset.map(lambda line: process_line(line, config, column_indices, categories_list, longest_sequence),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.batch(config['batch_size'])
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset



# Prepare the information needed to encode the raw data in a RepeatDataset structure. Store the raw validation and training files in a ReturnDataset structure that contains the logic to encode the sequences and groups as needed. Get the shape of first RepeatDataset data batch. Calculate the number of steps needed to process the entire training and validation datasets.

def get_column_indices_max_length_and_categories(directory_name, config):
    # Read the first line to get column names
    with tf.io.gfile.GFile(f"{directory_name}/created_datasets/original_file.tsv", 'r') as f:
        column_names = f.readline().strip().split(config['sep'])
        
    column_indices = {name: index for index, name in enumerate(column_names)}
    # Initialize to find the longest sequence and unique groups
    longest_sequence = 0
    unique_groups = set()

    with tf.io.gfile.GFile(f"{directory_name}/created_datasets/original_file.tsv", 'r') as f:
        next(f)  # Skip header line
        for line in f:
            fields = line.strip().split(config['sep'])
            if config['sequence_column_name']:
                sequence = fields[column_indices[config['sequence_column_name']]]
                length = len(sequence)
            else:
                signal = fields[column_indices[config['signal_column_name'][0]]]
                length = len(signal.split(','))

            group = fields[column_indices[config['group_column_name']]]
            longest_sequence = max(longest_sequence, length)
            unique_groups.add(group)

    # Sort the unique groups to ensure consistency
    categories_list = tf.constant(sorted(unique_groups))

    return column_indices, longest_sequence, categories_list

def get_shapes_of_inputs(validation_dataset):
    
    # Get the first batch from the dataset
    first_item_encoded = next(iter(validation_dataset.take(1)))
    
    # Extract features and labels from the first batch
    features, labels = first_item_encoded
    
    # Convert TensorFlow tensors to NumPy arrays and get shapes
    features_shape = features.numpy().shape
    labels_shape = labels.numpy().shape
    
    return features_shape, labels_shape

def prepare_the_data_for_training(config, directory_name):
    
    column_indices, longest_sequence, categories_list = get_column_indices_max_length_and_categories(directory_name, config)
    training_csv_out = f"{directory_name}/created_datasets/training.tsv"
    training_dataset = encode_from_csv(training_csv_out, config, column_indices, longest_sequence, categories_list).repeat()
    
    validation_csv_out = f"{directory_name}/created_datasets/validation.tsv"
    validation_dataset = encode_from_csv(validation_csv_out, config, column_indices, longest_sequence, categories_list).repeat()
    
    # Get the lengths of the CSV files without the header
    training_length = sum(1 for _ in open(training_csv_out)) - 1
    validation_length = sum(1 for _ in open(validation_csv_out)) - 1
    
    # Calculate the steps per epoch for training and validation
    training_steps = training_length // config['batch_size'] + 1
    validation_steps = validation_length // config['batch_size'] + 1
    
    # Get the shapes of the inputs
    features_shape, labels_shape = get_shapes_of_inputs(validation_dataset)
    
    return training_dataset, validation_dataset, training_steps, validation_steps, features_shape, labels_shape, categories_list


def dataset_to_numpy(dataset, steps):
    """
    Prepare the validation data as a numpy array - needed for the evaluation functions.
    """
    val_dataset = dataset.take(steps)
    # To get numpy arrays from the dataset
    features_list = []
    labels_list = []

    for features, labels in val_dataset:
        features_list.append(features.numpy())
        labels_list.append(labels.numpy())

    X_val = np.concatenate(features_list, axis=0) if features_list else np.array([])
    y_val = np.concatenate(labels_list, axis=0) if labels_list else np.array([])
    
    return X_val, y_val