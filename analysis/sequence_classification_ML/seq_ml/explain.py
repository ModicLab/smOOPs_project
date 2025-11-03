#!/usr/bin/env python3

from .utils import import_config, create_model_explain_out_dir
from .preprocess import encode_from_csv, get_column_indices_max_length_and_categories, dataset_to_numpy, one_hot_encode

import os
import tensorflow as tf
import pandas as pd
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase
import seaborn as sns
import matplotlib.gridspec as gridspec
import subprocess
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import shap
from itertools import product
from tqdm import tqdm
import umap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.manifold import TSNE, Isomap, MDS, SpectralEmbedding
from sklearn.decomposition import PCA, FactorAnalysis,  DictionaryLearning, NMF
from multiprocessing import Pool, cpu_count 


def create_model_explain_out_dir(directory_name):

    model_explanation_out = f"{directory_name}/results/model_explanation"
    os.makedirs(model_explanation_out, exist_ok=True)

    return model_explanation_out

def prepare_for_shap(config, directory_name, size_of_background = 200, interaction_analysis=False):
    """
    Prepares the necessary datasets and calculates the number of batches required for computing SHAP values
    to explain a machine learning model. This function loads and encodes the full dataset, determines the 
    number of batches needed for the entire dataset and a specified background dataset, and selects a 
    representative subset of the data as background for SHAP calculations.

    Parameters:
    config (dict): Configuration settings that include necessary file paths, column indices, and other 
                   parameters like batch size.
    directory_name (str): The base directory where processed datasets will be stored.
    size_of_background (int, optional): The size of the background dataset used for SHAP explanations.
                                        Default value is set to 200.

    Returns:
    tuple: A tuple containing:
           - full_dataset (tf.data.Dataset): A TensorFlow Dataset object containing the full dataset, 
             encoded and batched according to the specifications in the config.
           - background_data (np.ndarray): An array representing a subset of the full dataset, used as 
             the background dataset for SHAP explanations.
           - num_batches_needed (int): The total number of batches required to process the full dataset,
             calculated based on the batch size specified in the config.
    Example:
    config = {
        'full_file': 'path/to/dataset.csv',
        'batch_size': 32,
        'sequence_column_name': 'sequence',
        'group_column_name': 'group',
        'sep': ','
    }
    directory_name = 'path/to/results'
    full_dataset, background_data, num_batches_needed = prepare_for_shap(config, directory_name)
    """

    num_batches_needed_for_background = (size_of_background//config['batch_size']) + 1

    column_indices, longest_sequence, categories_list = get_column_indices_max_length_and_categories(directory_name, config)

    full_dataset = encode_from_csv(f"{directory_name}/created_datasets/original_file.tsv", config, column_indices, longest_sequence, categories_list, out_dir_for_scrambled_data=f'{directory_name}/created_datasets')

    full_dataset_length = sum(1 for _ in open(f"{directory_name}/created_datasets/original_file.tsv")) - 1

    num_batches_needed = (full_dataset_length//config['batch_size']) + 1

    print('Preparing the background data...')
    background_data, targets = dataset_to_numpy(full_dataset, num_batches_needed_for_background)
    if interaction_analysis:
        return full_dataset, background_data, targets, num_batches_needed
    return full_dataset, background_data, num_batches_needed

def integrated_gradients(model, input_sequence, num_steps=100):
    """
    Calculate the integrated gradients for a given input sequence and baseline.

    :param model: The trained Keras model
    :param input_sequence: The input sequence (one-hot encoded) for which to compute integrated gradients
    :param num_steps: The number of steps to use for the path integral approximation
    :return: Integrated gradients for the input sequence
    """
    
    baseline = np.zeros_like(input_sequence)
    
    diff = input_sequence - baseline

    interpolated_inputs = np.array([baseline + (float(i) / num_steps) * diff for i in range(num_steps + 1)])

    interpolated_inputs_tensor = tf.convert_to_tensor(interpolated_inputs, dtype=tf.float32)

    all_gradients = []
    
    for class_index in range(model.output_shape[-1]):
        with tf.GradientTape() as tape:
            tape.watch(interpolated_inputs_tensor)
            predictions = model(interpolated_inputs_tensor)
            target_prediction = predictions[:, class_index]

        gradients = tape.gradient(target_prediction, interpolated_inputs_tensor)
        
        avg_gradients = (gradients[:-1] + gradients[1:]) / 2.0
        
        integrated_gradients = tf.reduce_mean(avg_gradients, axis=0) * diff
        
        all_gradients.append(integrated_gradients)

    all_gradients = tf.stack(all_gradients, axis=-1)

    return all_gradients.numpy()

def compute_integrated_gradients(config, directory_name, size_of_background):

    full_dataset, background_data, num_batches_needed = prepare_for_shap(config, directory_name, size_of_background)
    if config['OPTIMISE']:
        model = load_model(f'{directory_name}/results/saved_trained_model_with_best_parameters/full_model.h5')
    else:
        model = load_model(f'{directory_name}/results/saved_trained_model/full_model.h5')
        

    all_integrated_gradients = []

    for i, batch in tqdm(enumerate(full_dataset.take(num_batches_needed)), total=num_batches_needed, desc='Explaining Batches of Samples'):

        features, labels = batch
        features = features.numpy()
        for num in range(features.shape[0]):
            integrated_gradients_array = integrated_gradients(model, features[num], num_steps=100)
            integrated_gradients_array = np.transpose(integrated_gradients_array, (2, 0, 1))
            integrated_gradients_array = integrated_gradients_array.reshape(2, 1, 3388, 4)
            all_integrated_gradients.append(integrated_gradients_array)


    all_integrated_gradients = np.concatenate(all_integrated_gradients, axis=1)
    return all_integrated_gradients

def compute_shaps(config, directory_name, size_of_background):
    """
    Computes SHAP values for a machine learning model using a prepared dataset. This function 
    facilitates the interpretation of the model by calculating the contribution of each feature 
    to predictions across the dataset.

    Parameters:
    - config (dict): Configuration dictionary containing model and dataset parameters.
    - directory_name (str): Directory where the model and results are stored.
    - size_of_background (int): Number of samples from the dataset used as background for 
                                the SHAP calculations.

    Returns:
    - np.ndarray: An array of SHAP values for each feature across all data points in the dataset.

    This function loads the model based on optimization settings, initializes a SHAP explainer, and 
    iterates over the dataset to compute SHAP values batch-wise. It accumulates these values and 
    returns a concatenated array of SHAP values.
    """
    full_dataset, background_data, num_batches_needed = prepare_for_shap(config, directory_name, size_of_background)
    if config['OPTIMISE']:
        model = load_model(f'{directory_name}/results/saved_trained_model_with_best_parameters/full_model.h5')
    else:
        model = load_model(f'{directory_name}/results/saved_trained_model/full_model.h5')
        

    explainer = shap.GradientExplainer(Model(inputs=model.inputs, outputs=model.output), background_data[:size_of_background])
    all_shap_values = []

    for i, batch in tqdm(enumerate(full_dataset.take(num_batches_needed)), total=num_batches_needed, desc='Explaining Batches of Samples'):
        

        features, labels = batch
        features = features.numpy()
        labels = labels.numpy()

        shap_values_batch = explainer.shap_values(features)

        shap_values_batch_array = np.stack(shap_values_batch)
        shap_values_batch_array_0 = shap_values_batch_array[:,:,:,0] * features
        shap_values_batch_array_1 = shap_values_batch_array[:,:,:,1] * features
        shap_values_batch_array = np.stack([shap_values_batch_array_0, shap_values_batch_array_1])
        all_shap_values.append(shap_values_batch_array)
        
    all_shap_values = np.concatenate(all_shap_values, axis=1)
    return all_shap_values

def add_shap_values_to_full_shuffled(all_shap_values, directory_name, config, model_explanation_out):

    full_shuffled = pd.read_csv(f'{directory_name}/created_datasets/original_file_shuffled.tsv', sep=config['sep'])
    unique_groups = sorted(full_shuffled[config['group_column_name']].unique())
    for i in range(all_shap_values.shape[0]):
        column_data = all_shap_values[i]

        full_shuffled[f'Contribution_Scores_for_Group_{unique_groups[i]}'] = list(column_data)
        
    def adjust_array_lengths(row):
        if config['sequence_column_name']:
            sequence_length = len(row[config['sequence_column_name']])
        else:
            sequence_length = row[config['signal_column_name'][0]].count(',') + 1
        for i in range(len(unique_groups)):
            key = f'Contribution_Scores_for_Group_{unique_groups[i]}'


            row[key] = row[key][:sequence_length]
        return row
    
    full_shuffled = full_shuffled.apply(adjust_array_lengths, axis=1)
    full_shuffled.to_pickle(f'{model_explanation_out}/original_file_shuffled_with_contribution_scores.pkl')


def create_model_explain_visualisation_dir(model_explanation_out):
    
    model_explanation_visualisation_out = f"{model_explanation_out}/visualisations"
    os.makedirs(model_explanation_visualisation_out, exist_ok=True)

    return model_explanation_visualisation_out

def create_k_mer_list(kmer_length, config):
    nucleotides = []
    
    if config.get("sequence_column_name", None) is not None:
        nucleotides.extend(['A', 'C', 'G', 'T'])
        
    score_column_names = config.get("signal_column_name") or []
    
    if kmer_length == 1 and config.get("signal_column_name", None) is not None:

        nucleotides.extend(score_column_names)
    
    kmers = [''.join(kmer) for kmer in product(nucleotides, repeat=kmer_length)]

    return nucleotides, kmers

def create_aggregate_heatmap_df_per_kmers(sequences, contribution_scores, explaining_params, model_explanation_out, config):
    print("Creating aggregate heatmap per kmers...")
    
    nucleotides, kmers = create_k_mer_list(explaining_params["kmer_length"], config)
    
    nucleotide_to_index = {nucleotide: idx for idx, nucleotide in enumerate(nucleotides)}

    heatmap_df = pd.DataFrame()
    
    for n, kmer in tqdm(enumerate(kmers), total=len(kmers), desc="Processing kmers"):
        
        relative_positions_all = []
        scores_all = []
        sequence_index = []

        for num, (seq, score) in tqdm(enumerate(zip(sequences, contribution_scores)), total=len(sequences), desc=f"Processing sequences for {kmer}"):

            for j in range(len(seq) - explaining_params["kmer_length"] + 1):

                if explaining_params["kmer_length"] == 1:
                    mean_score = score[j][nucleotide_to_index[kmer]]
                else:
                    mean_score = np.mean([score[j+i][nucleotide_to_index[kmer[i]]] for i in range(explaining_params["kmer_length"])])

                scores_all.append(mean_score)
                relative_positions_all.append(j/len(seq))
                sequence_index.append(num)

        if not scores_all:
            continue

        df = pd.DataFrame({
            'RelativePosition': relative_positions_all,
            f'Score_{kmer}': scores_all,
            'Seq_Index': sequence_index
        })

        df['Bin'] = pd.cut(df['RelativePosition'], bins=explaining_params["num_bins"], labels=False)

        df_mean_inside_seq = df.groupby(['Seq_Index', 'Bin'])[f'Score_{kmer}'].mean().reset_index()

        if heatmap_df.empty:
            heatmap_df = df_mean_inside_seq
        else:
            heatmap_df = pd.merge(heatmap_df, df_mean_inside_seq, on=['Seq_Index', 'Bin'])
    
    heatmap_df.to_pickle(f'{model_explanation_out}/heatmap_df_{explaining_params["kmer_length"]}mer.pkl')

def kmer_scores_df_to_array(heatmap_df, explaining_params):
    """
    Turn a dataframe to an array for ease of plotting.
    """
    kmers = [col.split('_')[1] for col in heatmap_df.columns if col.startswith('Score_')]
    num_bins = explaining_params["num_bins"]
    
    heatmap_data = np.zeros((len(kmers), num_bins))

    for n, kmer in enumerate(kmers):
        if f'Score_{kmer}' in heatmap_df.columns:  
            df_mean_scores = heatmap_df.groupby('Bin')[f'Score_{kmer}'].mean().reset_index()

            assert len(df_mean_scores) == num_bins , "Reduce number of bins"

            heatmap_data[n, df_mean_scores['Bin'].values] = df_mean_scores[f'Score_{kmer}'].values
            
    return heatmap_data

def plot_heatmap_final_plotting(heatmap_data, kmers, explaining_params, model_explanation_visualisation_out):
    absolute_maximum = np.nanmax(np.abs(heatmap_data))
    
    if explaining_params["kmer_length"] == 1:
        fig, ax = plt.subplots(figsize=(18, len(kmers)))
    else:
        fig, ax = plt.subplots(figsize=(18, 10))
        
    sns.heatmap(heatmap_data, cmap='seismic', cbar_kws={'label': 'Average Contribution Score'}, yticklabels=kmers, vmin=-absolute_maximum, vmax=absolute_maximum, ax=ax)
    ax.set_xlabel('Relative position', fontsize=16)
    ax.set_ylabel('Kmer', fontsize=16)

    x_ticks = np.linspace(0, explaining_params["num_bins"], 11) 
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{i*10}%' for i in range(11)])

    plt.title(f'Metaprofile of Average Importance Scores for Each {explaining_params["kmer_length"]}-mer for {explaining_params["group_to_visualise"]} predicted genes', fontsize=18)

    plt.savefig(f'{model_explanation_visualisation_out}/{explaining_params["group_to_visualise"]}_group_importance_score_binned_to_{explaining_params["num_bins"]}_bins_metaplot_for_{explaining_params["kmer_length"]}mer.pdf', bbox_inches='tight')
    plt.show()

def plot_clustermap_final_plotting(heatmap_data, kmers, explaining_params, model_explanation_visualisation_out):
    absolute_maximum = np.nanmax(np.abs(heatmap_data))
    
    clustermap = sns.clustermap(heatmap_data, col_cluster=False, cmap='seismic', cbar_kws={'label': 'Average Contribution Score'}, yticklabels=kmers, vmin=-absolute_maximum, vmax=absolute_maximum)

    clustermap.ax_heatmap.set_xlabel('Relative position')
    clustermap.ax_heatmap.set_ylabel('Kmer')

    x_ticks = np.linspace(0, explaining_params["num_bins"], 11)
    clustermap.ax_heatmap.set_xticks(x_ticks)
    clustermap.ax_heatmap.set_xticklabels([f'{i*10}%' for i in range(11)])

    clustermap.fig.suptitle(f'Metaprofile of Average Importance Scores for Each {explaining_params["kmer_length"]}-mer for {explaining_params["group_to_visualise"]} predicted genes')

    plt.savefig(f'{model_explanation_visualisation_out}/{explaining_params["group_to_visualise"]}_group_importance_score_binned_to_{explaining_params["num_bins"]}_bins_clustermap_for_{explaining_params["kmer_length"]}mer.pdf', bbox_inches='tight')
    plt.show()

class TextHandler(HandlerBase):
    def create_artists(self, legend, text ,xdescent, ydescent,
                        width, height, fontsize, trans):
        tx = plt.Text(width/2.,height/2, text.get_text(), fontsize=fontsize,
                    ha="center", va="center", color=text.get_color())
        return [tx]

def plot_lineplot_final_plotting(heatmap_data, explaining_params, model_explanation_visualisation_out, kmers, top_bottom=10):
    
    kmer_importance = np.nansum(heatmap_data, axis=1)
    sorted_indices = np.argsort(kmer_importance)
    if len(kmers) > 2 * top_bottom:
        specified_kmers = [kmers[i] for i in sorted_indices[:top_bottom]] + [kmers[i] for i in sorted_indices[-top_bottom:]]
    else:
        specified_kmers = kmers
    
    color_map = cm.get_cmap('rainbow', len(specified_kmers))

    plt.figure(figsize=(18, 8))

    for i, kmer in enumerate(kmers):
        if kmer not in specified_kmers:
            plt.plot(heatmap_data[i], color='gray', alpha=0.4)

    legend_texts = []
    for i, kmer in enumerate(specified_kmers):
        if kmer in kmers:
            index = kmers.index(kmer)
            color = color_map(i)
            plt.plot(heatmap_data[index], color=color)
            legend_texts.append(plt.Text(0, 0, kmer, color=color))
            
    plt.xlabel('Relative Positions', fontsize=18)
    plt.ylabel('Importance Scores', fontsize=18)
    
    x_ticks = np.linspace(0, heatmap_data.shape[1]-1, 11)  # 11 points for 0%, 10%, ..., 100%
    plt.xticks(x_ticks, [f'{int(i*100/(heatmap_data.shape[1]-1))}%' for i in x_ticks])

    plt.title('Importance Scores for each k-mer', fontsize=20)

    plt.legend(handles=legend_texts, frameon=False, handler_map={plt.Text: TextHandler()}, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    
    plt.savefig(f'{model_explanation_visualisation_out}/{explaining_params["group_to_visualise"]}_group_importance_score_binned_to_{explaining_params["num_bins"]}_bins_lineplot_for_{explaining_params["kmer_length"]}mer.pdf', bbox_inches='tight')

def plot_a_heatmap_of_importance_scores(heatmap_df, model_explanation_visualisation_out, explaining_params, config):
    """
    Plot a heatmap and clustermap of binned importance scores over a group.
    """    
    nucleotides, kmers = create_k_mer_list(explaining_params["kmer_length"], config)
    
    heatmap_data = kmer_scores_df_to_array(heatmap_df, explaining_params)
    
    print("Plotting a heatmap of importance scores...")
    plot_heatmap_final_plotting(heatmap_data, kmers, explaining_params, model_explanation_visualisation_out)
    
    print("Plotting a clustermap of importance scores...")
    plot_clustermap_final_plotting(heatmap_data, kmers, explaining_params, model_explanation_visualisation_out)
    
    print("Plotting a lineplot of importance scores...")
    plot_lineplot_final_plotting(heatmap_data, explaining_params, model_explanation_visualisation_out, kmers)
    
def dimensionality_reduction(pivot_df, explaining_params):
    """
    Find the best dimensionality reduction technique for your binned importances scores. 
    This is objectivly an overkill and was done for testing purposes.
    """
    if explaining_params["type_of_reduction"] == 'TSNE':
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
        data_reduced = tsne.fit_transform(pivot_df)
    if explaining_params["type_of_reduction"] == 'UMAP':
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        data_reduced = reducer.fit_transform(pivot_df)
    if explaining_params["type_of_reduction"] == 'PCA':
        pca = PCA(n_components=2)
        data_reduced = pca.fit_transform(pivot_df)
    if explaining_params["type_of_reduction"] == 'Isomap':
        isomap = Isomap(n_components=2)
        data_reduced = isomap.fit_transform(pivot_df)
    if explaining_params["type_of_reduction"] == 'Multidimensional scaling':
        mds = MDS(n_components=2)
        data_reduced = mds.fit_transform(pivot_df)
    if explaining_params["type_of_reduction"] == 'Spectral Embedding':
        spectral_embedding = SpectralEmbedding(n_components=2)
        data_reduced = spectral_embedding.fit_transform(pivot_df)
    if explaining_params["type_of_reduction"] == 'Factor Analysis':
        factor_analysis = FactorAnalysis(n_components=2)
        data_reduced = factor_analysis.fit_transform(pivot_df)
    if explaining_params["type_of_reduction"] == 'Dictionary Learning':
        dict_learning = DictionaryLearning(n_components=2)
        data_reduced = dict_learning.fit_transform(pivot_df)
        
    return data_reduced


def plot_clustered_heatmap_of_importance_scores(heatmap_df, model_explanation_visualisation_out, explaining_params, config):
    """
    Cluster the sequences based on their binned importance scores. 
    This is hepful to see possible subgroups of models predictions.
    """
    print("Plotting...")
    
    nucleotides, kmers = create_k_mer_list(explaining_params["kmer_length"], config)
    
    features = pd.DataFrame(heatmap_df.set_index(['Seq_Index', 'Bin']).stack()).reset_index()
    
    features.columns = ['Seq_Index', 'Bin', 'Nucleotide', 'Score']
    
    pivot_df = features.pivot_table(index='Seq_Index', columns=['Bin', 'Nucleotide'], values='Score').fillna(0)
    
    data_reduced = dimensionality_reduction(pivot_df, explaining_params)
    
    clusters_num = explaining_params["number_of_clusters"]
    print("Clustering...", clusters_num)
    aggclust = AgglomerativeClustering(n_clusters=clusters_num, distance_threshold=None, affinity='euclidean', linkage='ward')
    clusters = aggclust.fit_predict(data_reduced)

    pivot_df['Cluster'] = clusters

    plt.figure(figsize=(12, 10))
        
    sns.scatterplot(x=data_reduced[:, 0], y=data_reduced[:, 1], hue=clusters, palette="tab10", legend="full", s=50)

    plt.title(f'{explaining_params["type_of_reduction"]} projection of the dataset, colored by AgglomerativeClustering clusters', fontsize=16)
    plt.xlabel(f'{explaining_params["type_of_reduction"]} Dimension 1', fontsize=14)
    plt.ylabel(f'{explaining_params["type_of_reduction"]} Dimension 2', fontsize=14)
    plt.legend(title='Cluster')

    plt.legend(title='Cluster').get_title().set_fontsize('14')

    plt.savefig(f'{model_explanation_visualisation_out}/{explaining_params["type_of_reduction"]}_projection_of_{explaining_params["group_to_visualise"]}_group_importance_score_binned_to_{explaining_params["num_bins"]}_bins_for_{explaining_params["kmer_length"]}mer_clustered_into_{clusters_num}_clusters.pdf', bbox_inches='tight')
    
    

    pivot_df_info = pivot_df.reset_index()[['Seq_Index', 'Cluster']]
    cluster_series = pivot_df_info.iloc[:, -1]
    clusters_info = pd.DataFrame({
        'Seq_Index': cluster_series.index,
        'Cluster': cluster_series.values
    })
    heatmap_df_with_clusters = pd.merge(heatmap_df, clusters_info, on='Seq_Index')
        
    graph_y_axis = min((10*clusters_num), clusters_num * (len(kmers)))
    fig, axs = plt.subplots(clusters_num, 1, figsize=(18, graph_y_axis))
    fig.suptitle(f'Metaprofile of Average Importance Scores for Each {explaining_params["kmer_length"]}-mer for {explaining_params["group_to_visualise"]} predicted genes', fontsize=10)
    
    if clusters_num == 1:
        axs = [axs]
        
    absolute_maximum = 0    
    
    for cluster in range(clusters_num):
        cluster_data = heatmap_df_with_clusters[heatmap_df_with_clusters['Cluster'] == cluster]
        
        heatmap_data = kmer_scores_df_to_array(cluster_data, explaining_params)

        absolute_maximum = max(np.nanmax(np.abs(heatmap_data)), absolute_maximum)    
        
    for cluster in range(clusters_num):
        cluster_data = heatmap_df_with_clusters[heatmap_df_with_clusters['Cluster'] == cluster]
        
        heatmap_data = kmer_scores_df_to_array(cluster_data, explaining_params)
            
        ax = axs[cluster] 
        sns.heatmap(heatmap_data, cmap='seismic', cbar_kws={'label': 'Average Contribution Score'}, yticklabels=kmers, vmin=-absolute_maximum, vmax=absolute_maximum, ax=ax)
        ax.set_title(f'Cluster {cluster}')
        ax.set_xlabel('Relative position')
        ax.set_ylabel('Kmer')
        
        x_ticks = np.linspace(0, explaining_params["num_bins"], 11) 
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{i*10}%' for i in range(11)])

    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    fig.suptitle(f'Metaprofile of Average Importance Scores for Each {explaining_params["kmer_length"]}-mer for {explaining_params["group_to_visualise"]} predicted genes', y=1.02)


    plt.savefig(f'{model_explanation_visualisation_out}/Metaprofile_of_{explaining_params["type_of_reduction"]}_projection_of_{explaining_params["group_to_visualise"]}_group_importance_score_binned_to_{explaining_params["num_bins"]}_bins_for_{explaining_params["kmer_length"]}mer_clustered_into_{clusters_num}_clusters.pdf', bbox_inches='tight')

def create_heatmap(explaining_params):
    """
    Main code to create the binned importance scores data.
    """

    directory_name = f'{explaining_params["run_dir"]}'
    config = import_config(directory_name)
    model_explanation_out = f"{directory_name}/results/model_explanation"
    
    if not os.path.exists(f'{model_explanation_out}/original_file_shuffled_with_contribution_scores.pkl'):
        raise ValueError(f'{model_explanation_out}/original_file_shuffled_with_contribution_scores.pkl is prerequisite for the plotting.')
    
    full_shuffled_with_contribution_scores = pd.read_pickle(f'{model_explanation_out}/original_file_shuffled_with_contribution_scores.pkl')
    
    subset_group_shuffled_with_contribution_scores = full_shuffled_with_contribution_scores[full_shuffled_with_contribution_scores[config["group_column_name"]] == explaining_params["group_to_visualise"]]
    
    if not subset_group_shuffled_with_contribution_scores[config["group_column_name"]].isin([explaining_params["group_to_visualise"]]).any():
        raise ValueError(f'{explaining_params["group_to_visualise"]} is not a value in the {config["group_column_name"]} column.')
    
    sequences = subset_group_shuffled_with_contribution_scores[config["sequence_column_name"]].tolist()
    contribution_scores = subset_group_shuffled_with_contribution_scores[f'Contribution_Scores_for_Group_{explaining_params["group_to_visualise"]}'].tolist()
    create_aggregate_heatmap_df_per_kmers(sequences, contribution_scores, explaining_params, model_explanation_out, config)
        
def plot_heatmap(explaining_params):
    directory_name = f'{explaining_params["run_dir"]}'
    config = import_config(directory_name)
    check_column_config(explaining_params, config)
    model_explanation_out = f"{directory_name}/results/model_explanation"
    model_explanation_visualisation_out = create_model_explain_visualisation_dir(model_explanation_out)

    if not os.path.exists(f'{model_explanation_out}/heatmap_df_{explaining_params["kmer_length"]}mer.pkl'):
        raise ValueError(f'{model_explanation_out}/heatmap_df_{explaining_params["kmer_length"]}mer.pkl is prerequisite for the plotting.')
    
    heatmap_df = pd.read_pickle(f'{model_explanation_out}/heatmap_df_{explaining_params["kmer_length"]}mer.pkl')
    
    if explaining_params["PLOT_HEATMAP_AND_CLUSTERMAP"]:
        model_explanation_visualisation_heatmap_out = f"{model_explanation_visualisation_out}/heatmap_clustermap_lineplot"
        os.makedirs(model_explanation_visualisation_heatmap_out, exist_ok=True)
        plot_a_heatmap_of_importance_scores(heatmap_df, model_explanation_visualisation_heatmap_out, explaining_params, config)
    if explaining_params["PLOT_SEQUENCE_CLUSTERED_BY_BIN_VALUES"]:
        model_explanation_visualisation_heatmap_out = f"{model_explanation_visualisation_out}/clustered_heatmap"
        os.makedirs(model_explanation_visualisation_heatmap_out, exist_ok=True)
        plot_clustered_heatmap_of_importance_scores(heatmap_df, model_explanation_visualisation_heatmap_out, explaining_params, config)

def check_column_config(explaining_params, config):

    if explaining_params["kmer_length"] > 1:
        if config.get("sequence_column_name", None) is None:
            raise ValueError("There should be a sequence column present if the kmer_length is above 1.")
        if config.get("signal_column_name", None) is not None:
            raise ValueError("There should not be any score columns if the kmer_length is above 1.")
    elif explaining_params["kmer_length"] == 1:
        if config.get("sequence_column_name", None) is None and config.get("signal_column_name", None) is None:
            raise ValueError("There must be at least one sequence or score column if the kmer_length is 1.")


def draw_importance_scores_values(shap_values, index, model_explanation_visualisation_out, explaining_params, config):
    sequence_column_name = config.get("sequence_column_name")
    signal_column_names = config.get("signal_column_name") or []
    
    colors = ['blue', 'orange', 'green', 'red']
    nucleotide_mapping = ['A', 'C', 'G', 'U']
    

    absolute_max_value = np.percentile(np.abs(shap_values), 99.9)

    fig = plt.figure(figsize=(16, 4+len(signal_column_names)))
    gs = gridspec.GridSpec(2, 2, width_ratios=[50, 1], height_ratios=[4, len(signal_column_names)])


    if sequence_column_name is not None:
        ax_sequence = plt.subplot(gs[0, 0])
        for idx in range(4):
            ax_sequence.bar(range(shap_values.shape[0]), shap_values[:, idx], color=colors[idx], label=f'{nucleotide_mapping[idx]}', alpha=1)
        
        ax_sequence.set_xlim(0, shap_values.shape[0])
        ax_sequence.set_ylim(-absolute_max_value, absolute_max_value) 
        ax_sequence.set_xlabel('Sequence position', fontsize=16)
        ax_sequence.set_ylabel('Importance Score', fontsize=16)
        ax_sequence.set_title(f'Importance Scores for {explaining_params["group_to_visualise"]} group over sequence with index {explaining_params["sequence_indexes"][index]}', fontsize=16)
        ax_sequence.legend(nucleotide_mapping, loc='upper left')


    if signal_column_names:
        ax_heatmap = plt.subplot(gs[1, 0])
        ax_cbar = plt.subplot(gs[1, 1])
        num_signals = len(signal_column_names)
        signal_data = shap_values[:, 4:4+num_signals]

        sns.heatmap(signal_data.T, cmap='seismic', cbar=True, ax=ax_heatmap, cbar_ax=ax_cbar, yticklabels=signal_column_names, annot=False, vmin=-absolute_max_value, vmax=absolute_max_value)

        ax_heatmap.set_xlabel('Sequence position', fontsize=16)
        ax_heatmap.set_ylabel('Signal Columns', fontsize=16)
        ax_heatmap.set_title(f'Signal Values for {explaining_params["group_to_visualise"]} group over sequence with index {explaining_params["sequence_indexes"][index]}', fontsize=16)
        ax_heatmap.set_yticklabels(signal_column_names, rotation=0, fontsize=13)

    plt.subplots_adjust(hspace=0.5, wspace=0.2)
    plt.tight_layout()
    plt.savefig(f'{model_explanation_visualisation_out}/importance_scores_for_{explaining_params["group_to_visualise"]}_group_over_sequence_with_index_{explaining_params["sequence_indexes"][index]}.pdf', bbox_inches='tight')


def plot_individual_sequence_importance_scores(explaining_params):
    print("Plotting Individual Sequences...")
    directory_name = f'{explaining_params["run_dir"]}'
    config = import_config(directory_name)
    model_explanation_out = f"{directory_name}/results/model_explanation"
    model_explanation_visualisation_out = create_model_explain_visualisation_dir(model_explanation_out)
    model_explanation_visualisation_individual_sequences_out = f"{model_explanation_visualisation_out}/importance_scores_over_sequences"
    os.makedirs(model_explanation_visualisation_individual_sequences_out, exist_ok=True)

    if not os.path.exists(f'{model_explanation_out}/original_file_shuffled_with_contribution_scores.pkl'):
        raise ValueError(f'{model_explanation_out}/original_file_shuffled_with_contribution_scores.pkl is prerequisite for the plotting.')
    
    full_shuffled_with_contribution_scores = pd.read_pickle(f'{model_explanation_out}/original_file_shuffled_with_contribution_scores.pkl')

    arrays_in_index_rows = full_shuffled_with_contribution_scores.loc[explaining_params["sequence_indexes"], f'Contribution_Scores_for_Group_{explaining_params["group_to_visualise"]}'].tolist()

    for index, array_in_index_row in enumerate(arrays_in_index_rows):
        draw_importance_scores_values(array_in_index_row, index, model_explanation_visualisation_individual_sequences_out, explaining_params, config)

def run_modisco_lite(ohe_file, attr_file, explaining_params, output_path):
    
    print("Starting TF-MoDISco execution...")
    
    output_path_raw = f"{output_path}/Raw_Output"
    os.makedirs(output_path_raw, exist_ok=True)  
      

    output_file = f'{output_path_raw}/modisco_results_{explaining_params["group_to_visualise"]}_{explaining_params["max_seqlets"]}_seqlets_{explaining_params["n_leiden"]}_leiden_{explaining_params["window"]}_window.h5'
    max_seqlets = str(explaining_params["max_seqlets"])
    n_leiden = str(explaining_params["n_leiden"])
    window = str(explaining_params["window"])

    if os.path.exists(output_file):
        return output_file
    
    command = [
        'modisco', 'motifs',
        '-s', ohe_file,
        '-a', attr_file,
        '-n', max_seqlets,
        '-l', n_leiden,
        '-w', window,
        '-o', output_file
    ]


    result = subprocess.run(command, capture_output=True, text=True)


    if result.returncode == 0:
        print("TF-MoDISco execution successful.")
    else:
        print(f"TF-MoDISco execution failed with error: {result.stderr}")
        print(f"If the error is due to NaN weights, this is usually because the peak is near the ends of the seuqence and the window size is too large so it spans over the edge. Try reducing the window size incrementaly.")
    
    return output_file
        
def visualise_modisco_lite(output_file, output_path, explaining_params):
    
    print("Starting TF-MoDISco visualisation...")
    
    output_path_visualised = f'{output_path}/visualised_output/{explaining_params["group_to_visualise"]}_{explaining_params["max_seqlets"]}_seqlets_{explaining_params["n_leiden"]}_leiden_{explaining_params["window"]}_window'
    os.makedirs(output_path_visualised, exist_ok=True)  
    

    input_file = output_file
    output_dir = output_path_visualised
    
    if os.path.exists(f'{output_dir}/motifs.html'):
        return 


    command = [
        'modisco', 'report',
        '-i', input_file,
        '-o', output_dir,
        '-s', output_dir
    ]


    result = subprocess.run(command, capture_output=True, text=True)


    if result.returncode == 0:
        print("TF-MoDISco report generation successful.")
    else:
        print(f"TF-MoDISco report generation failed with error: {result.stderr}")
        
    return 

def extract_motifs(explaining_params):
    directory_name = f'{explaining_params["run_dir"]}'
    config = import_config(directory_name)
    
    if config.get("sequence_column_name", None) is None:
        raise ValueError("There must be a sequence column and present as the motifs will be extracted from the sequence importance scorees only.")
    
    model_explanation_out = f"{directory_name}/results/model_explanation"
    model_explanation_visualisation_out = create_model_explain_visualisation_dir(model_explanation_out)
    model_explanation_visualisation_modisco_out = f"{model_explanation_visualisation_out}/TF_Modisco_Motif_Finder"
    os.makedirs(model_explanation_visualisation_modisco_out, exist_ok=True)
    
    modisco_input_path = f"{model_explanation_visualisation_modisco_out}/TF_Modisco_Input"
    os.makedirs(modisco_input_path, exist_ok=True)
    
    modisco_one_hot_encoded_input_path = f"{modisco_input_path}/onehot_data_for_modisco_{explaining_params['group_to_visualise']}.npy"
    modisco_importance_scores_input_path = f"{modisco_input_path}/importance_scores_array_for_modisco_{explaining_params['group_to_visualise']}.npy"
    
    original_file_shuffled_with_contribution_scores_path = f'{model_explanation_out}/original_file_shuffled_with_contribution_scores.pkl'

    if not os.path.exists(original_file_shuffled_with_contribution_scores_path):
        raise ValueError(f'{original_file_shuffled_with_contribution_scores_path} is prerequisite for the plotting.')
    
    full_shuffled_with_contribution_scores = pd.read_pickle(original_file_shuffled_with_contribution_scores_path)
    full_shuffled_with_contribution_scores = full_shuffled_with_contribution_scores[full_shuffled_with_contribution_scores[config["group_column_name"]] == explaining_params["group_to_visualise"]]
    

    importance_scores = full_shuffled_with_contribution_scores[f'Contribution_Scores_for_Group_{explaining_params["group_to_visualise"]}'].tolist()
    
    max_length = max(len(seq) for seq in importance_scores)
    
    importance_scores = pad_sequences(importance_scores, maxlen=max_length, dtype='float32', padding='post')
    importance_scores_array = np.array(importance_scores)
    

    importance_scores_array = importance_scores_array[:, :, :4]
    
    full_shuffled_with_contribution_scores['padded_onehot'] = full_shuffled_with_contribution_scores[config["sequence_column_name"]].apply(lambda x: one_hot_encode(x, max_length))
    onehot_data_array = np.stack(full_shuffled_with_contribution_scores['padded_onehot'].values)
    

    if not os.path.exists(modisco_one_hot_encoded_input_path):
        print("Preparing the onehot data for modisco...")
        onehot_data_for_modisco = np.transpose(onehot_data_array, (0, 2, 1))
        np.save(modisco_one_hot_encoded_input_path, onehot_data_for_modisco)
    
    if not os.path.exists(modisco_importance_scores_input_path):    
        print("Preparing the importance scores for modisco...")
        importance_scores_array_for_modisco = np.transpose(importance_scores_array, (0, 2, 1))
        np.save(modisco_importance_scores_input_path, importance_scores_array_for_modisco)

    output_file = run_modisco_lite(modisco_one_hot_encoded_input_path, modisco_importance_scores_input_path, explaining_params, model_explanation_visualisation_modisco_out)
    visualise_modisco_lite(output_file, model_explanation_visualisation_modisco_out, explaining_params)

def add_predicitons(explaining_params):
    directory_name = f'{explaining_params["run_dir"]}'
    config = import_config(directory_name)

    column_indices, longest_sequence, categories_list = get_column_indices_max_length_and_categories(directory_name, config)

    full_dataset = encode_from_csv(f"{directory_name}/created_datasets/original_file.tsv", config, column_indices, longest_sequence, categories_list)

    full_dataset_length = sum(1 for _ in open(f"{directory_name}/created_datasets/original_file.tsv")) - 1
        

    num_batches_needed = (full_dataset_length // config['batch_size']) + 1
    
    if config['OPTIMISE']:
        model = load_model(f'{directory_name}/results/saved_trained_model_with_best_parameters/full_model.h5')
    else:
        model = load_model(f'{directory_name}/results/saved_trained_model/full_model.h5')


    predictions = model.predict(full_dataset, steps=num_batches_needed)
    print("Predictions completed.")
    

    original_data = pd.read_csv(f"{directory_name}/created_datasets/original_file.tsv", sep=config['sep'])


    categories_list = [cat.decode("utf-8") if isinstance(cat, bytes) else cat for cat in categories_list.numpy()]


    for i, category in enumerate(categories_list):
        original_data[f'Prediction_{category}'] = predictions[:len(original_data), i]
        
    if config['OPTIMISE']:
        original_data.to_csv(f'{directory_name}/results/evaluated_trained_model_with_best_parameters/original_file_with_predictions.tsv',
                             sep=config['sep'], index=False)
    else:
        original_data.to_csv(f'{directory_name}/results/evaluated_trained_model/original_file_with_predictions.tsv',
                         sep=config['sep'], index=False)

    
def plot_confusion_matrix(true_labels, predicted_labels, categories_list, explaining_params, directory_name, accuracy, auroc, dataset_type, config):
    

    confusion_matrix = tf.math.confusion_matrix(true_labels, tf.argmax(predicted_labels, axis=1), num_classes=len(categories_list))
    confusion_matrix = confusion_matrix / tf.reduce_sum(confusion_matrix, axis=1)[:, None]
    

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt=".2%", cmap='Blues', xticklabels=categories_list, yticklabels=categories_list)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for the {dataset_type} Dataset')


    plt.text(len(categories_list) - 0.5, -0.5, f'Accuracy: {accuracy:.2f}\nAUROC: {auroc:.2f}', 
             horizontalalignment='center', verticalalignment='center', color='red', fontsize=12, fontweight='bold')

    if config['OPTIMISE']:
        plt.savefig(f'{directory_name}/results/evaluated_trained_model_with_best_parameters/confusion_matrix_{dataset_type}_dataset.png', dpi=600, bbox_inches='tight')
    else:
        plt.savefig(f'{directory_name}/results/evaluated_trained_model/confusion_matrix_{dataset_type}_dataset.png', dpi=600, bbox_inches='tight')

def plot_confusion_matrix_with_stats_on_testing(explaining_params, dataset_type):
    directory_name = f'{explaining_params["run_dir"]}'
    config = import_config(directory_name)


    if config['OPTIMISE']:
        original_file_with_predictions = pd.read_csv(f'{directory_name}/results/evaluated_trained_model_with_best_parameters/original_file_with_predictions.tsv', sep=config['sep'])
    else:
        original_file_with_predictions = pd.read_csv(f'{directory_name}/results/evaluated_trained_model/original_file_with_predictions.tsv', sep=config['sep'])
    
    if dataset_type == "testing_validation":
        testing_dataset = pd.read_csv(f'{directory_name}/created_datasets/testing.tsv', sep=config['sep'])
        validation_dataset = pd.read_csv(f'{directory_name}/created_datasets/validation.tsv', sep=config['sep'])

        testing_dataset = pd.concat([testing_dataset, validation_dataset])
    else:
        testing_dataset = pd.read_csv(f'{directory_name}/created_datasets/{dataset_type}.tsv', sep=config['sep'])
    
    testing_dataset_with_predictions = original_file_with_predictions[original_file_with_predictions["index_column"].isin(testing_dataset["index_column"])]
        

    column_indices, longest_sequence, categories_list = get_column_indices_max_length_and_categories(directory_name, config)


    categories_list = [cat.decode("utf-8") if isinstance(cat, bytes) else cat for cat in categories_list.numpy()]


    true_labels = testing_dataset_with_predictions[config['group_column_name']].astype(str)   
    
    predicted_labels = testing_dataset_with_predictions[[f'Prediction_{category}' for category in categories_list]]


    category_to_binary = {category: idx for idx, category in enumerate(categories_list)}
    
    true_labels_binary = true_labels.map(category_to_binary)


    accuracy = accuracy_score(true_labels_binary, tf.argmax(predicted_labels, axis=1))



    true_labels_one_hot = tf.keras.utils.to_categorical(true_labels_binary, num_classes=len(categories_list))
    auroc = roc_auc_score(true_labels_one_hot, predicted_labels, multi_class='ovr')
        

    plot_confusion_matrix(true_labels_binary, predicted_labels, categories_list, explaining_params, directory_name, accuracy, auroc, dataset_type, config)
    
def final_evaluation_on_testing(explaining_params):
    plot_confusion_matrix_with_stats_on_testing(explaining_params, "training")
    plot_confusion_matrix_with_stats_on_testing(explaining_params, "testing")
    plot_confusion_matrix_with_stats_on_testing(explaining_params, "validation")
    plot_confusion_matrix_with_stats_on_testing(explaining_params, "testing_validation")

def explain_run(explaining_params):
    directory_name = f'{explaining_params["run_dir"]}'
    model_explanation_out = create_model_explain_out_dir(directory_name)
    config = import_config(directory_name)
    all_shap_values = compute_shaps(config, directory_name, explaining_params["num_background_samples"])
    add_shap_values_to_full_shuffled(all_shap_values, directory_name, config, model_explanation_out)

def prepare_data_for_interaction_analysis(config, directory_name, sequence_indexes_corr):
    max_sequence_indexes_corr = max(sequence_indexes_corr)
    full_dataset, sample_data, targets,  num_batches_needed = prepare_for_shap(config, directory_name, max_sequence_indexes_corr, interaction_analysis=True)
    sample_data = [sample_data[i] for i in sequence_indexes_corr]
    targets = [targets[i] for i in sequence_indexes_corr]
    return tf.stack(sample_data), tf.stack(targets)

def mutate_base(one_hot_base):
    bases = ['A', 'C', 'G', 'T']
    current_base_index = np.argmax(one_hot_base)
    possible_bases = [i for i in range(4) if i != current_base_index]
    new_base_index = np.random.choice(possible_bases)
    
    new_one_hot_base = [0, 0, 0, 0]
    new_one_hot_base[new_base_index] = 1
    return new_one_hot_base

def generate_mutated_samples(original_sequence, unpadded_length, num_samples=200):
    mutated_samples = []
    for _ in range(num_samples):
        mutated_sample = np.copy(original_sequence)

        num_mutations = np.random.randint(1, unpadded_length)
        mutation_indices = np.random.choice(unpadded_length, num_mutations, replace=False)
        
        for idx in mutation_indices:
            mutated_sample[0, idx, :] = mutate_base(mutated_sample[0, idx, :])
        
        mutated_samples.append(mutated_sample)
    
    return np.array(mutated_samples)


def compute_gradients(model, inputs, targets):
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        predictions = model(inputs, training=False)
        loss = loss_fn(targets, predictions)
    gradients = tape.gradient(loss, inputs)
    return gradients.numpy()

def compute_gradients_for_samples(model, samples, targets):
    gradients = []
    for i, sample in tqdm(enumerate(samples), total=len(samples), desc='Computing gradients for Samples:'):
        grad = compute_gradients(model, tf.convert_to_tensor(sample), tf.convert_to_tensor(targets))
        grad = grad * sample
        gradients.append(grad)
    return np.array(gradients)

def gradients_for_corr(model, baseline, input, targets, num_mutated_samples=200):

    unpadded_length = get_unpadded_length(input)

    mutated_samples = generate_mutated_samples(input, unpadded_length, num_samples=num_mutated_samples)

    gradients = compute_gradients_for_samples(model, mutated_samples, targets)
        

    return gradients

def calculate_correlation_worker(args):
    i, num_positions, gradients = args
    correlation_row = np.zeros(num_positions)
    grad_i = gradients[:, i, :].flatten()
    for j in range(num_positions):
        grad_j = gradients[:, j, :].flatten()
        correlation_row[j], _ = spearmanr(grad_i, grad_j)
    return i, correlation_row

def add_random_epsilon_to_gradients(gradients, stddev):
    random_noise = np.random.normal(0, stddev, gradients.shape)
    return gradients + random_noise

def calculate_correlation_matrix(gradients):
    gradients = add_random_epsilon_to_gradients(gradients, 1e-14)
    num_positions = gradients.shape[1]
    correlation_matrix = np.zeros((num_positions, num_positions))
    with Pool(cpu_count()) as pool:
        args = [(i, num_positions, gradients) for i in range(num_positions)]
        for i, correlation_row in tqdm(pool.imap(calculate_correlation_worker, args), total=num_positions, desc="Calculating correlations"):
            correlation_matrix[i, :] = correlation_row
            
    print("Correlation matrix computed.", correlation_matrix.shape)
    return correlation_matrix


def interaction_calculation(config, directory_name, sample_data, targets, model_explanation_out, explaining_params):
    print("Calculating interaction scores...")
    
    if config['OPTIMISE']:
        model = load_model(f'{directory_name}/results/saved_trained_model_with_best_parameters/full_model.h5')
    else:
        model = load_model(f'{directory_name}/results/saved_trained_model/full_model.h5')
        

    baseline = np.zeros((1, sample_data.shape[1], sample_data.shape[2]))


    num_samples = sample_data.shape[0]
    sequence_length = sample_data.shape[1]


    correlation_matrices = np.zeros((num_samples, sequence_length, sequence_length))

    for i in tqdm(range(num_samples), total=num_samples, desc="Processing sample:"):
        
        input_sequence = np.expand_dims(sample_data[i], axis=0)
        input_target = np.expand_dims(targets[i], axis=0)
        gradients = gradients_for_corr(model, baseline, input_sequence, input_target)
        gradients = gradients[:, 0, :, :]
        correlation_matrix = calculate_correlation_matrix(gradients)
        correlation_matrices[i] = correlation_matrix
   
    np.savez(f'{model_explanation_out}/correlation_matrices.npz', correlation_matrices=correlation_matrices)

def interaction_analysis(explaining_params):
    directory_name = f'{explaining_params["run_dir"]}'
    model_explanation_out = create_model_explain_out_dir(directory_name)
    config = import_config(directory_name)

    sample_data, targets = prepare_data_for_interaction_analysis(config, directory_name, explaining_params["sequence_indexes_corr"])
    
    interaction_calculation(config, directory_name, sample_data, targets, model_explanation_out, explaining_params)
    
def subset_correlation_matrix(correlation_matrix, length):

    return correlation_matrix[:length, :length]

def get_unpadded_length(sequence):
    sequence = sequence[0]
    
    unpadded_length = len(sequence)
    for i in range(len(sequence) - 1, -1, -1):
        if not np.array_equal(sequence[i], [-1, -1, -1, -1]):
            unpadded_length = i + 1
            break
    return unpadded_length

def plot_correlation_heatmap(model_visualisation_out, matrix, title):
    correlation_matrices_out = f'{model_visualisation_out}/correlation_matrices'
    os.makedirs(correlation_matrices_out, exist_ok=True)
    
    absolute_max = np.abs(matrix).max()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap='seismic', annot=False, vmin=-absolute_max, vmax=absolute_max)
    plt.title(title)
    plt.xlabel('Sequence')
    plt.ylabel('Sequence')
    
    plt.savefig(f'{correlation_matrices_out}/{title}.png', dpi=600, bbox_inches='tight')

        
def plot_interaction_analysis(explaining_params):
    directory_name = f'{explaining_params["run_dir"]}'
    model_explanation_out = f"{directory_name}/results/model_explanation"
    config = import_config(directory_name)
    model_visualisation_out = create_model_explain_visualisation_dir(model_explanation_out)
    sample_data, targets = prepare_data_for_interaction_analysis(config, directory_name, explaining_params["sequence_indexes_corr"])
    correlation_matrices = np.load(f'{model_explanation_out}/correlation_matrices.npz')['correlation_matrices']
    
    for i in tqdm(range(sample_data.shape[0]), total=sample_data.shape[0], desc="Plotting sample:"):
        
        sequence = sample_data[i]
        correlation_matrix = correlation_matrices[i]
        
        unpadded_length = get_unpadded_length(np.expand_dims(sequence, axis=0))
        
        subset_matrix = subset_correlation_matrix(correlation_matrix, unpadded_length)
        
        plot_correlation_heatmap(model_visualisation_out, subset_matrix, f'Sample {explaining_params["sequence_indexes_corr"][i]} Correlation Matrix', )    
    
    
    
