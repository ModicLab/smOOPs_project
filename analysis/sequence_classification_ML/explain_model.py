#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Don't show debug info while training
# from seq_ml import utils, preprocess, model, explain_model
from seq_ml.explain import add_predicitons, final_evaluation_on_testing, explain_run, plot_individual_sequence_importance_scores, create_heatmap, plot_heatmap, extract_motifs, interaction_analysis, plot_interaction_analysis

def main():
    # Conditionally execute functions based on the Boolean values in explaining_params
    explaining_params["ADD_PREDICTIONS"] and add_predicitons(explaining_params)
    explaining_params["FINAL_EVALUATION_ON_TESTING_DATASET"] and final_evaluation_on_testing(explaining_params)
    explaining_params["EXPLAIN"] and explain_run(explaining_params)
    explaining_params["PLOT_IMPORTANCE_SCORES_OVER_INDIVIDUAL_SEQUENCES"] and plot_individual_sequence_importance_scores(explaining_params) 
    explaining_params["BIN_IMPORTANCE_SCORES"] and create_heatmap(explaining_params)
    (explaining_params["PLOT_HEATMAP_AND_CLUSTERMAP"] or  explaining_params["PLOT_SEQUENCE_CLUSTERED_BY_BIN_VALUES"]) and plot_heatmap(explaining_params)
    explaining_params["EXTRACT_MOTIFS_IN_IMPORTANCE_SCORES"] and extract_motifs(explaining_params)
    explaining_params["INTERACTION_CORRELATION_CALCULATION"] and interaction_analysis(explaining_params)
    explaining_params["INTERACTION_CORRELATION_VISUALISATION"] and plot_interaction_analysis(explaining_params)

explaining_params = {
    "run_dir": '/ceph/hpc/home/novljanj/data_storage/projects/smOOPS_paper/Analysis/Exploratory/Model_Training/Models/Testing_models/individual_run_30082024_181729255', # The output directory for the whole training run
    
    "ADD_PREDICTIONS": True, # Add the prediction for each sequence to the original file
    "FINAL_EVALUATION_ON_TESTING_DATASET": True, # Do this after optimisation and after selecting your final best model, this will evaluate the model on the testing dataset that was set aside from the begining.
    
    "EXPLAIN": True, # Set to false if importance scores are already computed
    "num_background_samples": 800, # The number of samples to use for background in SHAP (use as many as you can afford in time and compute - best to use whole dataset)    
    
    "group_to_visualise": True, # The group to explain and visualise
    
    "PLOT_IMPORTANCE_SCORES_OVER_INDIVIDUAL_SEQUENCES": True, # Set to false if you don't want to plot the importance scores over individual sequences
    "sequence_indexes": [0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16], # Index of the sequence to plot from the original_file_shuffled, takes also lists of indexes
    
    "BIN_IMPORTANCE_SCORES": True, # Set to false if binned importance scores are already computed
    "kmer_length": 1, # The kmer to analyse 1=Nucleotide, 2=Dinucleotide, 3=Trinucleotide ...
    "num_bins": 100, # Make sure this is not much longer then the smallest sequences
    
    "PLOT_HEATMAP_AND_CLUSTERMAP": True, # Set to false if you don't want to plot the heatmap and clustermap
    
    "PLOT_SEQUENCE_CLUSTERED_BY_BIN_VALUES": True, # Set to false if you don't want to plot the sequences clustered based on their importance scores
    "type_of_reduction": "UMAP", # Choose from TSNE, UMAP, PCA, Isomap, MDS, Spectral Embedding, Factor Analysis, Dictionary Learning
    "number_of_clusters": 3, # Number of clusters formed after dim reduction
    
    "EXTRACT_MOTIFS_IN_IMPORTANCE_SCORES": False, # Set to false if you don't want to extract motifs
    "max_seqlets": 40000, # The maximum number of seqlets per metacluster.
    "n_leiden": 8, # The number of Leiden clusterings to perform with different random seeds.
    "window": 400, # The window surrounding the peak center that will be considered for motif discovery.
    
    ### EXPERIMENTAL ### (needs more testing... and a mathematician lol)
    "INTERACTION_CORRELATION_CALCULATION": False, # Set to false if you don't need to calculate interactions.
    "INTERACTION_CORRELATION_VISUALISATION": False, # Set to false if you don't want to plot interactions
    "sequence_indexes_corr": [1], # The indexes of samples to use for interaction analysis
    ### EXPERIMENTAL ###
}

if __name__ == "__main__":
   main()
