import pandas as pd
from multiprocessing import Pool
import os
import numpy as np

m6a = pd.read_csv('Data/m6a/crosslinks/m6a_mm39-esc-mouse-2i-lif-wt-20180606-jufastqgz_unmapped_single.bed', sep='\t', names=['Chromosome', 'Start', 'End', 'Score', 'filler_column', 'Strand'])

naive_transcripts = pd.read_csv("Data/machine_learning_input_prep/all_transcripts_with_fasta_clip.bed", sep='\t')

def map_clip_signal(row, experiment_data):
    print(row['transcript_id'])

    relevant_clips = experiment_data[
        (experiment_data['Chromosome'] == row['chromosome']) & 
        (experiment_data['Strand'] == row['strand']) & 
        (experiment_data['Start'] >= row['start']) & 
        (experiment_data['End'] <= row['end'])
    ]

    clip_signal = [0] * len(row['sequence'])

    for _, clip_row in relevant_clips.iterrows():
        clip_start = max(clip_row['Start'], row['start'])
        clip_end = min(clip_row['End'], row['end'])
        start_index = clip_start - row['start']
        end_index = clip_end - row['start']
        for idx in range(start_index, end_index):
            clip_signal[idx] += clip_row['Score']
            
    if row['strand'] == '-':
        clip_signal.reverse()
    
    return ','.join(map(str, clip_signal))

def process_row(row):
    return row['transcript_id'], map_clip_signal(row, m6a)

def parallel_apply(df, func, num_cores=40):
    df_split = np.array_split(df, num_cores)
    pool = Pool(num_cores)
    result = pool.map(func, df_split)
    pool.close()
    pool.join()
    return pd.concat(result)

results = []
with Pool(processes=40) as pool:
    results = pool.starmap(process_row, [(row,) for _, row in naive_transcripts.iterrows()])

transcript_ids, clip_signals = zip(*results)
naive_transcripts['m6a'] = clip_signals

naive_transcripts.to_csv("Data/machine_learning_input_prep/all_transcripts_with_fasta_clip_m6a.bed", sep='\t', index=False)