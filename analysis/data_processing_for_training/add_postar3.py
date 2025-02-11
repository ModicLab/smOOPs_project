import pandas as pd
from multiprocessing import Pool
import os
import numpy as np

postar3_data = pd.read_csv('POSTAR3_from_source/mouse_clipdb_mm39_formated.txt', sep='\t')

all_smoops_transcripts = pd.read_csv("Data/machine_learning_input_prep/all_transcripts_with_fasta_clip_m6a_paris_intra_paris_inter.bed", sep='\t')

def map_clip_signal(row, df_clip):
    relevant_clips = df_clip[
        (df_clip['Chromosome'] == row['chromosome']) & 
        (df_clip['Start'] >= row['start']) & 
        (df_clip['End'] <= row['end']) &
        (df_clip['Strand'] == row['strand']) 
    ]

    clip_signal = [0] * len(row['sequence'])

    for _, clip_row in relevant_clips.iterrows():
        clip_start = max(clip_row['Start'], row['start'])
        clip_end = min(clip_row['End'], row['end'])
        start_index = clip_start - row['start']
        end_index = clip_end - row['start']
        for idx in range(start_index, end_index):
            clip_signal[idx] = 1
            
    if row['strand'] == '-':
        clip_signal.reverse()
    
    return ','.join(map(str, clip_signal))

def process_row(row):
    return row['transcript_id'], map_clip_signal(row, postar3_data_subset)

def parallel_apply(df, func, num_cores=40):
    df_split = np.array_split(df, num_cores)
    pool = Pool(num_cores)
    result = pool.map(func, df_split)
    pool.close()
    pool.join()
    return pd.concat(result)

rpbs = np.sort(postar3_data['RBP'].unique())

for rbp in rpbs:
    print("Processing: ", rbp)
    postar3_data_subset = postar3_data[postar3_data['RBP'] == rbp]

    results = []
    with Pool(processes=40) as pool:
        results = pool.starmap(process_row, [(row,) for _, row in all_smoops_transcripts.iterrows()])

    transcript_ids, clip_signals = zip(*results)
    all_smoops_transcripts[f'{rbp}_postar3'] = clip_signals

all_smoops_transcripts.to_csv("Data/machine_learning_input_prep/all_transcripts_with_fasta_clip_m6a_paris_intra_paris_inter_postar3.bed", sep='\t', index=False)