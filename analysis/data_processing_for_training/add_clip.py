import pandas as pd
from multiprocessing import Pool
import os
import numpy as np

global_iclip_data1 = pd.read_csv('Data/Global_CLIP/Naive/Naive_G9_1.xl.bed', sep='\t', names=['Chromosome', 'Start', 'End', 'Name', 'Score', 'Strand'])
global_iclip_data2 = pd.read_csv('Data/Global_CLIP/Naive/Naive_G9_2.xl.bed', sep='\t', names=['Chromosome', 'Start', 'End', 'Name', 'Score', 'Strand'])
global_iclip_data3 = pd.read_csv('Data/Global_CLIP/Naive/Naive_G9_3.xl.bed', sep='\t', names=['Chromosome', 'Start', 'End', 'Name', 'Score', 'Strand'])
global_iclip_data = pd.concat([global_iclip_data1, global_iclip_data2, global_iclip_data3], ignore_index=True)

all_smoops_transcripts = pd.read_csv("Data/machine_learning_input_prep/all_smoops_transcripts_with_fasta.bed", sep='\t')

def map_clip_signal(row, df_clip):

    relevant_clips = df_clip[
        (df_clip['Chromosome'] == row['chromosome']) & 
        (df_clip['Strand'] == row['strand']) & 
        (df_clip['Start'] >= row['start']) & 
        (df_clip['End'] <= row['end'])
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
    return row['transcript_id'], map_clip_signal(row, global_iclip_data)

def parallel_apply(df, func, num_cores=40):
    df_split = np.array_split(df, num_cores)
    pool = Pool(num_cores)
    result = pool.map(func, df_split)
    pool.close()
    pool.join()
    return pd.concat(result)

results = []
with Pool(processes=40) as pool:
    results = pool.starmap(process_row, [(row,) for _, row in all_smoops_transcripts.iterrows()])

transcript_ids, clip_signals = zip(*results)
all_smoops_transcripts['global_iclip'] = clip_signals

all_smoops_transcripts.to_csv("Data/machine_learning_input_prep/all_smoops_transcripts_with_fasta_clip.bed", sep='\t', index=False)


