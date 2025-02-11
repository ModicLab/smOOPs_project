import pandas as pd
from multiprocessing import Pool
import os
import numpy as np

paris_inter_data = pd.read_csv('Data/PARIS/hybrids/mES_no_rRNA_or_tRNA_curated.tsv', sep='\t')
paris_data = pd.read_csv('Data/PARIS/hybrids/mES_no_rRNA_or_tRNA_curated.tsv', sep='\t')
paris_inter_data = paris_data[paris_data['type'] == 'intergenic']
paris_inter_data_L = paris_inter_data[['L_genomic_seqnames', 'L_genomic_start', 'L_genomic_end', 'L_genomic_strand', 'L_gene_name', 'L_region', 'intragroup']]
paris_inter_data_R = paris_inter_data[['R_genomic_seqnames', 'R_genomic_start', 'R_genomic_end', 'R_genomic_strand', 'R_gene_name', 'R_region', 'intragroup']]
paris_inter_data_L.columns = ['Chromosome', 'Start', 'End', 'Strand', 'gene_name', 'region', 'intragroup']
paris_inter_data_R.columns = ['Chromosome', 'Start', 'End', 'Strand', 'gene_name', 'region', 'intragroup']
paris_inter_data = pd.concat([paris_inter_data_L, paris_inter_data_R])
paris_inter_data_intragroup = paris_inter_data[paris_inter_data['intragroup'] == True]
paris_inter_data_intergroup = paris_inter_data[paris_inter_data['intragroup'] == False]

naive_transcripts = pd.read_csv("Data/machine_learning_input_prep/all_transcripts_with_fasta_clip_m6a_paris_intramol.bed", sep='\t')

def map_clip_signal(row, df_clip):
    print(row['transcript_id'])
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
            clip_signal[idx] += 1
            
    if row['strand'] == '-':
        clip_signal.reverse()
    
    return ','.join(map(str, clip_signal))

def process_row(row):
    return row['transcript_id'], map_clip_signal(row, paris_inter_data_intergroup)

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
naive_transcripts['paris_intergroup'] = clip_signals

###############################################################################################################################

def map_clip_signal(row, df_clip):
    print(row['transcript_id'])

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
            clip_signal[idx] += 1
            
    if row['strand'] == '-':
        clip_signal.reverse()
    
    return ','.join(map(str, clip_signal))

def process_row(row):
    return row['transcript_id'], map_clip_signal(row, paris_inter_data_intragroup)

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
naive_transcripts['paris_intragroup'] = clip_signals


naive_transcripts.to_csv("Data/machine_learning_input_prep/all_transcripts_with_fasta_clip_m6a_paris_intra_paris_inter.bed", sep='\t', index=False)