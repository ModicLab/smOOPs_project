#!/bin/bash

# Preprocess RIC-seq data so reads are trimmed and merged, so they can be used as input for amchakra/tosca proximity ligation pipeline in tosca_run.sh
# Author: Ira A Iosub

# Original data location: /camp/lab/ulej/home/users/klobuct/projects/ricseq/data/20230418_naive_epi/KI_TK_230313.zipKI_TK_230313.zip

WORKDIR=/camp/lab/ulej/home/users/luscomben/users/iosubi/projects/miha_tajda/20230418_naive_epi

cd $WORKDIR
mkdir trimmed
TRIMMEDDIR=$WORKDIR/trimmed

# ==========
# Get data
# ==========

# Params
UMIPAT=NNNCCCCNNN
ADAPTER=AGATCGGAAGAG

# ==========
# Trim adapters move UMI to read header
# ==========

# UMI-tools: had to swap R1 for R2 as only R2 contains the UMI
# The datasets don't need demultiplexing so using UMPIPAT

for i in NaiveG9_1_S87 NaiveG9_2_S88 NaiveG9_3_S89 EpiG9_1_S90 EpiG9_2_S91 EpiG9_3_S92 Input_NaiveG9_1_S93 Input_NaiveG9_2_S94; do

    echo $i

    umi_tools extract --extract-method=string --bc-pattern=$UMIPAT \
    -I ${i}_R2_001.fastq.gz --read2-in=${i}_R1_001.fastq.gz \
    -S $TRIMMEDDIR/${i}_R2.umi.fastq.gz --read2-out=$TRIMMEDDIR/${i}_R1.umi.fastq.gz -L $TRIMMEDDIR/${i}_umi_extract.log && \
    cutadapt -a $ADAPTER -A $ADAPTER -j 8 -n 10 -m 16 -q 20 --nextseq-trim=20 --pair-filter=any \
    -o $TRIMMEDDIR/${i}_R1.umi.trimmed.fastq.gz \
    -p $TRIMMEDDIR/${i}_R2.umi.trimmed.fastq.gz \
    $TRIMMEDDIR/${i}_R1.umi.fastq.gz $TRIMMEDDIR/${i}_R2.umi.fastq.gz > $TRIMMEDDIR/${i}_cutadapt.log

done

# ==========
# QC on trimmed reads
# ==========

cd $TRIMMEDDIR

fastqc *umi.trimmed.fastq.gz
multiqc *


# ==========
# Merge mates
# ==========

for i in NaiveG9_1_S87 NaiveG9_2_S88 NaiveG9_3_S89 EpiG9_1_S90 EpiG9_2_S91 EpiG9_3_S92 Input_NaiveG9_1_S93 Input_NaiveG9_2_S94; do

    echo $i
    bbmerge.sh in1=$TRIMMEDDIR/${i}_R1.umi.trimmed.fastq.gz in2=$TRIMMEDDIR/${i}_R2.umi.trimmed.fastq.gz \
    out=$TRIMMEDDIR/${i}.merged.fastq.gz outu1=$TRIMMEDDIR/${i}_R1.unmerged.fastq.gz outu2=$TRIMMEDDIR/${i}_R2.unmerged.fastq.gz

done