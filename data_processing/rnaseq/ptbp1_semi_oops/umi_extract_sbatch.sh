#!/bin/sh

# Processing of PTBP1 depletion time-course in cells at Naive stage, in control (normal trizol) RNA-seq, OOPS and semi-extractibility assay.
# Author: Ira A Iosub

# For consistency in processing, nf-core/rnaseq 3.4 will be used to process the data
# Because it doesn't support UMI extraction from R2, we extracted the UMIs in advance.

WORKDIR=/camp/lab/ulej/home/users/luscomben/users/iosubi/projects/miha_tajda/ptbp1

# ==========
# move UMI to read header
# ==========

# UMI-tools version: 1.1.1

# Params
UMIPAT=NNNNN

cd $WORKDIR

for i in *_R1_001.fastq.gz; do
    base_name="${i%%_R1*}"
    echo "${base_name}"

    sbatch -N 1 -t 24:00:00 --mem=32G -c 8 --wrap="umi_tools extract --extract-method=string --bc-pattern=$UMIPAT --bc-pattern2=$UMIPAT -I ${base_name}_R1_001.fastq.gz --read2-in=${base_name}_R2_001.fastq.gz -S umi_extract/${base_name}_R1.umi.fastq.gz --read2-out=umi_extract/${base_name}_R2.umi.fastq.gz -L ${base_name}_umi_extract.log"

done