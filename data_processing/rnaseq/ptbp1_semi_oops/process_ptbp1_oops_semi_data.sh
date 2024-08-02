#!/bin/sh
#SBATCH --job-name="smoops_rnaseq"
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --output=smoops_rnaseq-%A.out
#SBATCH --partition=ncpu

# Processing of PTBP1 depletion time-course in cells at Naive stage, in control (normal trizol) RNA-seq, OOPS and semi-extractibility assay.
# Author: Ira A Iosub

# For consistency in processing, nf-core/rnaseq 3.4 will be used to process the data
# Because this version doesn't support UMI extraction from R2, we extracted the UMIs in advance using umi_extract_sbatch.sh

WORKDIR=/camp/lab/ulej/home/users/luscomben/users/iosubi/projects/miha_tajda/ptbp1
REFDIR=/camp/lab/ulej/home/users/luscomben/home/users/iosubi/genomes/gencode_M27

## LOAD REQUIRED MODULES
ml purge
ml Nextflow/21.10.3
ml Singularity/3.6.4
ml Graphviz/2.38.0-foss-2016b

export NXF_SINGULARITY_CACHEDIR=/camp/lab/ulej/home/users/luscomben/home/users/iosubi/nfcore/rnaseq/singularity
export NXF_HOME=/nemo/lab/ulej/home/users/luscomben/users/iosubi/.nextflow

cd $WORKDIR

# ==========
# Run nf-core RNA-seq pipeline
# ==========

## UPDATE PIPELINE
nextflow pull nf-core/rnaseq -r 3.4

## RUN PIPELINE
nextflow run nf-core/rnaseq -r 3.4 \
--input samplesheet.csv \
--outdir $WORKDIR/results_ptbp1_smoops \
--fasta $REFDIR/GRCm39.primary_assembly.genome.fa.gz \
--gtf $REFDIR/gencode.vM27.annotation.gtf.gz \
--gencode \
--aligner star_salmon \
--with_umi \
--clip_r1 2 \
--clip_r2 2 \
--pseudo_aligner salmon \
-profile crick \
-resume
