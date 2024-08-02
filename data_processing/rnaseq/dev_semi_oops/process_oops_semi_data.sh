#!/bin/sh
#SBATCH --job-name="smoops_rnaseq"
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --output=smoops_rnaseq-%A.out
#SBATCH --partition=ncpu

# Processing of datafrom cells at Naive, Epi and Diff stages, in control (normal trizol) RNA-seq, OOPS and semi-extractibility assay.
# Author: Ira A Iosub

# Note: the raw data from samplesheet.csv has been archived.

WORKDIR=/camp/lab/ulej/home/users/luscomben/users/iosubi/projects/mesc_oops_semi
REFDIR=/camp/lab/ulej/home/users/luscomben/home/users/iosubi/genomes/gencode_M27

## LOAD REQUIRED MODULES
ml purge
ml Nextflow/21.10.3
ml Singularity/3.6.4
ml Graphviz/2.38.0-foss-2016b

export NXF_SINGULARITY_CACHEDIR=/camp/lab/luscomben/home/users/iosubi/nfcore/rnaseq/singularity
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
--outdir $WORKDIR/results_oops_semi \
--fasta $REFDIR/GRCm39.primary_assembly.genome.fa.gz \
--gtf $REFDIR/gencode.vM27.annotation.gtf.gz \
--gencode \
--aligner star_salmon \
--with_umi \
--umitools_bc_pattern 'NNNNNNNNNNNN' \
--pseudo_aligner salmon \
-profile crick \
-c custom.config \
--email ira.iosub@crick.ac.uk \
-resume