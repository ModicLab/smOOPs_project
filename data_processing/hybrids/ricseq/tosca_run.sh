#!/usr/bin/sh
#SBATCH --job-name=nf-ricseq
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=ncpu

# Analyses of samples preprocessed with preprocess.sh
# Tosca run with v1.0.0 on merged mates
# Author: Ira A Iosub

WORKDIR=/nemo/lab/ulej/home/users/luscomben/users/iosubi/projects/miha_tajda/20230418_naive_epi
GITHUBDIR=/nemo/lab/ulej/home/users/luscomben/users/iosubi/projects/miha_tajda/ric-seq-analysis/20230418_naive_epi
REFDIR=/camp/lab/ulej/home/users/luscomben/users/iosubi/projects/miha_tajda/ref/mouse

## LOAD REQUIRED MODULES
ml purge
ml Nextflow/23.10.0
ml Singularity/3.6.4
# ml Graphviz/2.38.0-foss-2016b

# ==========
# Run Tosca pipeline
# ==========

# export NXF_WORK=/camp/lab/luscomben/scratch/iosubi/ricseq_tajda_work
export NXF_SINGULARITY_CACHEDIR=/nemo/lab/ulej/home/shared/singularity
export NXF_HOME=/nemo/lab/ulej/home/users/luscomben/users/iosubi/.nextflow

mkdir -p $WORKDIR/results_naive_epi_v1

## UPDATE PIPELINE
nextflow pull amchakra/tosca -r v1.0.0

## RUN PIPELINE
nextflow run amchakra/tosca -r v1.0.0 \
-profile crick \
--input $GITHUBDIR/samplesheet_v1.csv \
--outdir $WORKDIR/results_naive_epi_v1  \
--genome_fai $REFDIR/GRCm39.fa.fai \
--star_genome $REFDIR/STAR_GRCm39_GencodeM27 \
--transcript_fa $REFDIR/GRCm39.gencode_M27.fa \
--transcript_fai $REFDIR/GRCm39.gencode_M27.fa.fai \
--transcript_gtf $REFDIR/GRCm39.gencode_M27.tx.gtf.gz \
--regions_gtf $REFDIR/regions_M27.gtf.gz \
--percent_overlap 0.5 \
--analyse_structures true \
--clusters_only true \
--shuffled_energies true \
--star_args '--limitOutSJcollapsed 5000000' \
--atlas false \
-resume