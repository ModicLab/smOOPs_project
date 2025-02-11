#!/bin/sh
# Script to generate
# Author: Tajda K


# nextflow version 22.10.6
# singularity SingularityPRO version 4.1.6-1.el8



nextflow pull nf-core/clipseq -r v1.0.0

nextflow run nf-core/clipseq -r v1.0.0 \
-profile singularity \
-resume \
--input /ceph/hpc/data/ki-erc/tklobucar/projects/mock_iCLIP/annotation.txt \
--fasta /ceph/hpc/data/ki-erc/tklobucar/projects/genomes/GRCm39.primary_assembly.genome.fa.gz \
--gtf /ceph/hpc/data/ki-erc/tklobucar/projects/genomes/gencode.vM27.primary_assembly.annotation.gtf.gz \
--smrna_org mouse \
--adapter AGATCGGAAGAGCACACGTCTG \
--umi_separator 'rbc:' \
--outdir /ceph/hpc/data/ki-erc/tklobucar/projects/mock_iCLIP/results_clipseq_nfcore

