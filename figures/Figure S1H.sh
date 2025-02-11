#!/bin/sh
# Script to generate clipplotR tracks for figure S1H
# Author: Tajda K


./clipplotr -x 'Naive_G9_1.xl.bed,Naive_G9_2.xl.bed,Naive_G9_3.xl.bed,Epi_G9_1.xl.bed,Epi_G9_2.xl.bed,Epi_G9_3.xl.bed' \
-l 'Naive 1,Naive 2,Naive 3, Epi 1, Epi 2, Epi 3' \
-a gene \
-g 'gencode.vM27.primary_assembly.annotation.gtf' \
--groups 'Naive,Naive,Naive,Epi,Epi,Epi' \
-c '#9ACDE7,#4682B4,#274966,#99CCCC,#008080,#004C4C' \
--coverage 'control_naive_1.reverse.bs5.cpm.bigWig,semi_naive_1.reverse.bs5.cpm.bigWig,oops_naive_1.reverse.bs5.cpm.bigWig' \
--coverage_labels 'control,semi,OOPS' \
--coverage_colours '#000000,#000000,#000000' \
--coverage_groups 'control,semi,OOPS' \
-r 'R3hdm2' \
-o 0_R3hdm2.pdf