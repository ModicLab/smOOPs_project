#!/bin/bash

# Paths to input files and directories
BAM_DIR="/camp/lab/ulej/home/users/luscomben/users/iosubi/projects/mesc_oops_semi/results_oops_semi/star_salmon"
GENES_GTF="genes.gtf"            # Path to genes GTF file
INTRONS_BED="smoops_introns.bed"  # Path to smoops introns BED file

# List of smOOPs BED files
SMOOPS_BED_FILES=("naive_smoops.bed" "epi_smoops.bed" "diff_smoops.bed")

# Output directory to store results
OUTPUT_DIR="/camp/lab/ulej/home/users/luscomben/users/iosubi/projects/mesc_oops_semi/read_counts"
mkdir -p $OUTPUT_DIR

# Loop through each smOOPs BED file
for SMOOPS_BED in "${SMOOPS_BED_FILES[@]}"; do

    # Create a summary file for each smOOPs BED file
    BED_BASENAME=$(basename $SMOOPS_BED .bed)
    SUMMARY_FILE="$OUTPUT_DIR/${BED_BASENAME}_summary.txt"

    # Initialize the summary file
    echo -e "Sample\tTotal_Smoops_Reads\tIntron_Overlapping_Reads\tJunction_Spanning_Reads" > $SUMMARY_FILE

    # Loop through each BAM file in the BAM_DIR
    for BAM_FILE in $BAM_DIR/*.bam; do
        # Get the sample name (without directory and extension)
        SAMPLE=$(basename $BAM_FILE .bam)
        echo "Processing $SAMPLE for $BED_BASENAME"

        # Step 1: Filter for sense reads that overlap genes in a strand-aware fashion
        SENSE_BAM="$OUTPUT_DIR/${SAMPLE}_sense.bam"
        bedtools intersect -abam $BAM_FILE -b $GENES_GTF -s > $SENSE_BAM

        # Step 2: Filter for unique mappers (MAPQ >= 255 for unique reads)
        FILTERED_BAM="$OUTPUT_DIR/${SAMPLE}_unique.bam"
        samtools view -b -q 255 $SENSE_BAM > $FILTERED_BAM

        # Step 3: Intersect with smOOPs regions in a strand-aware fashion and count the reads
        SMOOPS_INTERSECT_BAM="$OUTPUT_DIR/${SAMPLE}_${BED_BASENAME}_overlapping.bam"
        bedtools intersect -abam $FILTERED_BAM -b $SMOOPS_BED -s > $SMOOPS_INTERSECT_BAM
        TOTAL_SMOOPS_READS=$(samtools view -c $SMOOPS_INTERSECT_BAM)

        # Step 4: Intersect the smOOPs-overlapping BAM with introns in a strand-aware fashion
        INTERSECTED_INTRONS_BAM="$OUTPUT_DIR/${SAMPLE}_${BED_BASENAME}_intron_overlapping.bam"
        bedtools intersect -abam $SMOOPS_INTERSECT_BAM -b $INTRONS_BED -s > $INTERSECTED_INTRONS_BAM
        INTRON_OVERLAPPING_READS=$(samtools view -c $INTERSECTED_INTRONS_BAM)

        # Step 5: Count the number of junction-spanning reads (CIGAR containing 'N')
        JUNCTION_SPANNING_READS=$(samtools view $SMOOPS_INTERSECT_BAM | awk '$6 ~ /N/' | wc -l)

        # Step 6: Write the results to the summary file
        echo -e "${SAMPLE}\t${TOTAL_SMOOPS_READS}\t${INTRON_OVERLAPPING_READS}\t${JUNCTION_SPANNING_READS}" >> $SUMMARY_FILE

        # Clean up intermediate files
        rm $SENSE_BAM $FILTERED_BAM $SMOOPS_INTERSECT_BAM $INTERSECTED_INTRONS_BAM

    done

done

