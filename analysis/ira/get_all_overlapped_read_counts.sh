#!/bin/bash

# Paths to input files and directories
BAM_DIR="/camp/lab/ulej/home/users/luscomben/users/iosubi/projects/mesc_oops_semi/results_oops_semi/star_salmon"
GENES_GTF="genes.gtf"          # Path to genes GTF file
INTRONS_BED="introns.bed"      # Path to introns BED file
OUTPUT_DIR="/camp/lab/ulej/home/users/luscomben/users/iosubi/projects/mesc_oops_semi/read_counts"  # Directory to store output files
SUMMARY_FILE="$OUTPUT_DIR/summary_combined.txt"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Initialize the summary file
echo -e "Sample\tTotal_Reads\tJunction_Spanning_Reads\tIntron_Overlapping_Reads" > $SUMMARY_FILE

# Loop through each BAM file in the BAM_DIR
for BAM_FILE in $BAM_DIR/*.bam; do
    # Get the sample name (without directory and extension)
    SAMPLE=$(basename $BAM_FILE .bam)
    echo "Processing $SAMPLE"

    # Step 1: Filter for sense reads that overlap genes in a strand-aware fashion
    SENSE_BAM="$OUTPUT_DIR/${SAMPLE}_sense.bam"
    bedtools intersect -abam $BAM_FILE -b $GENES_GTF -s > $SENSE_BAM

    # Step 2: Filter for unique mappers (MAPQ >= 255 for unique reads)
    FILTERED_BAM="$OUTPUT_DIR/${SAMPLE}_unique.bam"
    samtools view -b -q 255 $SENSE_BAM > $FILTERED_BAM

    # Step 3: Count total reads in the filtered BAM
    TOTAL_READS=$(samtools view -c $FILTERED_BAM)

    # Step 4: Count the number of junction-spanning reads (CIGAR containing 'N')
    JUNCTION_SPANNING_READS=$(samtools view $FILTERED_BAM | awk '$6 ~ /N/' | wc -l)

    # Step 5: Intersect with introns in a strand-aware fashion
    INTERSECTED_BAM="$OUTPUT_DIR/${SAMPLE}_intron_overlapping.bam"
    bedtools intersect -abam $FILTERED_BAM -b $INTRONS_BED -s > $INTERSECTED_BAM

    # Step 6: Count the number of intron-overlapping reads
    INTRON_OVERLAPPING_READS=$(samtools view -c $INTERSECTED_BAM)

    # Step 7: Write the results to the summary file
    echo -e "${SAMPLE}\t${TOTAL_READS}\t${JUNCTION_SPANNING_READS}\t${INTRON_OVERLAPPING_READS}" >> $SUMMARY_FILE

    # Clean up intermediate files
    rm $SENSE_BAM $FILTERED_BAM $INTERSECTED_BAM
done

