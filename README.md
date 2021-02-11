# Permutation Tests for Cross CNV Paper
This repository contains all the scripts and notebooks for generating null models of betamaps through permutation and testing for significance.

Method summary:
- For each contrast (continuous scores and case-control) generated a null model of 5000 pseudo-betamaps
   - Case-control: Random permutation of case-control labels and generate betamap
   - Continuous scores: Random permutation of score values and generate betamap
- Significance of mean shift and variance
   - For each contrast, compared mean shift (variance) of actual betamap to mean shift (variance) of the 5000 pseudo-betamaps to get pvalue
- Significance of mirror effect (opposite mean shifts)
   - For loci with both DEL & DUP in dataset, for each pair (contrast1, contrast2):
      - Generate null model of differences b/w mean shifts
         - Compute difference between mean shift of actual betamap of constrast1 with mean shift of 5000 pseudo-betamaps of constrast2
         - Compute difference between mean shift of actual betamap of constrast2 with mean shift of 5000 pseudo-betamaps of contrast1
         - Get distribution of 5000 + 5000 = 10,000 differences
      - Compare difference b/w mean shift of actual betamaps for contrast1 and contrast2  with distribution of differences to get pvalue
- Significance of correlations
    - For each pair (contrast1, contrast2):
       - Generate null model of correlations
         - Compute correlation between actual betamap of constrast1 with 5000 pseudo-betamaps of constrast2
         - Compute correlation between actual betamap of constrast2 with 5000 pseudo-betamaps of costrast1
         - Get distribution of 5000 + 5000 = 10,000 correlations
       - Compare difference b/w mean shift of actual betamaps for contrast1 and contrast2  with distribution of differences to get pvalue
