# Permutation Tests for Cross CNV Paper
This repository contains all the scripts and notebooks for generating null models of betamaps through permutation and testing for significance.

## Method summary:
- For each contrast (continuous scores and case-control) generated a null model of 5000 pseudo-betamaps
   - Case-control: Random permutation of case-control labels and generate betamap
   - Continuous scores: Random permutation of score values and generate betamap
- Significance of mean shift and variance
   - For each contrast, compared mean shift (variance) of actual betamap to mean shift (variance) of the 5000 pseudo-betamaps to get pvalue
- Significance of mirror effect (opposite mean shifts)
   - For loci with both DEL & DUP in dataset, for each pair (contrast1, contrast2):
      - Generate null model of differences b/w mean shifts
      - Compare difference b/w mean shift of actual betamaps for contrast1 and contrast2  with distribution of differences to get pvalue
- Significance of correlations
    - For each pair (contrast1, contrast2):
       - Generate null model of correlations
         - Compute correlation between 5000 pairs of betamaps from null models of contrast1 and contrast2
       - Compare difference b/w correlation of actual betamaps for contrast1 and contrast2  with distribution of correlations to get pvalue


## Scripts:
- generate_betamaps.py
    - Generate the betamaps for each case-control pair and variable effect.
    - Arguments:
        - `path_pheno`
            - Path to the phenotype .csv file w/ cases one hot encoded.
        - `path_connectomes`
            - Path to connectomes .csv file w/ connectomes in upper triangular form.
        - `path_out`
            - Path to an output directory.
- generate_null_model.py
    - Generate 5000 pseudo-betamaps to form a null distribution for a case-control pair.
    - Arguments:
        - `case`
            - Which case to run the null model for, must be in:
            - ['IBD', 'DEL1q21_1', 'DEL2q13', 'DEL13q12_12', 'DEL15q11_2', 'DEL16p11_2', 'DEL17p12', 'DEL22q11_2', 'TAR_dup', 'DUP1q21_1', 'DUP2q13', 'DUP13q12_12', 'DUP15q11_2', 'DUP15q13_3_CHRNA7', 'DUP16p11_2', 'DUP16p13_11', 'DUP22q11_2', 'SZ', 'BIP', 'ASD', 'ADHD'].
        - `path_pheno`
            - Path to the phenotype .csv file w/ cases one hot encoded.
        - `path_connectomes`
            - Path to connectomes .csv file w/ connectomes in upper triangular form.
        - `path_out`
            - Path to an output directory.
- generate_null_model_continuous.py
    - Generate 5000 pseudo-betamaps to form a null distribution for a variable effect.
    - Arguments:
        - `contrast`
            - Which contrast to run.
        - `path_pheno`
            - Path to the phenotype .csv file w/ cases one hot encoded.
        - `path_connectomes`
            - Path to connectomes .csv file w/ connectomes in upper triangular form.
        - `path_out`
            - Path to an output directory.
- significance_corr.py
    - Get the significance for correlation b/w betamaps.
    - Arguments:
        - `n_path_mc`
            - Path to directory w/ mean corrected null models (output dir of generate_null_model.py and generate_null_model_continuous.py).
        - `b_path_mc`
            - Path to directory w/ mean corrected betamaps (output dir of generate_betamaps.py).
        - `path_out`
            - Path to an output directory.
        - `path_corr`
            - Path to directory w/ correlations & correlation distributions (intermediate outputs). If re-running script just for matrices, saves the heavy step of computing the distributions, should be the same as `path_out`.
    - Notes:
        - Needs to be complete to run i.e. check that all betamaps and null models exist in given directories for cases/contrasts in ilnes 125-130.
        - Edit the subsets according to what cases/contrasts are desired in each figure. FDR values are determined based on the cases/contrasts in the subset.
- significance_mean_shift_var.py
- significance_mirror_effect.py
