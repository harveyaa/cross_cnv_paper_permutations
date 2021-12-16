from cnvfc import stats
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import patsy as pat
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import os
from util import standardize
import time

# redefined function from util to only generate standardized betamaps
def variable_effect(pheno,var,regressors,conn):
    """
    Test effect of continuous variable.
    
    pheno = dataframe:
        -filtered to be only relevant subjects (use mask_var)
    var = column from pheno
    regressors = list of strings, formatted for patsy
    connectomes = n_subjects x n_edges array
    
    Returns:
    table = n_edges
        - betas_std = including standardization on controls
        - pvalues = pvalues
        - qvalues = fdr corrected pvalues alpha = 0.05
    """
    
    n_edges = conn.shape[1]
    contrast = np.zeros(1 + len(regressors))
    contrast[0] = 1
    
    betas_std = np.zeros(n_edges)
    pvalues = np.zeros(n_edges)
        
    formula = ' + '.join((regressors + [var]))
    dmat = pat.dmatrix(formula, pheno, return_type='dataframe',NA_action='raise')
    
    mask_std = np.ones(pheno.shape[0]).astype(bool)
    conn_std = standardize(mask_std, conn)
    
    for edge in range(n_edges):
        model_std = sm.OLS(conn_std[:,edge],dmat)
        results_std = model_std.fit()
        betas_std[edge] = results_std.params[var]
        pvalues[edge] = results_std.pvalues[var]
        
    mt = multipletests(pvalues,method='fdr_bh')
    reject = mt[0]
    qvalues = mt[1]
    
    table = pd.DataFrame(np.array([betas_std,pvalues,qvalues,reject]).transpose(),
                         columns=['betas_std','pvalues','qvalues','reject'])
    return table

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--case",help="case to run",dest='case')
    parser.add_argument("--path_pheno",help="path to phenotype .csv file",dest='path_pheno')
    parser.add_argument("--path_connectomes",help="path to connectomes .csv file",dest='path_connectomes')
    parser.add_argument("--path_out",help="path to output directory",dest='path_out')
    args = parser.parse_args()

    n_iter = 5000
    c = args.case
    path_pheno = args.path_pheno
    path_connectomes = args.path_connectomes
    path_out = args.path_out
    print(f'### {c} ###')
    
    #############
    # LOAD DATA #
    #############
    print('Loading data...')
    pheno = pd.read_csv(path_pheno, index_col=0)
    connectomes = pd.read_csv(path_connectomes,index_col=0)
    print('Done!')
    
    ###################
    # SELECT SUBJECTS #
    ###################
    p = pheno.copy()

    if ('Stand_PRS' in c):
        p = p[p['PI']=='UKBB']
        p = p[(p.PRS_eth == 'WB') | (p.PRS_eth == 'EUR')]

    if ('Stand_' not in c):
        print('Dropping ',p[c].isna().sum(),' subjects w/ NaN for {}.'.format(c))
        if (p[c].isna().sum() == p.shape[0]):
            print('ERROR: No subjects with data for {}.'.format(c))
        p = p.dropna(subset=[c])
        print('Z-scoring contrast...')
        p['{}_z'.format(c)] = (p[c] - p[c].mean())/p[c].std(ddof=0)
        c = '{}_z'.format(c)

    mask = np.array(~pheno[c].isnull())

    match_conn_mask = pheno.index.isin(p.index)
    conn = connectomes[match_conn_mask]

    idx = pheno[mask].index
    
    ######################
    # GENERATE BOOTSTRAP #
    ######################
    print(f'Generating bootstrap distribution with {n_iter} iterations...')
    regressors_mc = ['AGE','C(SEX)', 'FD_scrubbed', 'C(SITE)','mean_conn']
    
    bootstrap = np.zeros((n_iter,2080))
    start = time.time()
    for i in range(n_iter):
        # Resample index
        resample_idx = np.random.choice(idx,len(idx))

        # Generate betamap
        table = variable_effect(pheno.loc[resample_idx],
                            c,
                            regressors_mc,
                            conn.loc[resample_idx])
        bootstrap[i,:] = table['betas_std']

        if i%100 == 0:
            dur = time.time() - start
            remain = (dur/(i+1))*(n_iter - (i+1))

            print(f'{i}/{n_iter} - time:{np.round(dur,2)}')
            print(f'Est remaining: {np.round(remain,2)}\n')
            
    print('Saving...')
    np.save(os.path.join(path_out,f'{c}_bs_dist_std_mc.npy'),bootstrap)
    print('Done!')
    

    
    