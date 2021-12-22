import os
#import util
import numpy as np
import pandas as pd
from argparse import ArgumentParser

import random
import patsy as pat
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import StratifiedKFold

def standardize(mask,data):
    scaler = StandardScaler(with_mean=False, with_std=True)
    scaler.fit(data[mask])
    standardized=scaler.transform(data)
    return standardized

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
    parser.add_argument("--path_pheno",help="path to phenotype .csv file",dest='path_pheno')
    parser.add_argument("--path_connectomes",help="path to connectomes .csv file",dest='path_connectomes')
    parser.add_argument("--path_out",help="path to output directory",dest='path_out')
    parser.add_argument("--n_folds",help="Number of folds of cross_validation",dest='n_folds',type=int,default=5)
    args = parser.parse_args()
    
    #############
    # LOAD DATA #
    #############

    path_pheno = args.path_pheno
    path_connectomes = args.path_connectomes
    path_out = args.path_out
    n_folds = args.n_folds
    
    print('Loading data...')
    pheno = pd.read_csv(path_pheno,index_col=0)
    connectomes = pd.read_csv(path_connectomes,index_col=0)
    print('Done!')

    regressors_mc = ['AGE','C(SEX)','FD_scrubbed', 'C(SITE)', 'mean_conn']

    #####################
    # CONTINUOUS SCORES #
    #####################
    prs = ['Stand_PRS_height',
           'Stand_PRS_BMI',
           'Stand_PRS_BIP',
           'Stand_PRS_newCDG2_ukbb',
           'Stand_PRS_ASD',
           'Stand_PRS_SCZ',
          'Stand_PRS_LDL',
           'Stand_PRS_CKD',
           'Stand_PRS_IQ']
    cont = prs + ['fluid_intelligence_score_all','Neuroticism']
    
    cont_rows = []
    for c in cont:
        print(f'{c} - estimating effect size...')
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
            pheno['{}_z'.format(c)] = p['{}_z'.format(c)]
            c = '{}_z'.format(c)

        null_mask = np.array(~p[c].isnull())
        p = p[null_mask]

        mask = pheno.index.isin(p.index)

        idx = pheno[mask].index
        strat_col = np.ones(len(idx)) # DUMMY COL
        
        skf = StratifiedKFold(n_splits=n_folds)
        
        es_train = []
        es_test = []
        split_train_idx = []
        split_test_idx = []
        k=0
        for train_index, test_index in skf.split(idx, strat_col):
            print(f'Fold {k+1}/{n_folds}...')
            # Save split indexes for the bootstrap
            split_train_idx.append(idx[train_index].to_list())
            split_train_idx.append(idx[test_index].to_list())
            
            betamap_train = variable_effect(pheno.loc[idx[train_index]],
                                        c,
                                        regressors_mc,
                                        connectomes.loc[idx[train_index]].to_numpy())['betas_std']

            rank = pd.qcut(betamap_train.abs(),10,labels=False)
            decile_idx = rank[rank==9].index
            
            # Get train ES
            decile_train = betamap_train.abs()[decile_idx]
            mtd_train = np.mean(decile_train)
            es_train.append(mtd_train)

            betamap_test = variable_effect(pheno.loc[idx[test_index]],
                                        c,
                                        regressors_mc,
                                        connectomes.loc[idx[test_index]].to_numpy())['betas_std']
            # Get test ES
            decile_test = betamap_test.abs()[decile_idx]
            mtd_test = np.mean(decile_test)
            es_test.append(mtd_test)
            
            k += 1
            
        print(f'{c} - CV Effect size ',np.mean(es_test))
        cont_rows.append([c,np.mean(es_train),np.mean(es_test)])
        df_split_idx = pd.DataFrame(zip(split_train_idx,split_train_idx),columns=['train','test'])
        df_split_idx.to_csv(os.path.join(path_out,f'{c}_fold_idx.csv'))
        
    print('Saving continuous results...')
    df_cont = pd.DataFrame(cont_rows,columns=['case','train_ES','test_ES'])
    df_cont.to_csv(os.path.join(path_out,'cont_CV_es.csv'))
    print('Done!')