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
def case_control(pheno,case,regressors,conn):
    """
    pheno = dataframe:
        -filtered to be only relevant subjects for case control (use mask_cc)
        -case column is onehot encoded
    case = column from pheno
    regressors = list of strings, formatted for patsy
    connectomes = n_subjects x n_edges array
    
    Returns:
    table = n_edges
        - betas_std = including standardization on controls
        - pvalues = pvalues
        - qvalues = fdr corrected pvalues alpha = 0.05
    """
    n_edges = conn.shape[1]

    betas = np.zeros(n_edges)
    betas_std = np.zeros(n_edges)
    pvalues = np.zeros(n_edges)

    formula = ' + '.join((regressors + [case]))
    dmat = pat.dmatrix(formula, pheno, return_type='dataframe',NA_action='raise')
    
    mask_std = ~pheno[case].to_numpy(dtype=bool)
    conn_std = standardize(mask_std, conn)
    
    for edge in range(n_edges):
        model_std = sm.OLS(conn_std[:,edge],dmat)
        results_std = model_std.fit()
        betas_std[edge] = results_std.params[case]
        pvalues[edge] = results_std.pvalues[case]
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
    case = args.case
    path_pheno = args.path_pheno
    path_connectomes = args.path_connectomes
    path_out = args.path_out
    print(f'### {case} ###')
    
    #############
    # LOAD DATA #
    #############
    print('Loading data...')
    pheno = pd.read_csv(path_pheno, index_col=0)
    connectomes = pd.read_csv(path_connectomes,index_col=0)
    print('Done!')
    
    ###################
    # SELECT CONTROLS #
    ###################
    ipc = ['ADHD','ASD','BIP','SZ']
    control = 'non_carriers'
    if (case in ipc):
        control = 'CON_IPC'
    if case == 'IBD':
        control = 'no_IBD'
        group = 'IBD_str'
        case = 'IBD_K50_K51'

    cases = ['IBD','DEL1q21_1','DEL2q13','DEL13q12_12','DEL15q11_2','DEL16p11_2','DEL17p12','DEL22q11_2','TAR_dup',
        'DUP1q21_1','DUP2q13','DUP13q12_12','DUP15q11_2','DUP15q13_3_CHRNA7','DUP16p11_2','DUP16p13_11','DUP22q11_2',
        'SZ','BIP','ASD','ADHD']
    
    df_pi = pheno.groupby('PI').sum()[cases]
    mask_pi = (df_pi > 0)

    if case in ipc:
        mask_case = pheno[case].to_numpy(dtype=bool)
        mask_con = pheno[control].to_numpy(dtype=bool)
    elif case == 'IBD_K50_K51':
        mask_case = (pheno['IBD_str'] == 'IBD_K50_K51').to_numpy(dtype=bool)
        mask_con = (pheno['IBD_str'] == 'no_IBD').to_numpy(dtype=bool)
    else:
        mask_case = pheno[case].to_numpy(dtype=bool)
        pi_list = df_pi[mask_pi[case]].index.to_list()
        mask_con = np.array((pheno['PI'].isin(pi_list))&(pheno['non_carriers']==1))

    idx_case = pheno[mask_case].index
    idx_con = pheno[mask_con].index
    
    ######################
    # GENERATE BOOTSTRAP #
    ######################
    print(f'Generating bootstrap distribution with {n_iter} iterations...')
    regressors_mc = ['AGE','C(SEX)', 'FD_scrubbed', 'C(SITE)','mean_conn']
    
    bootstrap = np.zeros((n_iter,2080))
    start = time.time()
    for i in range(n_iter):
        # Resample index
        resample_idx_case = np.random.choice(idx_case,len(idx_case))
        resample_idx_con = np.random.choice(idx_con,len(idx_con))
        resample_idx = np.concatenate([resample_idx_case,resample_idx_con])

        # Generate betamap
        table = case_control(pheno.loc[resample_idx],
                             case,
                             regressors_mc,
                             connectomes.loc[resample_idx])
        bootstrap[i,:] = table['betas_std']
        
        if i%100 == 0:
            dur = time.time() - start
            remain = (dur/(i+1))*(n_iter - (i+1))
            
            print(f'{i}/{n_iter} - time:{np.round(dur,2)}')
            print(f'Est remaining: {np.round(remain,2)}\n')
            
    print('Saving...')
    np.save(os.path.join(path_out,f'{case}_bs_dist_std_mc.npy'),bootstrap)
    print('Done!')
    

    
    