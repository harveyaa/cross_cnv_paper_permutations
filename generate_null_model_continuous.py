from cnvfc import stats
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import patsy as pat
import sys
import time

def random_permutation(iterable, seed=None):
    np.random.seed(seed)
    return np.random.permutation(iterable)

def permutation_glm_continuous(pheno, conn, contrast, regressors='', n_iter=1000, stand=False):
    # Make sure pheno and conn have the same number of cases
    if not conn.shape[0] == pheno.shape[0]:
        print(f'Conn ({conn.shape[0]}) and pheno ({pheno.shape[0]}) must be same number of cases')
        
    sub_mask = stats.find_subset(pheno, contrast)
    sub_pheno = pheno.loc[sub_mask]
    sub_conn = conn[sub_mask, :]
    
    # Standardize the input data
    if stand:
        sub_conn = stats.standardize(sub_conn, np.ones(n_sub).astype(bool))
    n_conn = sub_conn.shape[1]
    
    betas = np.zeros(shape=(n_iter, n_conn))
    start = time.time()
    for i in range(n_iter):
        i_pheno = sub_pheno.copy()
        contrast_id = list(i_pheno.columns).index(contrast)
        i_pheno.iloc[:, contrast_id] = random_permutation(pheno[contrast].tolist(),seed=i)
        
        table = stats.glm_wrap_continuous(sub_conn, i_pheno, contrast, regressors=regressors,
                            report=False, fast=True)
        
        betas[i, :] = table['betas'].values

        elapsed = time.time() - start
        done = i + 1
        remaining = n_iter - done
        time_left = (elapsed / done) * remaining

        sys.stdout.write('\r {}/{}. {:.2f}s left ({:.3f}s per permutation)'.format(done, n_iter, time_left, elapsed / done))
        sys.stdout.flush()
    sys.stdout.write('\r Done. Took {:.2f}s'.format(elapsed))
    sys.stdout.flush()

    return betas

def get_year(s):
    if isinstance(s,str):
        return int(s.split('/')[-1])
    else:
        return np.nan

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--contrast",help="contrast to run",dest='contrast')
    args = parser.parse_args()

    n_iter = 5000
    contrast = args.contrast
    
    #pheno_p ='/home/harveyaa/Documents/fMRI/data/ukbb_9cohorts/pheno_drop_maillard_15q11_2del.csv'
    #connectomes_p = '/home/harveyaa/Documents/fMRI/data/ukbb_9cohorts/connectomes_drop_maillard_15q11_2del.npy'
    #out_p = '/home/harveyaa/Documents/fMRI/cnv_fmri/permutations/'

    pheno_p ='/home/harveyaa/scratch/pheno_drop_maillard_15q11_2del.csv'
    connectomes_p = '/home/harveyaa/scratch/connectomes_drop_maillard_15q11_2del.npy'
    out_p = '/home/harveyaa/scratch/null_models_continuous/'
    
    pheno = pd.read_csv(pheno_p, index_col=0)
    connectomes = np.load(connectomes_p)

    regressors_str_mc = ' + '.join(['AGE','C(SEX)', 'FD_scrubbed', 'C(SITE)','mean_conn'])
    regressors_str_nomc = ' + '.join(['AGE','C(SEX)', 'FD_scrubbed', 'C(SITE)'])

    p = pheno.copy()
    p['year'] = p['date_of_attending_assessment_centre_f53_2_0'].apply(get_year)
    
    if ('Stand_PRS' in contrast):
        p = p[p['PI']=='UKBB']
        p = p[(p.PRS_eth == 'WB') | (p.PRS_eth == 'EUR')]
    
    if (contrast in ['Stand_PRS_SA','Stand_PRS_thickness']):
        p = p.dropna(subset=['date_of_attending_assessment_centre_f53_2_0'])
        p =  p[p['year'] > 2017]
    
    if ('Stand_' not in contrast):
        print('Dropping ',p[contrast].isna().sum(),' subjects w/ NaN for {}.'.format(contrast))
        if (p[contrast].isna().sum() == p.shape[0]):
            print('ERROR: No subjects with data for {}.'.format(contrast))
        p = p.dropna(subset=[contrast])
        print('Z-scoring contrast...')
        p['{}_z'.format(contrast)] = (p[contrast] - p[contrast].mean())/p[contrast].std(ddof=0)
        contrast = '{}_z'.format(contrast)
        
    mask = np.array(~p[contrast].isnull())
    
    match_conn_mask = pheno.index.isin(p.index)
    conn = connectomes[match_conn_mask]
    
    betas_mc = permutation_glm_continuous(p[mask], conn[mask], contrast,regressors=regressors_str_mc, n_iter=n_iter, stand=False)
    
    #FIX FOR NAMING
    if (contrast[-2:] == '_z'):
        np.save(out_p + '{}_null_model_mc.npy'.format(contrast[:-2]), betas_mc)
    else:
        np.save(out_p + '{}_null_model_mc.npy'.format(contrast), betas_mc)

    betas_nomc = permutation_glm_continuous(p[mask], conn[mask], contrast,regressors=regressors_str_nomc, n_iter=n_iter, stand=False)
    
    #FIX FOR NAMING
    if (contrast[-2:] == '_z'):
        np.save(out_p + '{}_null_model_nomc.npy'.format(contrast[:-2]), betas_nomc)
    else:
        np.save(out_p + '{}_null_model_nomc.npy'.format(contrast), betas_nomc)