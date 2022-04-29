import os
import util
import numpy as np
import pandas as pd
from argparse import ArgumentParser

def iter_sample_var_effect(pheno,connectomes,var,regressors,p_out,n_sample=450,n_iter=1000,std=True):
    clean_regressors = [r[2:-1] if r[0]=='C' else r for r in regressors]

    p = pheno[clean_regressors + [var,'PRS_eth','PI']].copy()
    p = p[p['PI']=='UKBB']
    p = p[(p.PRS_eth == 'WB') | (p.PRS_eth == 'EUR')]
    mask = util.mask_var(p,var)
    p = p[mask]
    conn = connectomes.to_numpy()[pheno.index.isin(p.index)]

    betas = np.zeros((n_iter,2080))
    mtds = []
    for i in range(n_iter):
        sample_subs = np.random.choice(p.index,n_sample,replace=False)

        
        summary = util.variable_effect(p[p.index.isin(sample_subs)],
                                        var,
                                        regressors,
                                        conn[p.index.isin(sample_subs)],
                                        std=std)
        bbb = summary['betas_std'].to_numpy()
        betas[i,:] = bbb

        rank = pd.qcut(np.abs(bbb),10,labels=False)                    
        decile = np.abs(bbb)[rank[rank==9]]
        mtds.append(np.mean(decile))
    
    if not p_out is None:
        np.save(os.path.join(p_out,f'betas_{var}_{n_sample}_{n_iter}.npy'),betas)
        np.save(os.path.join(p_out,f'mtds_{var}_{n_sample}_{n_iter}.npy'),mtds)
    return betas, mtds

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path_pheno",help="path to phenotype .csv file",
                        default='/home/harveyaa/projects/def-pbellec/harveyaa/data/pheno_26-01-22.csv')
    parser.add_argument("--path_connectomes",help="path to connectomes .csv file",
                        default='/home/harveyaa/projects/def-pbellec/harveyaa/data/connectomes_01-12-21.csv')
    parser.add_argument("--path_out",help="path to output directory")
    parser.add_argument("--prs",help="which prs")
    parser.add_argument("--n_iter",help="number of iterations",type=int)
    parser.add_argument("--n_sample",help="number of subjects in sample",type=int)
    args = parser.parse_args()
    
    #############
    # LOAD DATA #
    #############
    pheno = pd.read_csv(args.path_pheno,index_col=0)
    connectomes = pd.read_csv(args.path_connectomes,index_col=0)

    regressors_mc = ['AGE','C(SEX)','FD_scrubbed', 'C(SITE)', 'mean_conn']

    iter_sample_var_effect(pheno,
                            connectomes,
                            args.prs,
                            regressors_mc,
                            args.path_out,
                            n_iter=args.n_iter,
                            n_sample=args.n_sample)

