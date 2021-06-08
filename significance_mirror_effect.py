import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import util
import time
import os 
from argparse import ArgumentParser
from scipy.stats import pearsonr

#SEBASTIEN URCHS
def p_permut(empirical_value, permutation_values):
    n_permutation = len(permutation_values)
    if empirical_value >= 0:
        return (np.sum(permutation_values > empirical_value)+1) / (n_permutation + 1)
    return (np.sum(permutation_values < empirical_value)+1) / (n_permutation + 1)

if __name__ == "__main__":
    parser.add_argument("--b_path_nomc",help="path to nomc cc betamaps dir",dest='b_path_nomc')
    parser.add_argument("--n_path_nomc",help="path to nomc cc nullmodels dir",dest='n_path_nomc')
    parser.add_argument("--path_out",help="path to output directory",dest='path_out')
    args = parser.parse_args()
    
    n_path_nomc = os.path.join(args.n_path_nomc,'{}_null_model_nomc.npy')
    b_path_nomc = os.path.join(args.b_path_nomc,'cc_{}_results_nomc.csv')
    path_out = args.path_out
    
    #############
    # LOAD DATA #
    #############
    cases = ['1q21_1','2q13','13q12_12','15q11_2','16p11_2','22q11_2']
    
    null_del = []
    null_dup = []
    beta_del = []
    beta_dup = []
    for c in cases:
        null_del.append(np.load(n_path_nomc.format('DEL'+c)))
        null_dup.append(np.load(n_path_nomc.format('DUP'+c)))

        beta_del.append(pd.read_csv(b_path_nomc.format('DEL'+c))['betas'])
        beta_dup.append(pd.read_csv(b_path_nomc.format('DUP'+c))['betas'])
    
    ########################
    # GET DIFFERENCE DISTS #
    ########################
    diff = np.zeros((len(cases),10000))
    for i in range(len(cases)):
        for j in range(5000):
            beta = beta_del[i]
            null = null_dup[i][j]
            diff[i,j] = np.abs(np.mean(beta) - np.mean(null))

        for j in range(5000):
            beta = beta_dup[i]
            null = null_del[i][j]
            diff[i,j + 5000] = np.abs(np.mean(beta) - np.mean(null))
    
    ##########################
    # GET ACTUAL DIFFERENCES #
    ##########################
    diff_bb = np.zeros(len(cases))
    for i in range(len(cases)):
        diff_bb[i] = np.abs(np.mean(beta_del[i]) - np.mean(beta_dup[i]))
    
    #########
    # PVALS #
    #########
    pval = []
    for i in range(len(cases)):
        pval.append(p_permut(diff_bb[i],diff[i,:]))
        
    df = pd.DataFrame(pval, index=cases,columns=['pvals'])
    df.to_csv(os.path.join(path_out,'mirror_effect_pvals.csv'))
        