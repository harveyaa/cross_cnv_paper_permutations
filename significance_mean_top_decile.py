import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import util
import time
import os 
from statsmodels.stats.multitest import multipletests
from argparse import ArgumentParser

#SEBASTIEN URCHS
def p_permut(empirical_value, permutation_values):
    n_permutation = len(permutation_values)
    if empirical_value >= 0:
        return (np.sum(permutation_values > empirical_value)+1) / (n_permutation + 1)
    return (np.sum(permutation_values < empirical_value)+1) / (n_permutation + 1)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_path",help="path to mc cc null models dir",dest='n_path')
    parser.add_argument("--b_path",help="path to mc cc betamaps dir",dest='b_path')
    parser.add_argument("--path_out",help="path to output directory",dest='path_out')
    args = parser.parse_args()
    
    n_path_mc = os.path.join(args.n_path,'{}_null_model_mc.npy')
    b_path_mc = os.path.join(args.b_path,'cc_{}_results_mc.csv')
    n_path_mc_cont = os.path.join(args.n_path,'{}_null_model_mc.npy')
    b_path_mc_cont = os.path.join(args.b_path,'cont_{}_results_mc.csv')
    path_out = args.path_out
    
    all_groups = ['DEL15q11_2',
                  #'DUP15q11_2',
                  #'DUP15q13_3_CHRNA7',
                  #'DEL2q13',
                  #'DUP2q13',
                  #'DUP16p13_11',
                  #'DEL13q12_12',
                  #'DUP13q12_12',
                  #'DEL17p12',
                  #'TAR_dup',
                  'DEL1q21_1',
                  'DUP1q21_1',
                  'DEL22q11_2',
                  'DUP22q11_2',
                  'DEL16p11_2',
                  'DUP16p11_2',
                  'SZ',
                  'BIP',
                  'ASD',
                  'ADHD',
                  'IBD'
                 ]
    prs = ['Stand_PRS_newCDG2_ukbb',
           'Stand_PRS_ASD',
           'Stand_PRS_SCZwave3',
           #'Stand_PRS_IQ',
           'Stand_PRS_MDD',
           #'Stand_PRS_LDL',
           #'Stand_PRS_CKD',
           'Stand_PRS_BIP',
           #'Stand_PRS_height',
           #'Stand_PRS_BMI'
          ]
    cont = prs + ['fluid_intelligence_score_all','Neuroticism']
    
    #############
    # LOAD DATA #
    #############
    
    null_models_mc = []
    beta_maps_mc = []

    for group in all_groups:
        null_models_mc.append(np.load(n_path_mc.format(group)))
        beta_maps_mc.append(pd.read_csv(b_path_mc.format(group))['betas']) 
        
    for group in cont:
        null_models_mc.append(np.load(n_path_mc_cont.format(group)))
        if (group in ['CT','SA','Vol','fluid_intelligence_score_all','Gfactor','Neuroticism']):
            beta_maps_mc.append(pd.read_csv(b_path_mc_cont.format(group+'_z'))['betas']) 
        else:
            beta_maps_mc.append(pd.read_csv(b_path_mc_cont.format(group))['betas'])
                
    ###################
    # MEAN TOP DECILE #
    ###################
    # Get mean top decile of null models
    mtd_null = np.zeros((len(null_models_mc),len(null_models_mc[0])))
    # For each null model
    for i,_ in enumerate(null_models_mc):
        label = (all_groups + cont)[i]
        print(f"Getting null model of MTD for {label}...")
        mod = null_models_mc[i] # 5000x2080
        
        # For each iteration (5000)
        for j in range(len(null_models_mc[0])):
            rank = pd.qcut(np.abs(mod[j,:]),10,labels=False)
            decile = []
            for k in range(mod.shape[1]):
                if rank[k]==9:
                    decile.append(np.abs(mod[j,:][k]))
            mean_top_dec = np.mean(decile)
            
            mtd_null[i,j] = mean_top_dec
        print('Done!')
    mtd_null = pd.DataFrame(np.transpose(mtd_null),columns=all_groups+cont)
    mtd_null.to_csv(os.path.join(path_out,'null_dist_mtd_17-03-22.csv'))
    
    print('Getting actual MTD values & calculating significance...')
    mtd = []
    p_val_mtd = []
    for i,label in enumerate(all_groups+cont):
        rank = pd.qcut(np.abs(beta_maps_mc[i]),10,labels=False)
        decile = []
        for k in range(beta_maps_mc[i].shape[0]):
            if rank[k]==9:
                decile.append(np.abs(beta_maps_mc[i])[k])
                
        mean_top_dec = np.mean(decile)
        mtd.append(mean_top_dec)
        
        p = p_permut(mean_top_dec,mtd_null[label].values)
        p_val_mtd.append(p)
    print('Done!')

    mtd_pval = pd.DataFrame(np.array([mtd,p_val_mtd]).transpose(),index=all_groups+cont,columns=['beta_map_mtd','p_permut'])
    mtd_pval.to_csv(os.path.join(path_out,'mtd_pval_unstandardized_17-03-22.csv'))