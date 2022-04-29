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
    n_path_nomc = os.path.join(args.n_path,'{}_null_model_nomc.npy')
    b_path_nomc = os.path.join(args.b_path,'cc_{}_results_nomc.csv')
    n_path_nomc_cont = os.path.join(args.n_path,'{}_null_model_nomc.npy')
    b_path_nomc_cont = os.path.join(args.b_path,'cont_{}_results_nomc.csv')
    path_out = args.path_out
    
    all_groups = ['DEL15q11_2','DUP15q11_2','DUP15q13_3_CHRNA7','DEL2q13','DUP2q13','DUP16p13_11','DEL13q12_12','DUP13q12_12',
        'DEL17p12','TAR_dup','DEL1q21_1','DUP1q21_1','DEL22q11_2','DUP22q11_2','DEL16p11_2','DUP16p11_2',
      'SZ','BIP','ASD','ADHD','IBD']
    prs = ['Stand_PRS_newCDG2_ukbb','Stand_PRS_ASD','Stand_PRS_SCZwave3','Stand_PRS_IQ',
           'Stand_PRS_LDL','Stand_PRS_CKD','Stand_PRS_BIP','Stand_PRS_height','Stand_PRS_BMI']
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
            
    null_models_nomc = []
    beta_maps_nomc = []

    for group in all_groups:
        null_models_nomc.append(np.load(n_path_nomc.format(group)))
        beta_maps_nomc.append(pd.read_csv(b_path_nomc.format(group))['betas'])

    for group in cont:
        null_models_nomc.append(np.load(n_path_nomc_cont.format(group)))
        if (group in ['CT','SA','Vol','fluid_intelligence_score_all','Gfactor','Neuroticism']):
            beta_maps_nomc.append(pd.read_csv(b_path_nomc_cont.format(group+'_z'))['betas'])
        else:
            beta_maps_nomc.append(pd.read_csv(b_path_nomc_cont.format(group))['betas'])
            
    ##############        
    # MEAN SHIFT #
    ##############
    mean_shift = np.zeros((len(null_models_nomc),len(null_models_nomc[0])))
    for i in range(len(null_models_nomc)):
        mod = null_models_nomc[i]
        for j in range(len(null_models_nomc[0])):
            mean_shift[i,j] = np.mean(mod[j])

    mean_shift = pd.DataFrame(np.transpose(mean_shift),columns=all_groups+cont)
    
    p_val = []
    for i in range(len(all_groups+cont)):
        p = p_permut(np.mean(beta_maps_nomc[i]),np.mean(null_models_nomc[i],1))
        p_val.append(p)

    mean_shift_maps = []
    for b_map in beta_maps_nomc:
        mean_shift_maps.append(np.mean(b_map))

    mean_shift_pval = pd.DataFrame(np.array([mean_shift_maps,p_val]).transpose(),index=all_groups+cont,columns=['beta_map_mean','p_permut'])
    mean_shift_pval.to_csv(os.path.join(path_out,'mean_shift_pval_unstandardized_14-12-21.csv'))
    
    ############
    # VARIANCE #
    ############
    var = np.zeros((len(null_models_mc),len(null_models_mc[0])))
    for i in range(len(null_models_mc)):
        mod = null_models_mc[i]
        for j in range(len(null_models_mc[0])):
            var[i,j] = np.var(mod[j])

    var = pd.DataFrame(np.transpose(var),columns=all_groups+cont)
    
    p_val_var = []
    for i in range(len(all_groups+cont)):
        p = p_permut(np.var(beta_maps_mc[i]),np.var(null_models_mc[i],1))
        p_val_var.append(p)

    var_maps = []
    for b_map in beta_maps_mc:
        var_maps.append(np.var(b_map))

    var_pval = pd.DataFrame(np.array([var_maps,p_val_var]).transpose(),index=all_groups+cont,columns=['beta_map_var','p_permut'])
    var_pval.to_csv(os.path.join(path_out,'var_pval_unstandardized_14-12-21.csv'))