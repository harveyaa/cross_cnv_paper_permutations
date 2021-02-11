import cnvfc
import numpy as np
import pandas as pd
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--case",help="case to run",dest='case')
    args = parser.parse_args()

    n_iter = 5000
    case = args.case

    #pheno_p ='/home/harveyaa/Documents/fMRI/data/ukbb_9cohorts/pheno_drop_maillard_15q11_2del.csv'
    #connectomes_p = '/home/harveyaa/Documents/fMRI/data/ukbb_9cohorts/connectomes_drop_maillard_15q11_2del.npy'
    #out_p = '/home/harveyaa/Documents/fMRI/cnv_fmri/permutations/'

    pheno_p ='/home/harveyaa/scratch/pheno_drop_maillard_15q11_2del.csv'
    connectomes_p = '/home/harveyaa/scratch/connectomes_drop_maillard_15q11_2del.npy'
    out_p = '/home/harveyaa/scratch/null_models/'

    pheno = pd.read_csv(pheno_p, index_col=0)
    connectomes = np.load(connectomes_p)
    group = 'CNV_name'

    regressors_str_mc = ' + '.join(['AGE','C(SEX)', 'FD_scrubbed', 'C(SITE)','mean_conn'])
    regressors_str_nomc = ' + '.join(['AGE','C(SEX)', 'FD_scrubbed', 'C(SITE)'])

    #RENAME PHENO TO GROUP IPC
    CNV_name = []
    for name in pheno['CNV_name']:
        if (name=='SZ_ds'):
            CNV_name.append('SZ')
        elif (name=='BIP_ds'):
            CNV_name.append('BIP')
        elif (name=='ADHD_ds'):
            CNV_name.append('ADHD')
        elif (name=='Autism'):
            CNV_name.append('ASD')
        else:
            CNV_name.append(name)
    pheno['CNV_name'] = CNV_name

    #SELECT CONTROLS BASED ON CASE
    ipc = ['ADHD','ASD','BIP','SZ']
    control = 'non_carriers'
    if (case in ipc):
        control = 'CON_IPC'

    cases = ['DEL1q21_1','DEL2q13','NRXN1del','DEL13q12_12','DEL15q11_2','DEL16p11_2','DEL16p13_11','DEL17p12','DEL22q11_2','TAR_dup',
        'DUP1q21_1','DUP2q13','DUP13q12_12','DUP15q11_2','DUP15q13_3_CHRNA7','DUP16p11_2','DUP16p13_11','DUP17p12','DUP22q11_2',
        'SZ','BIP','ASD','ADHD']
    
    df_pi = pheno.groupby('PI').sum()[cases]
    mask_pi = (df_pi > 0)

    if case in ipc:
        mask_case = pheno[case].to_numpy(dtype=bool)
        mask_con = pheno[control].to_numpy(dtype=bool)
        mask = mask_case + mask_con
    else:
        mask_case = pheno[case].to_numpy(dtype=bool)
        pi_list = df_pi[mask_pi[case]].index.to_list()
        mask_con = np.array((pheno['PI'].isin(pi_list))&(pheno['non_carriers']==1))
        mask = mask_case + mask_con

    #DO MEAN CORRECTED
    betas_mc = cnvfc.stats.permutation_glm(pheno[mask], connectomes[mask], group, case, control,regressors=regressors_str_mc, n_iter=n_iter, stand=False)
    np.save(out_p + '{}_null_model_mc.npy'.format(case), betas_mc)

    #DO NOMC
    betas_mc = cnvfc.stats.permutation_glm(pheno[mask], connectomes[mask], group, case, control,regressors=regressors_str_nomc, n_iter=n_iter, stand=False)
    np.save(out_p + '{}_null_model_nomc.npy'.format(case), betas_mc)
    
    