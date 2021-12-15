import os
import util
import numpy as np
import pandas as pd
from argparse import ArgumentParser

def get_year(s):
    if isinstance(s,str):
        return int(s.split('/')[-1])
    else:
        return np.nan

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path_pheno",help="path to phenotype .csv file",dest='path_pheno')
    parser.add_argument("--path_connectomes",help="path to connectomes .csv file",dest='path_connectomes')
    parser.add_argument("--path_out",help="path to output directory",dest='path_out')
    args = parser.parse_args()
    
    #############
    # LOAD DATA #
    #############

    path_pheno = args.path_pheno
    path_connectomes = args.path_connectomes
    path_out = args.path_out

    pheno = pd.read_csv(path_pheno,index_col=0)
    connectomes = pd.read_csv(path_connectomes,index_col=0)

    regressors_mc = ['AGE','C(SEX)','FD_scrubbed', 'C(SITE)', 'mean_conn']
    regressors_nomc = ['AGE','C(SEX)','FD_scrubbed', 'C(SITE)']

    ################
    # CASE CONTROL #
    ################

    cases =['SZ','BIP','ASD','ADHD','IBD','DEL1q21_1','DEL2q13','DEL13q12_12','DEL15q11_2','DEL16p11_2',
            'DEL17p12','DEL22q11_2','TAR_dup','DUP1q21_1','DUP2q13','DUP13q12_12','DUP15q11_2','DUP15q13_3_CHRNA7',
            'DUP16p11_2','DUP16p13_11','DUP22q11_2']
    ipc = ['SZ','BIP','ASD','ADHD']
    
    df_pi = pheno.groupby('PI').sum()[cases]
    mask_pi = (df_pi > 0)

    # MEAN CORRECTED
    summaries = []
    for case in cases:
        if case in ipc:
            mask = util.mask_cc(pheno,case,'CON_IPC')
        elif case == 'IBD':
            mask_case = (pheno['IBD_str'] == 'IBD_K50_K51').to_numpy(dtype=bool)
            mask_con = (pheno['IBD_str'] == 'no_IBD').to_numpy(dtype=bool)
            mask = mask_case + mask_con
        else:
            mask_case = pheno[case].to_numpy(dtype=bool)
            pi_list = df_pi[mask_pi[case]].index.to_list()
            mask_con = np.array((pheno['PI'].isin(pi_list))&(pheno['non_carriers']==1))
            mask = mask_case + mask_con
            print(case,pi_list)

        summary = util.case_control(pheno[mask],case,regressors_mc,connectomes.to_numpy()[mask],std=True)
        summary.to_csv(path_out + '/cc_{}_results_mc.csv'.format(case))
        np.savetxt(path_out + '/cc_{}_mc.tsv'.format(case),util.vec_to_connectome(summary['betas_std'].to_numpy()),delimiter='\t')

        summaries.append(summary)
        print('Completed {}.'.format(case))

    # NOT MEAN CORRECTED
    summaries_nomc = []
    for case in cases:
        if case in ipc:
            mask = util.mask_cc(pheno,case,'CON_IPC')
        elif case == 'IBD':
            mask_case = (pheno['IBD_str'] == 'IBD_K50_K51').to_numpy(dtype=bool)
            mask_con = (pheno['IBD_str'] == 'no_IBD').to_numpy(dtype=bool)
            mask = mask_case + mask_con
        else:
            mask_case = pheno[case].to_numpy(dtype=bool)
            pi_list = df_pi[mask_pi[case]].index.to_list()
            mask_con = np.array((pheno['PI'].isin(pi_list))&(pheno['non_carriers']==1))
            mask = mask_case + mask_con
            print(case,pi_list)
        summary = util.case_control(pheno[mask],case,regressors_nomc,connectomes.to_numpy()[mask],std=True)
        summary.to_csv(path_out + '/cc_{}_results_nomc.csv'.format(case))
        np.savetxt(path_out + '/cc_{}_nomc.tsv'.format(case),util.vec_to_connectome(summary['betas_std'].to_numpy()),delimiter='\t')
        summaries_nomc.append(summary)
        print('Completed {}.'.format(case))

    #####################
    # CONTINUOUS SCORES #
    #####################
    prs = ['Stand_PRS_height','Stand_PRS_BMI','Stand_PRS_BIP','Stand_PRS_newCDG2_ukbb','Stand_PRS_ASD','Stand_PRS_SCZ',
          'Stand_PRS_LDL','Stand_PRS_CKD','Stand_PRS_SA','Stand_PRS_MDD','Stand_PRS_IQ']
    cont = prs + ['CT','SA','Vol','fluid_intelligence_score_all','Gfactor','Neuroticism']

    # MEAN CORRECTED
    summaries_cont_mc = []
    for c in cont:
        p = pheno.copy()
        p['year'] = p['date_of_attending_assessment_centre_f53_2_0'].apply(get_year)

        if ('Stand_PRS' in c):
            p = p[p['PI']=='UKBB']
            p = p[(p.PRS_eth == 'WB') | (p.PRS_eth == 'EUR')]

        if (c in ['Stand_PRS_SA','Stand_PRS_thickness']):
            p = p.dropna(subset=['date_of_attending_assessment_centre_f53_2_0'])
            p =  p[p['year'] > 2017]

        if ('Stand_' not in c):
            print('Dropping ',p[c].isna().sum(),' subjects w/ NaN for {}.'.format(c))
            if (p[c].isna().sum() == p.shape[0]):
                print('ERROR: No subjects with data for {}.'.format(c))
            p = p.dropna(subset=[c])
            print('Z-scoring contrast...')
            p['{}_z'.format(c)] = (p[c] - p[c].mean())/p[c].std(ddof=0)
            c = '{}_z'.format(c)

        mask = util.mask_var(p,c)

        match_conn_mask = pheno.index.isin(p.index)
        conn = connectomes.to_numpy()[match_conn_mask]

        summary = util.variable_effect(p[mask],c,regressors_mc,conn[mask],std=True)
        summary.to_csv(path_out + '/cont_{}_results_mc.csv'.format(c))
        np.savetxt(path_out + '/cont_{}_mc.tsv'.format(c),util.vec_to_connectome(summary['betas_std'].to_numpy()),delimiter='\t')
        summaries_cont_mc.append(summary)
        print('Completed {}.'.format(c))

    # NOT MEAN CORRECTED
    summaries_cont_nomc = []
    for c in cont:
        p = pheno.copy()
        p['year'] = p['date_of_attending_assessment_centre_f53_2_0'].apply(get_year)

        if ('Stand_PRS' in c):
            p = p[p['PI']=='UKBB']
            p = p[(p.PRS_eth == 'WB') | (p.PRS_eth == 'EUR')]

        if (c in ['Stand_PRS_SA','Stand_PRS_thickness']):
            p = p.dropna(subset=['date_of_attending_assessment_centre_f53_2_0'])
            p =  p[p['year'] > 2017]

        if ('Stand_' not in c):
            print('Dropping ',p[c].isna().sum(),' subjects w/ NaN for {}.'.format(c))
            if (p[c].isna().sum() == p.shape[0]):
                print('ERROR: No subjects with data for {}.'.format(c))
            p = p.dropna(subset=[c])
            print('Z-scoring contrast...')
            p['{}_z'.format(c)] = (p[c] - p[c].mean())/p[c].std(ddof=0)
            c = '{}_z'.format(c)

        mask = util.mask_var(p,c)

        match_conn_mask = pheno.index.isin(p.index)
        conn = connectomes.to_numpy()[match_conn_mask]

        summary = util.variable_effect(p[mask],c,regressors_nomc,conn[mask],std=True)
        summary.to_csv(path_out + '/cont_{}_results_nomc.csv'.format(c))
        np.savetxt(path_out + '/cont_{}_nomc.tsv'.format(c),util.vec_to_connectome(summary['betas_std'].to_numpy()),delimiter='\t')

        summaries_cont_nomc.append(summary)
        print('Completed {}.'.format(c))