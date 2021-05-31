import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import util
import time
import os 
import itertools
from statsmodels.stats.multitest import multipletests
from scipy.stats import pearsonr
from argparse import ArgumentParser

#SEBASTIEN URCHS
def p_permut(empirical_value, permutation_values):
    n_permutation = len(permutation_values)
    if empirical_value >= 0:
        return (np.sum(permutation_values > empirical_value)+1) / (n_permutation + 1)
    return (np.sum(permutation_values < empirical_value)+1) / (n_permutation + 1)

def filter_fdr(df,contrasts):
    df_filtered = df[(df['pair0'].isin(contrasts)) & (df['pair1'].isin(contrasts))]
    _,fdr,_,_ = multipletests(df_filtered['pval'],method='fdr_bh')
    df_filtered['fdr_filtered'] = fdr
    return df_filtered

def mat_form(df,contrasts,value = 'betamap_corr'):
    n = len(contrasts)
    d = dict(zip(contrasts,range(n)))
    mat = np.zeros((n,n))
    for c in contrasts:
        #fill out vertical strip of mat
        for i in range(n):
            if (i == d[c]):
                val = 1
            else:
                val = df[((df['pair0']==c)|(df['pair1']==c))
                                & ((df['pair0']==contrasts[i])|(df['pair1']==contrasts[i]))][value]
            mat[i,d[c]] = val
            mat[d[c],i] = val
    return pd.DataFrame(mat,columns=contrasts,index=contrasts)

def make_matrices(df,contrasts,fdr = 'fdr_filtered'):
    "Param fdr can be set to 'fdr_filtered': FDR is performed using the pvalues only from the chosen contrasts"
    "                              or 'fdr': values taken from FDR performed on full set of 42 contrasts"
    if (fdr == 'fdr_filtered'):
        df = filter_fdr(df,contrasts)
    mat_corr = mat_form(df,contrasts,value = 'betamap_corr')
    mat_pval = mat_form(df,contrasts,value = 'pval')
    mat_fdr = mat_form(df,contrasts,value = fdr)
    return mat_corr,mat_pval,mat_fdr

def get_corr_dist(cases,nulls,path_out,tag='wholeconn'):
    # For each unique pair, between the null maps.
    n_pairs = int((len(cases))*(len(cases) -1)/2)
    corr = np.zeros((n_pairs,5000))

    pair = []
    l = 0
    for i in itertools.combinations(cases,2):
        for j in range(5000):
            corr[l,j] = pearsonr(nulls.loc[i[0]].values[j,:],nulls.loc[i[1]].values[j,:])[0]

        pair.append(i)
        if (l%50 == 0):
            print('{}/{}'.format(l,n_pairs))
        l = l + 1
        
    df = pd.DataFrame(corr)
    df['pair'] = pair
    df.to_csv(os.path.join(path_out,'correlation_dist_{}.csv'.format(tag)))
    return df

def get_corr(cases,betas,path_out,tag='wholeconn'):
    #For each unique pair, correlation between betamaps. Use standardized betas here (as in rest of paper).
    n_pairs = int((len(cases))*(len(cases) -1)/2)
    corr = np.zeros(n_pairs)

    pair = []
    l = 0
    for i in itertools.combinations(cases,2):
        corr[l] = pearsonr(betas.loc[i[0]].values,betas.loc[i[1]].values)[0]
        l = l + 1
        pair.append(i)
    df = pd.DataFrame(corr)
    df['pair'] = pair
    df.to_csv(os.path.join(path_out,'correlation_betas_{}.csv'.format(tag)))
    return df

def get_corr_pval(maps,nulls,betas,path_out,tag='wholeconn'):
    df = get_corr_dist(maps,nulls,path_out,tag=tag)
    df_bb = get_corr(maps,betas,path_out,tag=tag)
    
    df_bb = df_bb.rename(columns={0:'betamap_corr'})
    df_master = df_bb.merge(df,on='pair')
    
    # CALCULATE PVALS
    pval = []
    for i in df_master.index:
        p = p_permut(df_master.loc[i,'betamap_corr'],df_master[range(5000)].loc[i])
        pval.append(p)
    df_master['pval'] = pval
    
    # ADD LABELS
    pair0 = [p[0] for p in df['pair'].tolist()]
    pair1 = [p[1] for p in df['pair'].tolist()]
    df_master['pair0'] = pair0
    df_master['pair1'] = pair1
    
    df_compact = df_master[['pair0','pair1','betamap_corr','pval']]
    df_compact.to_csv(os.path.join(path_out,'corr_pval_null_v_null_{}.csv'.format(tag)))
    
    return df_compact
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_path_mc",help="path to phenotype table",dest='path_pheno')
    parser.add_argument("--cont_n_path_mc",help="path to connectomes .npy file",dest='path_connectomes')
    parser.add_argument("--b_path_mc",help="path to output directory",dest='path_out')
    parser.add_argument("--cont_b_path_mc",help="path to output directory",dest='path_out')
    args = parser.parse_args()
    
    n_path_mc = args.n_path_mc
    cont_n_path_mc = args.cont_n_path_mc
    b_path_mc = args.b_path_mc
    cont_b_path_mc = args.cont_b_path_mc

    cases = ['DEL15q11_2','DUP15q11_2','DUP15q13_3_CHRNA7','DEL2q13','DUP2q13','DUP16p13_11','DEL13q12_12','DUP13q12_12',
        'DEL17p12','TAR_dup','DEL1q21_1','DUP1q21_1','DEL22q11_2','DUP22q11_2','DEL16p11_2','DUP16p11_2',
      'SZ','BIP','ASD','ADHD']
    prs = ['Stand_PRS_newCDG2_ukbb','Stand_PRS_ASD','Stand_PRS_SCZ','Stand_PRS_MDD','Stand_PRS_IQ', 'Stand_PRS_height',
           'Stand_PRS_LDL','Stand_PRS_CKD','Stand_PRS_SA','Stand_PRS_thickness','Stand_PRS_IBD_ukbb']
    cont = prs + ['CT','SA','Vol','fluid_intelligence_score_all','Gfactor','Neuroticism']

    maps = cases + cont

    #############
    # LOAD DATA #
    #############
    null = []
    beta_std = []

    for c in cases:
        null.append(pd.DataFrame(np.load(n_path_mc.format(c))))
        beta_std.append(pd.read_csv(b_path_mc.format(c))['betas_std'].values) #standardized betas


    for c in cont:
        null.append(pd.DataFrame(np.load(cont_n_path_mc.format(c))))
        if c not in prs:
            c = '{}_z'.format(c)
        beta_std.append(pd.read_csv(cont_b_path_mc.format(c))['betas_std'].values) #standardized betas

    betamaps_std = pd.DataFrame(beta_std,index=maps)
    nullmodels = pd.concat(null,keys=maps)
    
    ####################
    # WHOLE CONNECTOME #
    ####################
    df = get_corr_pval(maps,nullmodels,betamaps_std,path_out,tag='wholeconn')
    
    #####################
    # MAKE REGION MASKS #
    #####################
    mask = np.tri(64,k=0,dtype=bool)
        
    THAL = np.zeros((64,64),bool)
    THAL[:,3] = True
    THAL_mask = THAL + np.transpose(THAL)
    THAL_mask = np.tril(THAL_mask)
    THAL_mask = THAL_mask[mask]

    MOTnet_dl = np.zeros((64,64),bool)
    MOTnet_dl[:,55] = True
    MOTnet_dl_mask = MOTnet_dl + np.transpose(MOTnet_dl)
    MOTnet_dl_mask = np.tril(MOTnet_dl_mask)
    MOTnet_dl_mask = MOTnet_dl_mask[mask]
    
    ############
    # THALAMUS #
    ############
    
    # FILTER MAPS
    null_THAL = [n.transpose()[THAL_mask].transpose() for n in null]
    beta_std_THAL = [b[THAL_mask] for b in beta_std]

    betamaps_std_THAL = pd.DataFrame(beta_std_THAL,index=maps)
    nullmodels_THAL = pd.concat(null_THAL,keys=maps)
    
    df_THAL = get_corr_pval(maps,nullmodels_THAL,betamaps_std_THAL,path_out,tag='THAL')
    
    #############
    # MOTnet_DL #
    #############
    
    # FILTER MAPS
    null_MOT = [n.transpose()[MOTnet_dl_mask].transpose() for n in null]
    beta_std_MOT = [b[MOTnet_dl_mask] for b in beta_std]

    betamaps_std_MOT = pd.DataFrame(beta_std_MOT,index=maps)
    nullmodels_MOT = pd.concat(null_MOT,keys=maps)
    
    df_MOT = get_corr_pval(maps,nullmodels_MOT,betamaps_std_MOT,path_out,tag='THAL')
    
    #################
    # MAKE MATRICES #
    #################
    
    # WHOLE CONNECTOME
    subset_WC = ['DEL1q21_1','DUP22q11_2','DEL22q11_2','Stand_PRS_ASD','BIP','SZ','Neuroticism',
              'Stand_PRS_MDD','ASD','Stand_PRS_SCZ','DEL15q11_2','DUP16p11_2','DEL16p11_2',
              'DUP1q21_1','Stand_PRS_SA','SA','CT','Gfactor','fluid_intelligence_score_all','Stand_PRS_IQ']

    corr,pval,fdr = make_matrices(df_compact,subset_WC,fdr='fdr_filtered')
    
    corr.to_csv(os.path.join(path_out,'FC_corr_fig4_wholebrain_mc_null_v_null.csv'))
    pval.to_csv(os.path.join(path_out,'FC_corr_pval_fig4_wholebrain_mc_null_v_null.csv'))
    fdr.to_csv(os.path.join(path_out,'FC_corr_fdr_filtered_fig4_wholebrain_mc_null_v_null.csv'))

    # THALAMUS
    subset_THAL = ['CT','Gfactor','fluid_intelligence_score_all','Stand_PRS_IQ','Stand_PRS_SA',
               'SA','Stand_PRS_ASD','DUP16p11_2','DEL1q21_1','DUP22q11_2','DEL16p11_2',
               'DUP1q21_1','DEL22q11_2','BIP','SZ','Neuroticism','ASD','DEL15q11_2',
              'Stand_PRS_SCZ','Stand_PRS_MDD']

    corr_THAL,pval_THAL,fdr_THAL = make_matrices(df_compact_THAL,subset_THAL,fdr='fdr_filtered')
    
    corr_THAL.to_csv(os.path.join(path_out,'FC_corr_fig5_THAL_mc_null_v_null.csv'))
    pval_THAL.to_csv(os.path.join(path,'FC_corr_pval_fig5_THAL_mc_null_v_null.csv'))
    fdr_THAL.to_csv(os.path.join(path_out,'FC_corr_fdr_filtered_fig5_THAL_mc_null_v_null.csv'))
    
    # MOTnet_DL
    subset_MOT = ['CT','Gfactor','Stand_PRS_IQ','fluid_intelligence_score_all','SA',
               'Stand_PRS_SA','Stand_PRS_ASD','DUP1q21_1','DUP16p11_2','DEL16p11_2','Neuroticism',
               'Stand_PRS_MDD','DEL22q11_2','BIP','SZ','DEL15q11_2','Stand_PRS_SCZ','ASD',
              'DUP22q11_2','DEL1q21_1']

    corr_MOT,pval_MOT,fdr_MOT = make_matrices(df_compact_MOT,subset_MOT,fdr='fdr_filtered')
    
    corr_MOT.to_csv(os.path.join(path_out,'FC_corr_fig6_MOT_mc_null_v_null.csv'))
    pval_MOT.to_csv(os.path.join(path_out,'FC_corr_pval_fig6_MOT_mc_null_v_null.csv'))
    fdr_MOT.to_csv(os.path.join(path_out,'FC_corr_fdr_filtered_fig6_MOT_mc_null_v_null.csv'))