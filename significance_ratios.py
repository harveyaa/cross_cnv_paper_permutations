import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import itertools
from argparse import ArgumentParser

#SEBASTIEN URCHS
def p_permut(empirical_value, permutation_values):
    n_permutation = len(permutation_values)
    if empirical_value >= 0:
        return (np.sum(permutation_values > empirical_value)+1) / (n_permutation + 1)
    return (np.sum(permutation_values < empirical_value)+1) / (n_permutation + 1)

# argparse statements
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_path",help="path to mc null models dir",dest='n_path')
    parser.add_argument("--b_path",help="path to mc betamaps dir",dest='b_path')
    parser.add_argument("--path_out",help="path to output directory",dest='path_out')
    args = parser.parse_args()

    n_path_mc = os.path.join(args.n_path,'{}_null_model_mc.npy')
    cont_n_path_mc = os.path.join(args.n_path,'{}_null_model_mc.npy')
    b_path_mc = os.path.join(args.b_path,'cc_{}_results_mc.csv')
    cont_b_path_mc = os.path.join(args.b_path,'cont_{}_results_mc.csv')
    path_out = args.path_out
    
    cases = ['IBD','DEL15q11_2','DUP15q11_2','DUP15q13_3_CHRNA7','DEL2q13','DUP2q13','DUP16p13_11','DEL13q12_12','DUP13q12_12',
        'TAR_dup','DEL1q21_1','DUP1q21_1','DEL22q11_2','DUP22q11_2','DEL16p11_2','DUP16p11_2',
      'SZ','BIP','ASD','ADHD']
    prs = ['Stand_PRS_newCDG2_ukbb','Stand_PRS_ASD','Stand_PRS_SCZ','Stand_PRS_BIP','Stand_PRS_IQ',
         'Stand_PRS_LDL','Stand_PRS_CKD']
    cont = prs + ['fluid_intelligence_score_all','Neuroticism']

    maps = cases + cont

    #############
    # LOAD DATA #
    #############
    print('Loading Data...')
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

    betamaps = pd.DataFrame(beta_std,index=maps)
    nullmodels = pd.concat(null,keys=maps)
    print('Done!')
    
    ####################
    # CREATE MTD TABLE #
    ####################
    print('Creating mean top decile table...')
    table = []
    for i in betamaps.index:
        rank = pd.qcut(betamaps.loc[i].abs(),10,labels=False)
        idx = rank[rank==9].index
        decile = betamaps.loc[i].abs()[idx]
        mean_top_dec = np.mean(decile)
        desc = [betamaps.loc[i].min(),betamaps.loc[i].max(),betamaps.loc[i].mean(),betamaps.loc[i].abs().mean(),mean_top_dec]
        table.append(desc)
    table = pd.DataFrame(table,columns=['min','max','mean','mean_abs','mean_top_dec'],index=betamaps.index)
    print('Saving...')
    table.to_csv(os.path.join(path_out,'mean_top_dec.csv'))
    print('Done!')
    
    ##################
    # GET MTD RATIOS #
    ##################
    print('Generating ratios of mean top decile...')

    rows = []
    for i in itertools.combinations(betamaps.index,2):    
        big = 1
        small = 0
        if np.abs(table['mean_top_dec'].loc[i[0]]) > np.abs(table['mean_top_dec'].loc[i[1]]):
            big = 0
            small = 1

        mean_top_dec_ratio = np.abs(table['mean_top_dec'].loc[i[big]])/np.abs(table['mean_top_dec'].loc[i[small]])

        row = [i[big],i[small],mean_top_dec_ratio]
        rows.append(row)

    summary_ratios = pd.DataFrame(rows,columns=['big','small','mean_top_dec_ratio'])
    summary_ratios.to_csv(os.path.join(path_out,'mean_top_dec_ratios.csv'))
    print('Done!')
    
    ########################
    # GET NULL DIST OF MTD #
    ########################
    if os.path.exists(os.path.join(path_out,'mean_top_dec_dist.csv')):
        print('Loading null distributions of mean top decile...')
        df_table_dist = pd.read_csv(os.path.join(path_out,'mean_top_dec_dist.csv'),index_col=0)
        print('Done!')
    else:
        print('Null distributions of mean top decile not found.')
        print('Generating null distributions...')
        
        table_dist = []
        for i in betamaps.index:
            print(i)
            for j in range(5000):
                rank = pd.qcut(nullmodels.loc[i].iloc[j].abs(),10,labels=False)
                idx = rank[rank==9].index
                decile = nullmodels.loc[i].iloc[j].abs()[idx]
                mean_top_dec = np.mean(decile)
                desc =[i,j,nullmodels.loc[i].iloc[j].min(), nullmodels.loc[i].iloc[j].max(),nullmodels.loc[i].iloc[j].mean(),
                       nullmodels.loc[i].iloc[j].abs().mean(),mean_top_dec]
                table_dist.append(desc)

                if (j%100 ==0):
                    print(j)
        df_table_dist = pd.DataFrame(table_dist,columns=['case','iter','min','max','mean','mean_abs','mean_top_dec'])
        print('Saving...')
        df_table_dist.to_csv('mean_top_dec_dist.csv')
        print('Done!')
    
    ###############################
    # GET NULL DIST OF MTD RATIOS #
    ###############################
    if os.path.exists(os.path.join(path_out,'mtd_dist_ratios_observed_denom.csv')):
        print('Loading null distributions of mean top decile ratios...')
        summary_ratios_dist = pd.read_csv(os.path.join(path_out,'mtd_dist_ratios_observed_denom.csv'),index_col=0)
        print('Done!')
    else:
        print('Null distributions of mean top decile ratios not found.')
        print('Generating null distributions of ratios...')
        
        rows_dist = []
        for i in itertools.combinations(betamaps.index,2):
            print(i)
            big = 1
            small = 0
            if np.abs(table['mean_top_dec'].loc[i[0]]) > np.abs(table['mean_top_dec'].loc[i[1]]):
                big = 0
                small = 1

            # Always use observed value for denominator
            mtd_small = table['mean_top_dec'].loc[i[small]]
            for j in range(5000):
                mtd_big = df_table_dist[(df_table_dist['case'] == i[big])&(df_table_dist['iter'] == j)]['mean_top_dec'].values[0]
                mean_top_dec_ratio = mtd_big/mtd_small

                row = [i[big],i[small],j,mean_top_dec_ratio]
                rows_dist.append(row)

                if (j%100 == 0):
                    print(j)

        summary_ratios_dist = pd.DataFrame(rows_dist,columns=['big','small','iter','mean_top_dec_ratio'])
        print('Saving...')
        summary_ratios_dist.to_csv(os.path.join(path_out,'mtd_dist_ratios_observed_denom.csv'))
        print('Done!')

    ########################
    # GET PVALS FOR RATIOS #
    ########################
    print('Getting pvals for mean top decile ratios...')
    
    rows_sig = []
    for i in itertools.combinations(table.index,2):
        big = 0
        small = 1
        if summary_ratios[(summary_ratios['big'] == i[0]) & (summary_ratios['small'] == i[1])].shape[0] == 0:
            big = 1
            small = 0

        val = summary_ratios[(summary_ratios['big'] == i[big]) & (summary_ratios['small'] == i[small])]['mean_top_dec_ratio'].to_numpy()[0]
        dist = summary_ratios_dist[(summary_ratios_dist['big'] == i[big]) & (summary_ratios_dist['small'] == i[small])]['mean_top_dec_ratio'].to_numpy()
        p = p_permut(val,dist)

        row = [i[big],i[small],val,p]
        rows_sig.append(row)

    sig_mean_top_dec_ratio = pd.DataFrame(rows_sig, columns = ['big','small','mean_top_dec_ratio','p_mean_top_dec_ratio'])
    print('Saving...')
    sig_mean_top_dec_ratio.to_csv(os.path.join(path_out,'mtd_dist_ratios_observed_denom_sig.csv'))
    print('Done!')