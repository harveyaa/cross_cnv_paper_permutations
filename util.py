import random
import numpy as np
import pandas as pd
import patsy as pat
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

def thresh_simple(mat,thresh=0.1,kind='percentage'):
    """
    Simple connectome thresholding.
    
    mat = n_subjects x n_edges connectome
    thresh = threshold
    kind = strategy, either 'percentage' or 'cutoff'
    
    Returns
    mat_t = n_subjects x n_edges thresholded connectome
    """
    if(kind=='percentage'):
        n = int(np.ceil(thresh*mat.shape[1]))
        mat_t = np.array(mat)
        for row in mat_t:
            top = np.sort(row)[-n:]
            #mat_t = mat_t.reshape(-1,mat.shape[1])
            row[row<np.min(top)]=0
    elif(kind=='cutoff'):
        mat_t = np.array(mat)
        mat_t[mat_t<thresh] = 0
    return mat_t

def vec_to_connectome(a,dim=64):
    """
    Turn a vector representation of lower triangular matrix to connectome.
    
    a = vector
    dim = dimension of connectome
    
    Returns:
    dim x dim connectome
    """
    A = np.zeros((dim,dim))
    mask = np.tri(dim,dtype=bool, k=0)
    A[mask]=a
    B = np.array(A).transpose()
    np.fill_diagonal(B,0)
    return A + B

def standardize(mask,data):
    scaler = StandardScaler(with_mean=False, with_std=True)
    scaler.fit(data[mask])
    standardized=scaler.transform(data)
    return standardized

def mask_cc(pheno,case,control):
    """
    pheno = df w/ all subjects
    case = col (onehot)
    control = col (onehot)
    
    Returns:
    mask = bool mask with True for subs that are case + control
    """
    mask_case = pheno[case].to_numpy(dtype=bool)
    mask_con = pheno[control].to_numpy(dtype=bool)
    return mask_case + mask_con

def mask_var(pheno,var):
    """
    pheno = df w/ all subjects
    case = col (onehot)
    control = col (onehot)
    
    Returns:
    mask = bool mask with True where subjects have var info
    """
    mask = np.array(~pheno[var].isnull())
    return mask

def case_control(pheno,case,regressors,conn,std=False):
    """
    pheno = dataframe:
        -filtered to be only relevant subjects for case control (use mask_cc)
        -case column is onehot encoded
    case = column from pheno
    regressors = list of strings, formatted for patsy
    connectomes = n_subjects x n_edges array
    
    Returns:
    table = n_edges
        - betas = the difference between case + control
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
        model = sm.OLS(conn[:,edge],dmat)
        results = model.fit()
        betas[edge] = results.params[case]
        pvalues[edge] = results.pvalues[case]
        
        if std:
            model_std = sm.OLS(conn_std[:,edge],dmat)
            results_std = model_std.fit()
            betas_std[edge] = results_std.params[case]
        
    mt = multipletests(pvalues,method='fdr_bh')
    reject = mt[0]
    qvalues = mt[1]
    
    table = pd.DataFrame(np.array([betas,betas_std,pvalues,qvalues,reject]).transpose(),
                         columns=['betas','betas_std','pvalues','qvalues','reject'])
    return table


def variable_effect(pheno,var,regressors,conn,std=False):
    """
    Test effect of continuous variable.
    
    pheno = dataframe:
        -filtered to be only relevant subjects (use mask_var)
    var = column from pheno
    regressors = list of strings, formatted for patsy
    connectomes = n_subjects x n_edges array
    
    Returns:
    table = n_edges
        - betas = the difference between case + control
        - betas_std = including standardization on controls
        - pvalues = pvalues
        - qvalues = fdr corrected pvalues alpha = 0.05
    """
    
    n_edges = conn.shape[1]
    contrast = np.zeros(1 + len(regressors))
    contrast[0] = 1
    
    betas = np.zeros(n_edges)
    betas_std = np.zeros(n_edges)
    pvalues = np.zeros(n_edges)
        
    formula = ' + '.join((regressors + [var]))
    dmat = pat.dmatrix(formula, pheno, return_type='dataframe',NA_action='raise')
    
    mask_std = np.ones(pheno.shape[0]).astype(bool)
    conn_std = standardize(mask_std, conn)
    
    
    for edge in range(n_edges):
        model = sm.OLS(conn[:,edge],dmat)
        results = model.fit()
        betas[edge] = results.params[var]
        pvalues[edge] = results.pvalues[var]
        
        if std:
            model_std = sm.OLS(conn_std[:,edge],dmat)
            results_std = model_std.fit()
            betas_std[edge] = results_std.params[var]
        
    mt = multipletests(pvalues,method='fdr_bh')
    reject = mt[0]
    qvalues = mt[1]
    
    table = pd.DataFrame(np.array([betas,betas_std,pvalues,qvalues,reject]).transpose(),
                         columns=['betas','betas_std','pvalues','qvalues','reject'])
    return table


def plot_dist(dists,labels,out_path=None,save=False,xlabel='Beta estimates',stacked=False,title=None):
    """
    Plot distributions of betas/effects.
    
    dists = list distributions to plot
    labels = list of labels
    out_path = path to save folder
    save = bool, saves to out_path if True
    xlabel = xlabel of plot
    """
    if stacked:
        n = len(dists)
        fig, axs = plt.subplots(n,sharex=True,sharey=True,figsize=(5,n),
                        gridspec_kw={'hspace': -0.1})
        sns.despine(left=True)

        for (i,dist,label) in zip(range(len(dists)),dists,labels):
            sns.distplot(dist,hist=False,kde=True,kde_kws={"shade":True}, ax=axs[i])
            axs[i].axes.get_yaxis().set_visible(False)
            axs[i].set(ylabel=label,xlabel='')
            axs[i].patch.set_alpha(0)
        for (i,dist,label) in zip(range(len(dists)),dists,labels):
            xmin, xmax = axs[i].get_xaxis().get_view_interval()
            pos = axs[i].get_position()
            axs[i].text(1.1*xmin,pos.bounds[1],label,ha="right", va="bottom")
        plt.xlabel(xlabel)
    
    else:
        fig = plt.figure(facecolor='white',figsize=(10,8))
        ax = plt.axes(frameon=False)
        ax.axes.get_yaxis().set_visible(False)

        for (dist,label) in zip(dists,labels):
            sns.distplot(dist,hist=False,label=label)

        plt.xlabel(xlabel)
        xmin, xmax = ax.get_xaxis().get_view_interval()
        ymin, ymax = ax.get_yaxis().get_view_interval()
        ax.add_artist(Line2D((xmin, xmax), (ymin, ymin),color='black', linewidth=2))
        plt.axvline(0, 5,0,ls='--',color='black')
    
    if save:
        if (title == None):
            if stacked:
                title = '/stacked_{}.png'.format('_'.join([xlabel]+labels))
            else:
                title = '/{}.png'.format('_'.join([xlabel]+labels))
        else:
            title = '/'+ title + '.png'
        plt.savefig(out_path + title,bbox_inches='tight')
    plt.show()
