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

from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

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
    
def plot_confusion_matrix(cm,target_names,title='Confusion matrix',save=False,save_path = None):
    """
    Given an sklearn confusion matrix (cm), make a nice plot
    
    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
    target_names: list of class names
    title:        string title

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    cmap = plt.get_cmap('Blues')
    
    size = (8, 5)
    if (len(target_names)>10):
        size = (13, 11)
    plt.figure(figsize=size)
    plt.imshow(cm,cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if save:
        plt.savefig('{}.png'.format(save_path + title))
    else:
        plt.show()
    
#FUNCTION IS SPECIFIC TO MY REFORMAT OF CLARA'S PHENO FILE
def select_groups(pheno,data,groups,control = 'non_carriers',con_strategy='match',lump=False):
    """
    Select a subset of the dataset and pheno file.
    
    Arguments
    --------------
    pheno:        pheno file loaded as dataframe
    data:         numpy array n_subjects x n_features
    groups:       list of columns labels from dataframe
    control:      column label from dataframe
    con_strategy: 'all','match','n_x' - x is int of choice, 'none'
    lump:         bool, put all groups into single category
    
    Returns
    --------------
    X:  subset of data corresponding to groups & control
    y:  labels for subset of data
    counts: num subjects for each group (incl. controls)
    """
    if (pheno.shape[0] != data.shape[0]):
        raise Exception('Pheno and data have mismatch size.')
        
    counts = []
        
    group_masks = []
    for group in groups:
        mask = pheno[group].to_numpy()
        group_masks.append(mask)
        counts.append(pheno[group].sum())
        
    num_patients = pheno[groups].sum(axis=0).sum()
    
    if (con_strategy == 'all'):
        control_mask = pheno[control].to_numpy()
        counts.append(control_mask.sum())
    elif (con_strategy == 'match'):
        if (groups == ['DEL22q11_2']):
            control_mask = np.array((pheno['PI']=='UCLA')&(pheno['non_carriers']==1))
        else:
            idx = random.sample([ l[0] for l in np.argwhere(pheno[control].to_numpy()).tolist()],k=num_patients)
            control_mask = np.zeros(data.shape[0])
            control_mask[idx]=1
            counts.append(control_mask.sum())
    elif ('n_' in con_strategy):
        n = int(con_strategy.split('_')[1])
        all_controls = [ l[0] for l in np.argwhere(pheno[control].to_numpy()).tolist()]
        if (n > len(all_controls)):
            n = len(all_controls)
        idx = random.sample(all_controls,k=n)
        control_mask = np.zeros(data.shape[0])
        control_mask[idx]=1
        counts.append(control_mask.sum())
    elif (con_strategy == 'none'):
        control_mask = np.zeros(data.shape[0])
        counts.append(control_mask.sum())
    
    group_mask = np.sum(group_masks,axis=0)
    mask = group_mask + control_mask
    mask = mask.astype(bool)
    
    X = data[mask]
        
    if lump:
        #case-control so should be fine to return single one hot col
        y = pheno[mask][groups].sum(axis=1)
    else:
        y = pheno[mask][groups + [control]].idxmax(axis=1).to_numpy()
    
    return X,y,counts

def try_model(X,y,model='random_forest',k=5):
    if (model=='random_forest'):
        classifier = RandomForestClassifier(n_estimators=50,class_weight='balanced')
    elif (model=='LDA'):
        classifier = LDA() #can adjust priors for imbalanced data?
    elif (model=='kNN'):
        classifier = KNeighborsClassifier(n_neighbors=3) #not sure how to adjust for imbalanced data
    elif (model=='naive_bayes'):
        classifier = GaussianNB() #can adjust priors for imbalanced data?
    elif (model=='SVC'):
        classifier = SVC(C=100,class_weight='balanced')
    elif (model=='ada_boost'):
        base = RandomForestClassifier(n_estimators=50,class_weight='balanced')
        classifier = AdaBoostClassifier(base_estimator=base)
    elif (model=='logit'):
        classifier = LogisticRegression(C=100,class_weight='balanced',max_iter=500) #this is L2 penalty

    labels = np.unique(y)
    
    i=0
    if (not (k==None)):
        confusion = np.zeros((len(labels),len(labels)))
        accuracy=0
        precision=np.zeros(len(labels))
        recall=np.zeros(len(labels))
        kf = StratifiedKFold(n_splits=k,shuffle=True)
        for train_index, test_index in kf.split(X,y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            
            confusion = confusion + confusion_matrix(y_test, y_pred,labels=labels)
            accuracy = accuracy + accuracy_score(y_test, y_pred)
            prec,rec,_,_ = precision_recall_fscore_support(y_test, y_pred,labels=labels)
            precision = precision + prec
            recall = recall + rec
            i = i + 1
            
        accuracy = accuracy/k
        precision = precision/k
        recall = recall/k
        confusion = confusion/k
        cm = confusion
    
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred,labels=labels)
        accuracy = accuracy_score(y_test, y_pred)
        precision,recall,_,_ = precision_recall_fscore_support(y_test, y_pred)
    
    return cm, accuracy, precision, recall

def run_models(X,y,models=['LDA','SVC','random_forest'],out_path = None,save=False,k=5,title=None):
    labels=np.unique(y)
    columns= ['model', 'accuracy']
    for label in labels:
        columns.append('precision_{}'.format(label))
        columns.append('recall_{}'.format(label))
    rows = []
    
    if ('all' in models):
        models = ['LDA','SVC','random_forest','kNN','naive_bayes','logit','ada_boost']
        
    for m in models:
        if (not (k==None)):
            if (k=='LOO'):
                k = X.shape[0]
        cm, accuracy,precision,recall = try_model(X,y,model=m,k=k)
        title = 'confusion_mat_{}'.format(m)
        plot_confusion_matrix(cm,labels, save_path=out_path, title=title, save=save)
        row = [m,accuracy]
        for i in range(len(labels)):
            row.append(precision[i])
            row.append(recall[i])
        rows.append(row)

    results = pd.DataFrame(rows,columns=columns)
    if save:
        if (title==None):
            title='results'
        results.to_csv(out_path+'{}.csv'.format(title),index=False)
    return results