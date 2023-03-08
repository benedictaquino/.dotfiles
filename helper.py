import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil, floor
from pathlib import Path
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, \
    auc, f1_score, log_loss, precision_score, recall_score, accuracy_score, confusion_matrix


def confusion_matrix_dask(truth,predictions,labels_list=[]):
    TP=0
    FP=0
    FN=0
    TN=0
    if not labels_list:
        TP=(truth[predictions==1]==1).sum()
        FP=(truth[predictions!=1]==1).sum()
        TN=(truth[predictions!=1]!=1).sum()
        FN=(truth[predictions==1]!=1).sum()
    for label in labels_list:
        TP=(truth[predictions==label]==label).sum()+TP
        FP=(truth[predictions!=label]==label).sum()+FP
        TN=(truth[predictions!=label]!=label).sum()+TN
        FN=(truth[predictions==label]!=label).sum()+FN

    return np.array([[TN.compute(), FP.compute()] , [TN.compute() ,FN.compute()]])

def plot_confusionmatrix(y_test, y_pred, ax, labels=None, classes=None, normalize=False, title=''):
    '''helper function to plot confusion matrices'''
    cm = confusion_matrix(y_test, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    labels = labels if labels is not None else classes[np.unique(np.append(y_test, y_pred))]
    if normalize:
        ax = sns.heatmap(cm, cbar=False, annot=True, cmap=plt.cm.Reds, yticklabels=labels, xticklabels=labels, ax=ax)
    else:
        ax = sns.heatmap(cm, cbar=False, annot=True, cmap=plt.cm.Reds, yticklabels=labels, xticklabels=labels, ax=ax, fmt='d')
    fmt = '{:.2f}' if normalize else '{:f}'

    for text in ax.texts:
        if normalize:
            text.set_text(str(fmt.format(float(text.get_text()))).strip('0').rstrip('.'))

        if floor(text.get_position()[0]) == floor(text.get_position()[1]):
            text.set_size(12)
            text.set_weight('bold')
            text.set_style('italic')

    if normalize:
        ax.set_title('{} Confusion Matrix (Normalized)'.format(title))
    else:
        ax.set_title('{} Confusion Matrix'.format(title))
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()


def plot_ROC(y_true, score, ax, asc_x=True, lab='', title=''):
    '''plots ROC curve and shows AUC'''
    roc_auc = roc_auc_score(y_true, score if asc_x else -score)
    fpr, tpr, thresholds = roc_curve(y_true, score if asc_x else -score)

    ax.plot(fpr, tpr, label='{} (AUC = {:.3f})'.format(lab, roc_auc))
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    d = 0.02
    ax.set_xlim([-d, 1 + d])
    ax.set_ylim([-d, 1 + d])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve {}'.format(title))
    ax.legend(loc='lower right')


def plot_PR_curve(y_true, score, ax, asc_x=True, lab='', title=''):
    '''plots precision-recall curve'''
    p, r, t = precision_recall_curve(y_true, score if asc_x else -score)
    auc_x = auc(r, p)
    ax.plot(r, p, label='{} (AUC = {:.3f})'.format(lab, auc_x))
    prec_ran = y_true.mean()
    ax.plot([0, 1], [prec_ran, prec_ran], color='grey', linestyle='--')
    d = 0.02
    ax.set_xlim([-d, 1 + d])
    ax.set_ylim([-d, 1 + d])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('P-R Curve {}'.format(title))
    ax.legend(loc='upper right')


def plot_all(y_test, y_train, y_test_hat, y_train_hat, p_test_hat, p_train_hat, normalize=False, title=''):
    '''takes the above three functions and plots everything'''
    ax = plt.subplots(2, 2, figsize=(20, 12))[1].flat
    sns.set()
    plot_PR_curve(y_train, p_train_hat[:,1], ax=ax[0], lab='Train')
    plot_ROC(y_train, p_train_hat[:,1], ax=ax[1], lab='Train')
    plot_PR_curve(y_test, p_test_hat[:,1], ax=ax[0], lab='Test')
    plot_ROC(y_test, p_test_hat[:,1], ax=ax[1], lab='Test')
    plot_confusionmatrix(y_train, y_train_hat, ax[2], labels=[False, True], normalize=normalize, title='Train')
    plot_confusionmatrix(y_test, y_test_hat, ax[3], labels=[False, True], normalize=normalize, title='Test')
    Path('/mnt/artifacts/plots').mkdir(parents=True, exist_ok=True)
    if normalize:
        plt.savefig(f'/mnt/artifacts/plots/{title}.png')
    else:
        plt.savefig(f'/mnt/artifacts/plots/{title}_normalized.png')
    plt.show()



def get_gains(scores, y, asc_score=False, top=None, normalize=True):
    """ 
    Computes and returns gain, defined as the expanding cumulative sum of 1s, 
    given that the labels are sorted on a certain score vector 'scores'.
    As scores can have multiplicity for certain values, the sortation within a set of records
    with same values is arbitrary. Hence, to deal with this, this procedure returns the
    best possible gain, worst possible gain and average gain. Where average gain is defined
    as expected gain given a random 2nd level subsort of the y variable.

    When normalize == True, gain is equivalent to the rolling accuracy/conversion.
    """
    f = 'mean' if normalize else 'sum'
    df = pd.DataFrame({'scores': scores, 'y': y.values})
    best = df.sort_values(['scores', 'y'], ascending=[asc_score, False])\
               ['y'].expanding().agg(f).values[:top]
    worst = df.sort_values(['scores', 'y'], ascending=[asc_score, True])\
               ['y'].expanding().agg(f).values[:top]
    ideal = df['y'].sort_values(ascending=False).expanding().agg(f).values[:top]

    avg = df.pivot_table('y', 'scores', aggfunc=('mean', 'count')).sort_index(ascending=asc_score)
    avg = avg['mean'].repeat(avg['count']).expanding().agg(f).values[:top]

    return best, worst, avg, ideal


def get_lift(score, y, asc_score=False, top=None, norm_x=True, man_fracs=None, skip_frac=None):
    '''returns the lift values based off the gains'''
    avg_lift = get_gains(score, y, asc_score, top)[2]
    x = np.arange(1, len(avg_lift) + 1) 
    x = x / len(avg_lift) if norm_x else x
    avg_lift = (avg_lift / y.mean() - 1) * 100

    if man_fracs:
        man_idx = [ceil(f * (len(avg_lift) - 1)) for f in man_fracs]
        x = x[man_idx]
        avg_lift = avg_lift[man_idx]

    # Skip initial entries as these are often very jumpy
    # But only skip if no manual index is specified
    elif skip_frac:
        skip_idx = ceil(skip_frac * len(avg_lift))
        x = x[skip_idx:]
        avg_lift = avg_lift[skip_idx:]

    return x, avg_lift


def get_negative_lift(score, y, asc_score=False, top=None, norm_x=True, man_fracs=None, skip_frac=None):
    '''flips the above function to show how the worst predictions lift looks like'''
    avg_lift = get_gains(1 - score, y, asc_score, top)[2]
    x = np.arange(1, len(avg_lift) + 1) 
    x = x / len(avg_lift) if norm_x else x
    avg_lift = (avg_lift / y.mean() - 1) * 100

    if man_fracs:
        man_idx = [ceil(f * (len(avg_lift) - 1)) for f in man_fracs]
        x = x[man_idx]
        avg_lift = avg_lift[man_idx]

    # Skip initial entries as these are often very jumpy
    # But only skip if no manual index is specified
    elif skip_frac:
        skip_idx = ceil(skip_frac * len(avg_lift))
        x = x[skip_idx:]
        avg_lift = avg_lift[skip_idx:]

    return x, avg_lift


def plot_lgbm_imp(model, feature_names , num = 20, fig_size = (16, 10), title=''):
    feature_imp = pd.DataFrame({'Value':model.feature_importances_,'Feature':feature_names})
    fig, ax = plt.subplots(figsize=fig_size)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num], color='darkblue', ax=ax)
    ax.set(title='Top Features')
    fig.tight_layout()
    Path('/mnt/artifacts/plots').mkdir(parents=True, exist_ok=True)
    fig.savefig(f'/mnt/artifacts/plots/lgbm_{title}.png')
    return fig
    


def iqr_outlier_rejection(X, y, cut_off_val=1.5):
    
    features = X.columns
    df = X.copy()
    df['target'] = y
    
    indices = [x for x in df.index]    
    out_indexlist = []
        
    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]) == False:
            continue
       
        #Using nanpercentile instead of percentile because of nan values
        Q1 = np.nanpercentile(df[col], 25.)
        Q3 = np.nanpercentile(df[col], 75.)
        
        cut_off = (Q3 - Q1) * cut_off_val
        upper, lower = Q3 + cut_off, Q1 - cut_off
                
        outliers_index = df[col][(df[col] < lower) | (df[col] > upper)].index.tolist()
        outliers = df[col][(df[col] < lower) | (df[col] > upper)].values        
        out_indexlist.extend(outliers_index)
        
    #using set to remove duplicates
    out_indexlist = list(set(out_indexlist))
    
    clean_data = np.setdiff1d(indices,out_indexlist)

    return X.loc[clean_data], y.loc[clean_data]


def iqr_outlier_cap(X, y, cut_off_val=1.5, strategy='both'):
    
    features = X.columns
    X_cap = X.copy()
        
    for col in features:
        if pd.api.types.is_numeric_dtype(X_cap[col]) == False:
            continue
       
        #Using nanpercentile instead of percentile because of nan values
        Q1 = np.nanpercentile(X_cap[col], 25.)
        Q3 = np.nanpercentile(X_cap[col], 75.)
        
        cut_off = (Q3 - Q1) * cut_off_val
        upper, lower = Q3 + cut_off, Q1 - cut_off
                
        # change values
        if strategy == 'both':
            X_cap.loc[X_cap[col] > upper, col] = upper
            X_cap.loc[X_cap[col] < lower, col] = lower
        elif strategy == 'top':
            X_cap.loc[X_cap[col] > upper, col] = upper
        elif strategy == 'bottom':
            X_cap.loc[X_cap[col] < lower, col] = lower
        

    return X, y


def score_plot_all(X_train, X_test, y_train, y_test, model, normalize=True, title=''):
    y_train_hat = model.predict(X_train)
    p_train_hat = model.predict_proba(X_train)
    y_test_hat = model.predict(X_test)
    p_test_hat = model.predict_proba(X_test)
    print('Train Log Loss: ', log_loss(y_train, p_train_hat))
    print('Test Log Loss: ', log_loss(y_test, p_test_hat))
    print('Train F1 Score: ', f1_score(y_train, y_train_hat))
    print('Test F1 Score: ', f1_score(y_test, y_test_hat))
    print('Train Accuracy: ', accuracy_score(y_train, y_train_hat))
    print('Test Accuracy: ', accuracy_score(y_test, y_test_hat))
    print('Train Recall Score: ', recall_score(y_train, y_train_hat))
    print('Test Recall Score: ', recall_score(y_test, y_test_hat))
    print('Train Precision Score: ', precision_score(y_train, y_train_hat))
    print('Test Precision Score: ', precision_score(y_test, y_test_hat))
    sns.set()
    plot_all(y_test, y_train, y_test_hat, y_train_hat, p_test_hat, p_train_hat, normalize, title)
    lift_df = calc_lift(p_test_hat, y_test)

    return {
        'p_train_hat': p_train_hat,
        'p_test_hat': p_test_hat,
        'y_train_hat': y_train_hat,
        'y_test_hat': y_test_hat,
        'lift': lift_df,
        }


def calc_lift(p_test_hat, y_test):
    man_fracs=[.01, .05, .10, .15, .25, .5, .75]
    lift = pd.DataFrame(index=['Top 1%', 'Top 5%', 'Top 10%', 'Top 15%', 'Top 25%', 'Top 50%', 'Top 75%'])
    lift['Lift'] = get_lift(p_test_hat[:,1], y_test, man_fracs=man_fracs)[1]
    lift.round(0).astype(int)

    return lift
