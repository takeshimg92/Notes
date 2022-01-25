import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import auc, roc_curve, RocCurveDisplay, roc_auc_score, PrecisionRecallDisplay
from sklearn.metrics import average_precision_score, precision_recall_curve
from scipy.spatial import ConvexHull
from warnings import catch_warnings, simplefilter



def hull_roc_auc(y_true, y_score):
    """
    Computes coordinates (TPR, FPR) and ROC AUC for the convex hull 
    of a ROC curve built from a ground truth y_true (0s and 1s) and 
    a vector of scores y_score
    """

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # add artificial vertex at (1,0)
    fpr, tpr = np.append(fpr, [1]), np.append(tpr, [0])

    points = np.array([fpr, tpr]).T
    hull = ConvexHull(points)

    # get vertices and remove artificial vertex
    vertices = np.array([points[v] for v in hull.vertices if not np.array_equal(points[v],np.array([1., 0.]))])
    fpr_hull, tpr_hull = vertices[:,0], vertices[:,1]

    # hull AUC
    hull_auc = auc(fpr_hull, tpr_hull)
    
    return hull_auc, fpr_hull, tpr_hull


def build_gain_curve(y_actual, y_pred, 
                     ascending=True,
                     return_ideal_curve=False):
    
    """
    Returns the gain curve from actual (0 or 1) data and
    predicted scores.
    
    Also returns what the ideal gain curve for this problem 
    would be, if return_ideal_curve = True
    """
    
    df = pd.DataFrame({'y_actual': y_actual,
                   'prob_default': y_pred})

    # sort from low to high scores
    df = df.sort_values('prob_default', ascending=ascending)
    
    # build cumulative_default
    df['cumulative_default'] = df['y_actual'].cumsum()
    df['gain'] = df['cumulative_default']/df['y_actual'].sum()

    # create index starting from 0 and normalize
    df = df.reset_index(drop=True).reset_index()
    df['index'] = df['index']/(df['index'].iloc[-1])
    df = df.set_index('index')
    
    if return_ideal_curve:
        df_perfect = df.sort_values('y_actual', ascending=ascending)
        df_perfect['cumulative_default'] = df_perfect['y_actual'].cumsum()
        df_perfect['gain'] = df_perfect['cumulative_default']/df_perfect['cumulative_default'].iloc[-1]
        df_perfect = df_perfect.reset_index(drop=True).reset_index()
        df_perfect['index'] = df_perfect['index']/(df_perfect['index'].iloc[-1])
        df_perfect = df_perfect.set_index('index')
        
        return df['gain'], df_perfect['gain']
    
    return df['gain']


def plot_roc_and_gain(y_true, y_pred, ax, ascending=False):
    
    
    # gain curve
    gain, ideal_gain = build_gain_curve(y_true, y_pred, ascending=ascending, return_ideal_curve=True)
    aug = auc(x=gain.index, y=gain)
    aug_ideal = auc(x=ideal_gain.index, y=ideal_gain)
    gain.plot(ax=ax, label='Gain (area = {0:.2f})'.format(aug), color='blue')
    ideal_gain.plot(ax=ax, label='Ideal gain (area = {0:.2f})'.format(aug_ideal), linestyle='-.', color='lightblue')

    # roc curve
    auc_ = roc_auc_score(y_true, y_pred)
    RocCurveDisplay.from_predictions(y_true, y_pred, ax=ax, label='ROC (area = {0:.2f})'.format(auc_), color='orange')

    # random classifier
    ax.plot(np.linspace(0,1), np.linspace(0,1), linestyle='--', color='gray', label='Random (area = 0.50)')

    # plot settings
    ax.set_title("ROC and Gain curves plot together")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend()
    
    
def plot_precision_recall(y_true, y_probs, ax):
    
    PrecisionRecallDisplay.from_predictions(y_true, y_probs, ax=ax)
    ax.set_ylim(-.05,1.05)
    
    # prevalence of positive class
    p = y_true.sum()/len(y_true)
    ax.axhline(p, linestyle='--', color='gray', label=f'Prevalence ({round(p,2)})')
    plt.legend()


def harm_scale(x, a, b):
    
    with catch_warnings():
        simplefilter("ignore")
        res = (x-a)*b/(x* (b-a))
    return res


def from_pr_to_harmonic_pr(rec, prec, r):
    
    pi1 = r/(r+1)
    hprec = harm_scale(prec, pi1, 1)
    hrec  = harm_scale(rec, pi1, 1)
    
    # restrict to [0,1]
    restriction = hrec >= 0
    hrec, hprec = hrec[restriction], hprec[restriction]
    
    return hrec, hprec


def from_roc_to_harmonic_pr(fpr, tpr, r):
    
    rec, prec = from_roc_to_pr(fpr, tpr, r)

    return from_pr_to_harmonic_pr(rec, prec, r)


def harmonic_precision_recall(y_true, y_pred):
    
    from sklearn.metrics import roc_curve, precision_recall_curve
    
    prec, rec, thresh = precision_recall_curve(y_true, y_pred)

    r = (y_true==1).sum()/(y_true==0).sum()
    pi1 = r/(r+1)
    
    hprec = harm_scale(prec, pi1, 1)
    hrec  = harm_scale(rec, pi1, 1)
    
    # restrict to [0,1]
    restriction = hrec >= 0
    hrec, hprec, thresh = hrec[restriction], hprec[restriction], thresh[restriction[:-1]]
    
    return hprec, hrec, thresh


def plot_harmonic_precision_recall(y_true, y_pred, ax, label=False):
        
    hprec, hrec, _ = harmonic_precision_recall(y_true, y_pred)
    area = auc(hrec, hprec)
    
    ax.plot(hrec,hprec, label='AUhPR (area={0:.2f})'.format(area))
    
    x = np.linspace(0,1)
    if label:
        ax.plot(x,1-x, linestyle='--', color= 'gray', label='All-positive')
    else:
        ax.plot(x,1-x, linestyle='--', color= 'gray')
    ax.set_xlim(-0.05,1.05)
    
    
def harmonic_auc(y_true, y_probs):
    """Calculates the area under the harmonic PR curve"""
    
    hprec, hrec, _ = harmonic_precision_recall(y_true, y_probs)
    return auc(hrec, hprec)
    
    
def roc_pr_hpr_report(y_true, y_probs, ascending_gain=False):
    
    fig, ax = plt.subplots(figsize=(12,4), ncols=3)

    ## ROC plane (+ gain curve)
    plot_roc_and_gain(y_true, y_probs, ax=ax[0], ascending=ascending_gain)
    ax[0].set_title("ROC + gain curve")
    ax[0].legend()

    ## PR plane
    # PR curve
    prec, rec, _ = precision_recall_curve(y_true, y_probs)

    # baseline curve
    pi = y_true.mean()
    x = np.linspace(pi,1)
    y = 1/(-1/x + (1+1/pi))

    ax[1].plot(rec, prec, label='Model')
    ax[1].plot(x, y, linestyle='--', color='gray', label='Baseline F1 model')
    ax[1].set_title("PR curve")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
    ax[1].legend()

    ## harmonic PR plane
    plot_harmonic_precision_recall(y_true, y_probs, ax[2])
    ax[2].set_title("Harmonic PR curve")
    ax[2].set_xlabel("Harmonic recall"); ax[2].set_ylabel("Harmonic precision")
    ax[2].legend()

    plt.tight_layout()
    plt.show()
    