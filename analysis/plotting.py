import matplotlib.pyplot as plt
from matplotlib import colors
import os
import numpy as np
import sklearn.metrics as skl


def is_outlier_percentile(points, percentile=99.9):
    diff = (100 - percentile) / 2.0
    minval, maxval = np.percentile(points, [diff, 100 - diff])
    return (points < minval) | (points > maxval)

def clip_outlier(data):
    idx = is_outlier_percentile(data)
    return data[~idx]


def subplots_rows_cols(n):
    ''' get number of subplot rows and columns needed to plot n histograms in one figure '''
    return int(np.round(np.sqrt(n))), int(np.ceil(np.sqrt(n)))


def plot_multihist(data, bins=100, suptitle='histograms', titles=[], clip_outlier=False, plot_name='histograms', fig_dir=None, fig_format='.pdf'):
    ''' plot len(data) histograms on same figure 
        data = list of features to plot (each element is flattened before plotting)
    '''
    rows_n, cols_n = subplots_rows_cols(len(data))
    fig, axs = plt.subplots(nrows=rows_n,ncols=cols_n, figsize=(9,9))
    for ax, dat, title in zip(axs.flat, data, titles):
        if clip_outlier:
            dat = clip_outlier(dat.flatten())
        plot_hist_on_axis(ax, dat.flatten(), bins=bins, title=title)
    [a.axis('off') for a in axs.flat[len(data):]] # turn off unused subplots
    plt.suptitle(suptitle)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    if fig_dir is not None:
        fig.savefig(os.path.join(fig_dir, plot_name + fig_format))
    else:
        plt.show();
    plt.close(fig)


def plot_hist_on_axis(ax, data, bins, xlabel='', ylabel='', title='histogram', legend=[], ylogscale=True, density=True, ylim=None, xlim=None):
    if ylogscale:
        ax.set_yscale('log', nonpositive='clip')
    counts, edges, _ = ax.hist(data, bins=bins, density=density, histtype='step', label=legend)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    return counts, edges

def get_label_and_score_arrays(neg_class_losses, pos_class_losses):
    labels = []
    losses = []

    for neg_loss, pos_loss in zip(neg_class_losses, pos_class_losses):
        labels.append(np.concatenate([np.zeros(len(neg_loss)), np.ones(len(pos_loss))]))
        losses.append(np.concatenate([neg_loss, pos_loss]))

    return [labels, losses]


def plot_roc(neg_class_losses, pos_class_losses, legend=[], title='ROC', legend_loc='best', plot_name='ROC', fig_dir=None, xlim=None, log_x=True):

    # styles
    plt.style.use(hep.style.CMS)
    palette = ['#3E96A1', '#EC4E20', '#FF9505', '#713E5A']

    class_labels, losses = get_label_and_score_arrays(neg_class_losses, pos_class_losses) # neg_class_loss array same for all pos_class_losses

    aucs = []
    fig = plt.figure(figsize=(7, 7))

    for y_true, loss, label, color in zip(class_labels, losses, legend, palette):
        fpr, tpr, threshold = skl.roc_curve(y_true, loss)
        aucs.append(skl.roc_auc_score(y_true, loss))
        if log_x:
            plt.loglog(tpr, 1./fpr, label=label + " (auc " + "{0:.3f}".format(aucs[-1]) + ")", color=color)
        else:
            plt.semilogy(tpr, 1./fpr, label=label + " (auc " + "{0:.3f}".format(aucs[-1]) + ")", color=color)

    dummy_res_lines = [Line2D([0,1],[0,1],linestyle='-', color=c) for c in palette[:2]]
    if log_x:
        plt.loglog(np.linspace(0, 1, num=100), 1./np.linspace(0, 1, num=100), linewidth=1.2, linestyle='solid', color='silver')
    else:
        plt.semilogy(np.linspace(0, 1, num=100), 1./np.linspace(0, 1, num=100), linewidth=1.2, linestyle='solid', color='silver')

    plt.grid()
    if xlim:
        plt.xlim(left=xlim)
    plt.xlabel('True positive rate')
    plt.ylabel('1 / False positive rate')
    plt.legend(loc=legend_loc)
    plt.tight_layout()
    plt.title(title)
    if fig_dir:
        fig.savefig(os.path.join(fig_dir, plot_name + '.png'), bbox_inches='tight' )
   # plt.close(fig)
    plt.show()

    return aucs



