import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as spc
from matplotlib import pyplot as plt

def correlation_dendrogram(corr, min_corr, cols):
    """
    plot dendrogram based on correlation and select groups based on min_cor
    
    corr: float, narray
    min_cor: float within 0 to 1
    cols: list of labels
    """
    max_dist = 1-min_corr
    pdist = spc.distance.squareform(1 - np.abs(corr))
    pdist = np.clip(pdist, 0, 1)
    linkage = spc.linkage(pdist, method='complete')
    
    plt.rcParams['figure.figsize'] = [25, 40]
    spc.dendrogram(
        linkage,
        leaf_font_size=10,
        orientation='right',
        labels=cols
    )
    plt.axvline(x=max_dist, color='k', linestyle='--')
    plt.show()
    
    cluster = spc.fcluster(linkage, max_dist, criterion='distance')
    print(f'number of clusters: {np.max(cluster)} at minimum correlation: {min_corr}')
    
    num_group = pd.DataFrame({'label': cols, 'cluster': cluster})
    
    return num_group

def feature_selection(df, max_feat):
    """
    select high score feature clusters
    
    df: dataframe of feature clusters and scores
    max_feat: int, max feature selected
    """
    group_score = (
        df.groupby('cluster')
        .agg(
            avg_score = ('feat_ip', 'mean'),
            feat_cnt = ('label', 'count')
        )
        .sort_values('avg_score', ascending=False)
    )
    feat_cumsum = group_score.feat_cnt.cumsum()
    clusters = list(feat_cumsum[feat_cumsum<=max_feat].index)
    labels = df[df.cluster.isin(clusters)].label.values
    
    return labels, len(labels)