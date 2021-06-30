# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:03:02 2019

@author: icruicks
"""
import sknetwork as skn
import numpy as np, pandas as pd
from sklearn.cluster import SpectralCoclustering
from sknetwork.clustering import BiLouvain
from sklearn.base import ClusterMixin


class BGPA(ClusterMixin):
    """Bipartite Graph Partitioning Algorithm
    
    BGPA+ : treat the objects-by-bse clsuterings
    as a bipartite graph partitioning problem
    
    Parameters
    ----------
    base_clusters : list of array_like or pandas dataframe
        List of cluster labels where each set of labels must be in the same order
        and have shape (n_objects,). Note: standard output from sci-kit learn 
        clustering algorithms will return this format. If pandas dataframe, 
        should be shape (n_objects, n_clusters)
    metaclustering_alg : string
        string to denote how to cluster the object-to-object cluster association
        graph. Options are: 'louvain','spectral','METIS'. Default is 'louvain'
    n_clusters : (int,int)
        Number of clusters to find to find for the objects and the base clusters. 
        Used only for spectral. Default is None
    
    Returns
    -------
    z : ndarray
        cluster labels of shape (n,).
    """
    
    
    def __init__(self, metaclustering_alg = 'louvain', n_clusters = (None,None),
                 refine_clusters = False, ensemble_iterations=10):
        
        self.metaclustering_alg = metaclustering_alg
        self.n_clusters = n_clusters
        self.refine_clusters = refine_clusters
        self.ensemble_iterations = ensemble_iterations
    
    def fit_predict(self, base_clusters):
        if self.refine_clusters:
            #note: using a refinement process in BGPA is still in development
            #for now, its best to not use it.
            converged = False
            ba_matrix = self._create_ba_matrix(base_clusters)
            while not converged:
                final_clusterer = meta_alg()
                clusters = [final_clusterer.cluster(ba_matrix, self.n_clusters, self.metaclustering_alg)
                            for _ in range(self.ensemble_iterations)]
                ba_matrix = self._create_ba_matrix(clusters)
                C = ba_matrix @ ba_matrix.T
                if np.all(C[np.nonzero(C)] >= len(base_clusters)):
                    converged = True
            return clusters
            
        else:
            ba_matrix = self._create_ba_matrix(base_clusters)
            final_clusterer = meta_alg()
            self.obj_clusters, self.cluster_clusters = final_clusterer.cluster(ba_matrix, self.n_clusters, self.metaclustering_alg)
            return self.obj_clusters
        
    def _create_ba_matrix(self, clusters):
        '''
        Internal function for converting list of clustering labels into a
        binary-association matrix that is (object,clusters)
        '''
        
        if isinstance(clusters, pd.DataFrame):
            clusters = [clusters.loc[:,i].to_numpy(dtype=int) for i in clusters.columns]
        else:
            clusters = [clustering.astype(np.int64) for clustering in clusters]
        
        ba_matrices =[]
        for base_cluster in clusters:
            ba_matrix = np.zeros((base_cluster.size, base_cluster.max()+1))
            ba_matrix[np.arange(base_cluster.size), base_cluster] = 1
            ba_matrices.append(ba_matrix)
            
        return np.concatenate(ba_matrices, axis=1)
        
    
    
class LWBG(ClusterMixin):
    """Locally Weighted Bipartite Graph Partitioning Algorithm 
    
    LWBG : treat the objects-by-base clsuters as a bipartite graph 
    partitioning problem. In this case the bipartite graph is weighted by 
    information-thoeretic measures
    
    Parameters
    ----------
    base_clusters : list of array_like or pandas dataframe
        List of cluster labels where each set of labels must be in the same order
        and have shape (n_objects,). Note: standard output from sci-kit learn 
        clustering algorithms will return this format. If pandas dataframe, 
        should be shape (n_objects, n_clusters)
    metaclustering_alg : string
        string to denote how to cluster the object-to-object cluster association
        graph. Options are: 'louvain','spectral','METIS'. Default is 'louvain'
    n_clusters : (int,int)
        Number of clusters to find to find for the objects and the base clusters. 
        Used only for spectral. Default is None
    theta : float
        Controls the impact of the local weighting. Default is 0.5
    
    Returns
    -------
    z : ndarray
        cluster labels of shape (n,).
    """
    
    def __init__(self, metaclustering_alg = 'louvain', n_clusters = (None,None),
                 theta = 0.5):
        
        self.metaclustering_alg = metaclustering_alg
        self.n_clusters = n_clusters
        self.theta = theta
    
    def fit_predict(self, base_clusters):
        ba_matrices =[]
        idxs = []
        idxs_iter = 0
        
        if isinstance(base_clusters, pd.DataFrame):
            base_clusters = [base_clusters.loc[:,i].to_numpy(dtype=int) for i in base_clusters.columns]
        else:
            base_clusters = [clustering.astype(np.int64) for clustering in base_clusters]
        
        for base_cluster in base_clusters:
            ba_matrix = np.zeros((base_cluster.size, base_cluster.max()+1))
            ba_matrix[np.arange(base_cluster.size), base_cluster] = 1
            ba_matrices.append(ba_matrix)
            idxs.append([idxs_iter, idxs_iter+ba_matrix.shape[1]])
            idxs_iter += ba_matrix.shape[1]
            
        ba_matrix = np.concatenate(ba_matrices, axis=1)
        
        H_matrix = np.zeros((ba_matrix.shape[1], len(ba_matrices)))
        final_clusterer = meta_alg()       
        
        for col_idx in range(ba_matrix.shape[1]):
            for mode in range(len(idxs)):
                idx = idxs[mode]
                if idx[0] <= col_idx <=idx[1]:
                    pass
                else:
                    H_m = []
                    for alt_idx in range(idx[0], idx[1]):
                        C_i = ba_matrix[:,col_idx]
                        C_j = ba_matrix[:,alt_idx]
                        P_ij = np.dot(C_i, C_j)/ np.count_nonzero(C_i)
                        if P_ij > 0:
                            H_m.append(P_ij * np.log2(P_ij))
                        else:
                            H_m.append(0)
                    H_matrix[col_idx, mode] = -1*np.sum(H_m)
        
        ECI = np.exp(-1* np.sum(H_matrix, axis=1)/(self.theta*len(ba_matrices)))
        weighted_ba_matrix = ba_matrix * ECI
        self.obj_clusters, self.cluster_clusters = final_clusterer.cluster(weighted_ba_matrix, self.n_clusters, self.metaclustering_alg)
        return self.obj_clusters
    
    
    # def _get_weighted_subgroup_counts(self, a, cluster_weights):    
    #     N = a.max()+1
    #     weighted_counts = np.zeros((a.shape[0], N-1))
    #     for label in range(1,N):
    #         label_counts = (a==label) * cluster_weights
    #         normalization = np.count_nonzero(np.sum(a == label, axis=0) >0)
    #         weighted_counts[:,label-1] = np.sum(label_counts, axis=1)/normalization
    #     return weighted_counts
        
    
class meta_alg:
    '''
    Factory class that does the clustering of the bipartite graphs for the other
    method classes
    '''
    
    def cluster(self, graph, n_clusters, format):
        clustering = self._get_clustering(format)
        return clustering(graph, n_clusters)
    
    def _get_clustering(self, format):
        if format == 'louvain':
            return self._graph_louvain
        elif format == 'spectral':
            return self._spectral
        else:
            raise ValueError(format)
            
    def _spectral(self, graph, n_clusters):
        clstr = SpectralCoclustering(n_clusters=n_clusters).fit(graph)
        return clstr.labels_row_, clstr.labels_col_
    
    def _graph_louvain(self, graph, n_clusters):
        clstr = BiLouvain()
        clstr.fit(graph)
        return clstr.labels_row_, clstr.labels_col_
        
        

        
        
        