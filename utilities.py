# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:48:57 2021

@author: Gian Maria
"""

import pandas as pd, numpy as np, os, igraph as ig, leidenalg as la
import cvxpy as cp
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from kneed import KneeLocator
from sklearn.utils.validation import check_symmetric
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt
from sklearn.neighbors import kneighbors_graph
from Bipartite_Ensembling import BGPA


def read_in_data(directory_names, years):
    data = {}
    for year in years:
        data_modes=[]
        for directory in directory_names:
            for filename in os.listdir(os.path.join('C:\\Users\\Gian Maria\\Desktop\\Unitn\\Iain\\CORRECT_DATA\\Data', directory)):
                if year in filename:
                    datum = pd.read_csv(os.path.join('C:\\Users\\Gian Maria\\Desktop\\Unitn\\Iain\\CORRECT_DATA\\Data',directory, filename), index_col=0)
                    datum.fillna(value=0, inplace=True)
                    data_modes.append(datum)
                
        data_modes_index = np.unique(np.concatenate([mode.index for mode in data_modes]))
        data_modes = [mode.reindex(data_modes_index) for mode in data_modes]
        data_modes = [mode.fillna(value=0) for mode in data_modes]
        data[year] = data_modes.copy()
    return data


class Leiden_Unimodal:
    
    def __init__(self, obj_type='RB_Mod', resolution=1.0, n_iterations =-1):
        obj_types = {'CPM': la.CPMVertexPartition, 
                     'RBER': la.RBERVertexPartition, 
                     'RB_Mod': la.RBConfigurationVertexPartition,
                     'Mod': la.ModularityVertexPartition,
                     'Surprise': la.SurpriseVertexPartition
                         }
        self.obj_type = obj_type
        self.obj_func = obj_types[obj_type]
        self.resolution = resolution
        self.n_iterations = n_iterations
        
        
    def fit_transform(self, graph):
        if type(graph) is ig.Graph:
            G =graph
        else:
            G = self._scipy_to_igraph(graph)
            
        if self.obj_type in ['CPM', 'RBER', 'RB_Mod']:
            partition = la.find_partition(G, self.obj_func, n_iterations=self.n_iterations,
                                            resolution_parameter=self.resolution)
        else:
            partition = la.find_partition(G, self.obj_func, n_iterations=self.iterations)
        
        self.modularity_ = partition.quality()
        self.labels_ = np.array(partition.membership)
        return self.labels_
        
        
    def _scipy_to_igraph(self, matrix):
        # matrix.eliminate_zeros()
        sources, targets = matrix.nonzero()
        weights = matrix[sources, targets]
        graph = ig.Graph(n=matrix.shape[0], edges=list(zip(sources, targets)), directed=True, edge_attrs={'weight': weights})
        
        try:
            check_symmetric(matrix, raise_exception=True)
            graph = graph.as_undirected()
        except ValueError:
            pass
        
        return graph
    
    
class Leiden_Multiplex:
    
    def __init__(self, obj_types=None, resolutions=None, modal_weights=None, n_iterations=-1):
        
        self.obj_types = obj_types
        self.resolutions = resolutions
        self.modal_weights = modal_weights
        self.n_iterations = n_iterations
        
    def fit_transform(self, graphs):
        obj_table = {'CPM': la.CPMVertexPartition, 
              'RBER': la.RBERVertexPartition, 
              'RB_Mod': la.RBConfigurationVertexPartition,
              'Mod': la.ModularityVertexPartition,
              'Surprise': la.SurpriseVertexPartition
                  }
        
        G=[]
        for graph in graphs:
            if type(graph) is ig.Graph:
                G.append(graph)
            else:
                G.append(self._scipy_to_igraph(graph))
        
        optimiser = la.Optimiser()
        partitions = []
        for i in range(len(G)):
            if self.obj_types is None:
                partitions.append(la.RBConfigurationVertexPartition(G[i], resolution_parameter=1.0))
            elif self.resolutions is None:
                obj = obj_table[self.obj_types[i]]
                partitions.append(obj(G[i]))
            else:
                obj = obj_table[self.obj_types[i]]
                partitions.append(obj(G[i], resolution_parameter=self.resolutions[i]))
       
        if self.modal_weights is None:
            diff = optimiser.optimise_partition_multiplex(partitions, n_iterations=self.n_iterations)
        else:
            diff = optimiser.optimise_partition_multiplex(partitions, layer_weights = self.modal_weights, n_iterations=self.n_iterations)
            
        self.modularities = [part.modularity for part in partitions]
        self.labels_ = np.array(partitions[0].membership)
        return self.labels_
        
        
    def _scipy_to_igraph(self, matrix):
        matrix.eliminate_zeros()
        sources, targets = matrix.nonzero()
        weights = matrix[sources, targets]
        graph = ig.Graph(n=matrix.shape[0], edges=list(zip(sources, targets)), directed=True, edge_attrs={'weight': weights})
        
        try:
            check_symmetric(matrix, raise_exception=True)
            graph = graph.as_undirected()
        except ValueError:
            pass
        
        return graph
    
    
class MVMC:
    
    def __init__(self, n_iterations=-1, max_clusterings=20, 
                 resolution_tol=1e-2, weight_tol=1e-2, verbose=False):
        
        self.n_iterations = n_iterations
        self.max_clusterings = max_clusterings
        self.resolution_tol = resolution_tol
        self.weight_tol = weight_tol
        self.verbose = verbose
        
    def fit_transform(self, graphs):
        G=[]
        for graph in graphs:
            if type(graph) is ig.Graph:
                G.append(graph)
            else:
                G.append(self._scipy_to_igraph(graph))
                
        if self.verbose:
            for i in range(len(G)):
                print("View Graph {}: num_nodes: {}, num_edges: {}, directed: {}, num_components: {}, num_isolates: {}"
                      .format(i, G[i].vcount(), G[i].ecount(), G[i].is_directed(), 
                              len(G[i].components(mode='WEAK').sizes()), G[i].components(mode='WEAK').sizes().count(1)))
        
        self.weights = []
        self.resolutions =[]
        self.best_modularity =-np.inf
        self.best_clustering = None
        self.best_resolutions = None
        self.best_weights = None
        self.modularities =[]
        self.clusterings =[]
        self.final_iteration = 0
        self.best_iteration = 0
        
        weights = [1]*len(G)
        resolutions =[1]*len(G)
        
        for iterate in range(self.max_clusterings):
            partitions = []
            for i in range(len(G)):
                partitions.append(la.RBConfigurationVertexPartition(G[i], resolution_parameter=resolutions[i]))
                
            optimiser = la.Optimiser()
            diff = optimiser.optimise_partition_multiplex(partitions, layer_weights = weights, n_iterations=self.n_iterations)
            self.clusterings.append(np.array(partitions[0].membership))
            self.modularities.append([part.quality()/(part.graph.ecount() if part.graph.is_directed() else 2*part.graph.ecount()) 
                                      for part in partitions])
            self.weights.append(weights.copy())
            self.resolutions.append(resolutions.copy())
            self.final_iteration +=1
            
            
            if self.verbose:
                print("--------")
                print("Iteration: {} \n Modularities: {} \n Resolutions: {} \n Weights: {}"
                      .format(self.final_iteration, self.modularities[-1], resolutions, weights))
            
            # if np.sum(np.array(self.weights[-1]) * np.array(self.modularities[-1])) > self.best_modularity:
            self.best_clustering = self.clusterings[-1]
            self.best_modularity = np.sum(np.array(self.weights[-1]) * np.array(self.modularities[-1]))
            self.best_resolutions = self.resolutions[-1]
            self.best_weights = self.weights[-1]
            self.best_iteration = self.final_iteration
                
            theta_in, theta_out = self._calculate_edge_probabilities(G)
            for i in range(len(G)):
                resolutions[i] = (theta_in[i] - theta_out[i])/ (np.log(theta_in[i]) - np.log(theta_out[i]))
                weights[i] = (np.log(theta_in[i]) - np.log(theta_out[i]))/(np.mean([np.log(theta_in[j]) - np.log(theta_out[j]) for j in range(len(G))]))

                
            if (np.all(np.abs(np.array(self.resolutions[-1])-np.array(resolutions)) <= self.resolution_tol)
                and np.all(np.abs(np.array(self.weights[-1])-np.array(weights)) <= self.resolution_tol)):
                break
        else:
            best_iteration = np.argmax([np.sum(np.array(self.weights[i]) * np.array(self.modularities[i]))
                                        for i in range(len(self.modularities))])
            self.best_clustering = self.clusterings[best_iteration]
            self.best_modularity = np.sum(np.array(self.weights[best_iteration]) * np.array(self.modularities[best_iteration]))
            self.best_resolutions = self.resolutions[best_iteration]
            self.best_weights = self.weights[best_iteration]
            self.best_iteration = best_iteration
            
            if self.verbose:
                print("MVMC did not converge, best result found: Iteration: {}, Modularity: {}, Resolutions: {}, Weights: {}"
                      .format(self.best_iteration, self.best_modularity, self.best_resolutions, self.best_weights))


        return self.best_clustering
        
    
    def _scipy_to_igraph(self, matrix):
        matrix.eliminate_zeros()
        sources, targets = matrix.nonzero()
        weights = list(matrix.data)
        graph = ig.Graph(n=matrix.shape[0], edges=list(zip(sources, targets)), directed=True, edge_attrs={'weight': weights})
        
        try:
            check_symmetric(matrix, raise_exception=True)
            graph = graph.as_undirected()
        except ValueError:
            pass
        
        if not graph.is_weighted():
            graph.es['weight'] = [1.0] * graph.ecount()
        
        return graph
    
    
    def _calculate_edge_probabilities(self, G):
        theta_in =[]
        theta_out =[]
        clusters = self.clusterings[-1].copy()
        for i in range(len(G)):
            m_in = 0
            m = sum(e['weight'] for e in G[i].es)
            kappa =[]
            G[i].vs['clusters'] = clusters
            for cluster in np.unique(clusters):
                nodes = G[i].vs.select(clusters_eq=cluster)
                m_in += sum(e['weight'] for e in G[i].subgraph(nodes).es)
                if G[i].is_directed():
                    degree_products = np.outer(np.array(G[i].strength(nodes, mode = 'IN', weights='weight')), 
                                               np.array(G[i].strength(nodes, mode = 'OUT', weights='weight')))
                    np.fill_diagonal(degree_products,0)
                    kappa.append(np.sum(degree_products, dtype=np.int64))
                else:
                    kappa.append(np.sum(np.array(G[i].strength(nodes, weights='weight')), dtype=np.int64)**2)
            
            if G[i].is_directed():
                if m_in <=0:
                    # Case when there are no internal edges; every node in its own  cluster
                    theta_in.append(1/G[i].ecount())
                else:
                    theta_in.append((m_in)/(np.sum(kappa, dtype=np.int64)/(2*m)))
                if m-m_in <=0:
                    # Case when all edges are internal; 1 cluster or a bunch of disconnected clusters
                    theta_out.append(1/G[i].ecount())
                else:
                    theta_out.append((m-m_in)/(m-np.sum(kappa, dtype=np.int64)/(2*m)))
            else:
                if m_in <=0:
                    # Case when there are no internal edges; every node in its own  cluster
                    theta_in.append(1/G[i].ecount())
                else:
                    theta_in.append((m_in)/(np.sum(kappa, dtype=np.int64)/(4*m)))
                if m-m_in <=0:
                    # Case when all edges are internal; 1 cluster or a bunch of disconnected clusters
                    theta_out.append(1/G[i].ecount())
                else:
                    theta_out.append((m-m_in)/(m-np.sum(kappa, dtype=np.int64)/(4*m)))

        return theta_in, theta_out
    


def create_neighbors_plot(list_of_dfs, metric='cosine'):
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=.5, wspace=.2)
    i = 1
    for df in list_of_dfs:
        X = df.iloc[:,2:].values
        n_neighbors = int(np.ceil(np.log2(X.shape[0])))
        nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, n_jobs=-1)
        neighbors = nearest_neighbors.fit(X)
        distances, indices = neighbors.kneighbors(X)
        distances = np.sort(distances[:,n_neighbors-1], axis=0)
        
        d = np.arange(len(distances))
        knee = KneeLocator(d, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
        
        #ax = fig.add_subplot(2, 5, i)
        #ax.plot(distances)
        print("knee value: {}".format(distances[knee.knee]))
        knee.plot_knee()
        #ax.set_xlabel("Points")
        #ax.set_ylabel("Distance")
        
        i +=1
        
        

def create_nearest_neighbors_graph(list_of_dfs, metric='cosine'):
    graphs = []
    for df in list_of_dfs:
        X = df.values
        '''Row normalize the data'''
        #X = normalize(X, axis=1, norm='l1')
        
        '''Create a k-nearest neighbors graph'''
        n_neighbors = int(np.ceil(np.log2(X.shape[0])))
        graph = kneighbors_graph(X, n_neighbors=n_neighbors, metric=metric, 
                                       mode='distance')
        
        '''converting to similarity and limit to only edges where there is overlap in the
        feature space'''
        graph.data = 1-graph.data
        graph.eliminate_zeros()
        graph.data = (graph.data - np.min(graph.data)) / (np.max(graph.data) - np.min(graph.data))
        
        '''symmetrizing the graphs'''
        #graph = 0.5 * (graph + graph.T)
        #graph = graph.minimum(graph.T)
        #graph = graph.maximum(graph.T)
        #graph.eliminate_zeros()
        
        graphs.append(graph)
    return graphs


def create_radius_ball_graph(list_of_dfs, metric='euclidean'):
    graphs = []
    for df in list_of_dfs:
        #X = df.iloc[:,2:].values
        X = df.values
        '''Row normalize the data'''
        #X = normalize(X, axis=1, norm='l1')
        
        '''Create radius ball graph'''
        n_neighbors = int(np.ceil(np.log2(X.shape[0])))
        nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, n_jobs=-1)
        neighbors = nearest_neighbors.fit(X)
        distances, indices = neighbors.kneighbors(X)
        distances = np.sort(distances[:,n_neighbors-1], axis=0)
        d = np.arange(len(distances))
        knee = KneeLocator(d, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
        graph = radius_neighbors_graph(X, radius = distances[knee.knee],  metric=metric, mode='distance')
        
        '''converting to similarity and limit to only edges where there is overlap in the
        feature space'''
        graph.data = np.around(np.exp(-0.5 * graph.data / np.std(graph.data)), decimals=4)
        #graph.data = 1-graph.data
        #graph.eliminate_zeros()
        #graph.data = (graph.data - np.min(graph.data)) / (np.max(graph.data) - np.min(graph.data))
        
        '''symmetrizing the graphs'''
        #graph = 0.5 * (graph + graph.T)
        #graph = graph.minimum(graph.T)
        #graph = graph.maximum(graph.T)
        #graph.eliminate_zeros()
        
        graphs.append(graph)
    return graphs

def create_lrr_sparse_graph(list_of_dfs):
    graphs = []
    for df in list_of_dfs:
        X = df.values
        n = X.shape[0]
        m = X.shape[1]
        W = cp.Variable(shape=(n,n))
        obj = cp.Minimize(cp.norm(cp.norm(W@X - X, p=2, axis=0), p=1)+ 100*cp.norm(W, p=1))
        constraint = [cp.diag(W)==0]
        prob = cp.Problem(obj, constraint)
        optimal_value = prob.solve()
        graph = np.round((np.abs(W.value) + np.transpose(np.abs(W.value)))/2, 2)
        
        graphs.append(csr_matrix(graph))
    return graphs


def pd_fill_diagonal(df_matrix, value=0): 
    mat = df_matrix.values
    n = mat.shape[0]
    mat[range(n), range(n)] = value
    return pd.DataFrame(mat)

def projected_graph(list_of_dfs):
    proj_graphs = []
    for df in list_of_dfs:
        df = df.dot(df.T)
        pd_fill_diagonal(df, value=0)
        df.fillna(0, inplace=True)
        df=df.div(df.sum(axis=1),axis=0)
        df = 0.5*(df+df.T)
        graph = csr_matrix(df.values)
        graph.data[np.isnan(graph.data)] = 0.0
        proj_graphs.append(graph)
    return proj_graphs


def scipy_to_igraph(matrix):
    matrix.eliminate_zeros()
    sources, targets = matrix.nonzero()
    weights = list(matrix.data)
    graph = ig.Graph(n=matrix.shape[0], edges=list(zip(sources, targets)), directed=True, edge_attrs={'weight': weights})

    try:
        check_symmetric(matrix, raise_exception=True)
        graph = graph.as_undirected()
    except ValueError:
        pass

    if not graph.is_weighted():
        graph.es['weight'] = [1.0] * graph.ecount()

    return graph


def get_graph_stats(yearly_networks, view_names):
    for year in yearly_networks.keys():
        for i in range(len(view_names)):
            datum ={}
            datum['Key'] = year+"_"+view_names[i]
            datum['Year'] = year
            datum['View'] = view_names[i]
            G = scipy_to_igraph(yearly_networks[year][i])
            datum['Num_Nodes'] = G.vcount()
            datum['Num_Edges'] = G.ecount()
            datum['Density'] = G.density()
            datum['Num_Components'] = len(G.components(mode='WEAK').sizes())
            datum['Num_Isolates'] = G.components(mode='WEAK').sizes().count(1)
            datum['Clustering_Coefficient'] = G.transitivity_undirected(mode="zero")
            datum['Average Path Length'] = G.average_path_length(directed=False)
            datum['Avg Neighbors'] = G.knn(vids=None)
            datum['Assortativity'] = G.assortativity_degree(directed = False)
            yield datum
            

def find_multi_view_clusters(names, graphs, view_names, num_clusterings=10):
    ensembler = BGPA()
    modularities = []
    resolutions =[]
    weights = []
    iterations = []
    clusterings = []
    mvmc_clstr= MVMC(resolution_tol=0.01, weight_tol=0.01, max_clusterings=40)
    for _ in range(num_clusterings):
        community_labels = mvmc_clstr.fit_transform(graphs)
        clusterings.append(community_labels)
        modularities.append(mvmc_clstr.modularities[-1])
        resolutions.append(mvmc_clstr.resolutions[-1])
        weights.append(mvmc_clstr.weights[-1])
        iterations.append(mvmc_clstr.final_iteration)

    performance_results ={}
    performance_results['view_names'] = view_names
    performance_results['modularity'] = np.average(np.array(modularities))
    performance_results['resolution'] = np.average(np.array(resolutions), axis=0)
    performance_results['weights'] = np.average(np.array(weights), axis=0)
    performance_results['iterations'] = np.average(np.array(iterations))
    return pd.DataFrame(index = names, data = ensembler.fit_predict(clusterings)), pd.DataFrame(performance_results)


def convert_partial_labels_to_df(labels):
    '''Used for when the clusterings may not fully overlap on the objects being
    clustered. Occurs when there are partial views of the data. Takes a list 
    of pandas Series that have object names for keys and labels for values, and 
    outputs a dataframe where all objects have cluster labels in all clusterings,
    which can then be input to cluster ensembling.'''
    
    master_node_names = np.unique(np.concatenate([label.index for label in labels]))
    
    master_df = pd.DataFrame(index=master_node_names)
    
    for i in range(len(labels)):
        master_df[i] = labels[i]
    
    master_df = master_df.fillna(999) #Fill a dummy label
    
    master_df = master_df.astype('int32')
    
    return master_df