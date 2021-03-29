import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# network tools
import networkx as nx
import igraph as ig
import cdlib
from cdlib import algorithms
from cdlib import viz
from cdlib import NodeClustering
from cdlib import evaluation

# metrics
from sklearn import metrics
#%%
df_edgelist = pd.read_csv('full_edgelist.csv')
df_edgelist['Target_id'] = df_edgelist['Target_id'].astype('int')
df_edgelist['uid'] = df_edgelist['mmsi']
df_edgelist.drop(['Unnamed: 0', 'mmsi'], inplace=True, axis=1)

df_edges = df_edgelist[['Source', 'Target']]


#%% define graph name, graph, and ground truth communities
graph_name = "AIS Network"
# need to read the edges df as a list of values into igraph and networkx
ig_graph = ig.Graph.TupleList(df_edges.values)
nx_graph = nx.from_edgelist(df_edges.values)

# most algos need the largest connected component (lcc) to find communities, so lets do that next.
# use networkx to build the graph, find the nodes of the lcc, and build the subgraph of interest
lcc = max(nx.connected_components(nx_graph), key=len)
nx_g = nx_graph.subgraph(lcc)

# define the positions here so all the cluster plots in the loop are the same structure
pos = nx.spring_layout(nx_g)

#%%
plt.figure(figsize=(10, 10))
# nodes
nx.draw_networkx_nodes(nx_g, pos)
# edges
nx.draw_networkx_edges(nx_g, pos)
plt.axis('off')
plt.title('Full Network Plot')
plt.show()

# %% define algorithms and their names to iterate through
algo_dict = {'louvain': algorithms.louvain,
             'leidan': algorithms.leiden,
             'greed_modularity': algorithms.greedy_modularity,
             'label_prop': algorithms.label_propagation,
             'walktrap': algorithms.walktrap,
             'infomap': algorithms.infomap,
             'eigenvector': algorithms.eigenvector,
             'spinglass': algorithms.spinglass}

#%% iterate through all the algorithms
# set variables for the iteration loop
# make a dict to store results about each algo
results_dict = dict()
# make a df with all the nodes.  will capture each model's clustering
df_nodes = pd.DataFrame(list(nx_g.nodes))
df_nodes.columns = ['node']

# iterate through the alogrithms
for name, algo in algo_dict.items():
    # run algos to make node_clustering objects
    pred_coms = algo(nx_g)
    communities = pred_coms.communities

    # need to convert the community groups from list of lists to a dict of lists for ingest to df
    coms_dict = dict()
    for c in range(len(communities)):
        for i in communities[c]:
            coms_dict[i] = [c]

    # make a df with the results of the algo
    df_clusts = pd.DataFrame.from_dict(coms_dict).T.reset_index()
    df_clusts.columns = ['node', name]
    # merge this results with the df_nodes to keep track of all the nodes' clusters
    df_nodes = pd.merge(df_nodes, df_clusts, how='left', left_on='node', right_on='node')

    # plot the network clusters
    viz.plot_network_clusters(nx_g, pred_coms, pos, figsize=(5, 5))
    plt.title(f'{name} algo of {graph_name}')
    plt.show()

    # plot the graph
    viz.plot_community_graph(nx_g, pred_coms, figsize=(5, 5))
    plt.title(f'Communities for {name} algo of {graph_name}.')
    plt.show()

#%%
odd_ports = ['ATLANTIC CITY', 'OCEAN CITY', 'KEY WEST']
df_odd_ports = df_edgelist[(df_edgelist['Source'].isin(odd_ports)) | (df_edgelist['Target'].isin(odd_ports))]

#%% explore communities in communities

pred_coms = algorithms.eigenvector(nx_g)
communities = pred_coms.communities
coms_dict = dict()
for c in range(len(communities)):
    com_list = list()
    for i in communities[c]:
        com_list.append(i)
    coms_dict[c] = com_list

df_com0 = df_edgelist[(df_edgelist['Source'].isin(coms_dict[0])) | (df_edgelist['Target'].isin(coms_dict[0]))]

#%%
nx_com0 = nx.from_edgelist(df_com0[['Source', 'Target']].values)

pred_coms_com0 = algorithms.eigenvector(nx_com0)
communities = pred_coms_com0.communities

for c in range(len(communities)):
    com_list = list()
    for i in communities[c]:
        com_list.append(i)
    coms_dict[c] = com_list

# define the positions here so all the cluster plots in the loop are the same structure
pos = nx.spring_layout(nx_com0)

#%%
plt.figure(figsize=(10, 10))
# nodes
nx.draw_networkx_nodes(nx_com0, pos, label=True)
# edges
nx.draw_networkx_edges(nx_com0, pos)
plt.axis('off')
plt.title('Community 0 from Eigenvector Plot')
plt.show()

nx.draw(nx_com0, pos, with_labels=True)
plt.show()

# plot the network clusters
viz.plot_network_clusters(nx_com0, pred_coms_com0, pos, figsize=(5, 5))
plt.title('Community 0 from Eigenvector Plot')
plt.show()