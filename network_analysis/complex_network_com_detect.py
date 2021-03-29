import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# network tools
import networkx as nx
import igraph as ig
from cdlib import algorithms
from cdlib import viz
from cdlib import NodeClustering
from cdlib import evaluation

# metrics
from sklearn import metrics

#%% load ground truth community data and the network data
# graph_name = "Euro University Emails"
# df_truth = pd.read_csv('email/email-Eu-core-department-labels.txt', sep=' ', header=None, dtype=str)
# df_truth.columns = ['node', 'truth']
#
# df_edges = pd.read_csv('email/email-Eu-core.txt', sep=' ', header=None, dtype=str)
# df_edges.columns = ['Source', 'Target']


#%% define graph name, graph, and ground truth communities
graph_name = 'AIS Data'
df_truth = pd.read_csv('sites_with_regions.csv')
df_truth = df_truth[['port_id', 'region']]
df_truth.columns = ['node_id', 'truth']

df_edges = pd.read_csv('ais_edgelist_for_cd.csv')



#%%

# need to read the edges df as a list of values into networkx
nx_graph_weighted = nx.from_pandas_edgelist(df_edges, source='Source_id', target='Target_id', edge_attr='weight')
nx_graph = nx.from_pandas_edgelist(df_edges, source='Source_id', target='Target_id')

# most algos need the largest connected component (lcc) to find communities, so lets do that next.
# use networkx to build the graph, find the nodes of the lcc, and build the subgraph of interest
lcc = max(nx.connected_components(nx_graph), key=len)
nx_g = nx_graph.subgraph(lcc)
# next reduce our ground truth df down to just those nodes in the lcc.
df_lcc_truth = df_truth[df_truth['node_id'].isin(list(lcc))]

# many of the formulas need an igraph object with sequential vertices.
ig_g = ig.Graph.TupleList(nx_g.edges(), directed=True)
# now map the igraph vertex id to the original id
dict_node_id = {}
for i in range(len(ig_g.vs)):
    node_id = ig_g.vs[i]['name']
    dict_node_id[node_id] = i
# and map it back to our truth dataframe
df_lcc_truth['node'] = df_lcc_truth['node_id'].map(dict_node_id)

# finally delete the name vertex attributes so they arent given later
del(ig_g.vs['name'])
#%%
# need to make a nodeclustering object for ground truth, which takes a set of frozensets for communities
# find all the unique communities and define an empty set to catch a frozenset of all nodes in each community
truth_coms = df_lcc_truth['truth'].unique()
communities_set = set()
# iterate through the unique communities to get the nodes in each community and add them to the set as frozensets
for com in truth_coms:
    com_nodes = df_lcc_truth[df_lcc_truth['truth'] == com].iloc[:, 0].values
    communities_set.add(frozenset(com_nodes))
# build the nodeclustering object
ground_truth_com = NodeClustering(communities=communities_set, graph=nx_g, method_name="ground_truth")

#%% define and plot ground truth
# define the positions here so all the cluster plots in the loop are the same structure
pos = nx.fruchterman_reingold_layout(nx_g)
#%%
# plot the original network with the ground truth communities
viz.plot_network_clusters(nx_g, ground_truth_com, pos, figsize=(5, 5))
#nx.draw_networkx_labels(nx_g, pos=pos)
plt.title(f'Ground Truth of {graph_name}')
plt.show()

#%% evaluate ground communitiy metrics
viz.plot_com_properties_relation(ground_truth_com, evaluation.size,
                                 evaluation.internal_edge_density)
plt.show()

#%%
viz.plot_com_stat(ground_truth_com, evaluation.conductance)
plt.show()
viz.plot_com_stat(ground_truth_com, evaluation.average_internal_degree)
plt.show()


algo_dict = {'louvain': algorithms.louvain,
             'leidan': algorithms.leiden,
             'greed_modularity': algorithms.greedy_modularity,
             'label_prop': algorithms.label_propagation,
             'walktrap': algorithms.walktrap,
             'infomap': algorithms.infomap,
             'eigenvector': algorithms.eigenvector,
             'spinglass': algorithms.spinglass}
# %% define algorithms and their names to iterate through
algo_dict = {'louvain': algorithms.louvain,
             'greed_modularity': algorithms.greedy_modularity,
             'label_prop': algorithms.label_propagation,
             'infomap': algorithms.infomap}

# iterate through all the algorithms
# set variables for the iteration loop
# make a dict to store results about each algo
results_dict = dict()
# # make a df with all the nodes.  will capture each model's clustering
# df_nodes = pd.DataFrame(list(nx_g.nodes))
# df_nodes.columns = ['node']
# df_nodes = pd.merge(df_nodes, df_lcc_truth, how='left', left_on='node', right_on='node')
df_nodes = df_lcc_truth.copy()

#%%
algo_dict = {'girvan_newman': algorithms.girvan_newman}
#%%
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
    df_clusts.columns = ['node_id', name]
    # merge this results with the df_nodes to keep track of all the nodes' clusters
    df_nodes = pd.merge(df_nodes, df_clusts, how='left', left_on='node_id', right_on='node_id')
    # We can then just adjusted mutual info to find similarity score
    ami_score = metrics.adjusted_mutual_info_score(df_nodes[name], df_nodes['truth'])
    print(f'The AMI for {name} algorithm is {round(ami_score, 3)}, and there were {len(communities)} communities.')
    results_dict[name] = {'AMI' : round(ami_score, 3),
                          'pred_coms' : pred_coms,
                          'numb_communities' : len(communities),
                          'truth_communities' : len(df_nodes['truth'].unique())}

    # plot the network clusters
    viz.plot_network_clusters(nx_g, pred_coms, pos, figsize=(5, 5))
    plt.title(f'{name} algo of {graph_name}, AMI = {round(ami_score, 3)}')
    plt.show()

    # plot the graph
    viz.plot_community_graph(nx_g, pred_coms, figsize=(5, 5))
    plt.title(f'Communities for {name} algo of {graph_name}.')
    plt.show()

#%% analysis plots
coms = [ground_truth_com]
for name, results in results_dict.items():
    coms.append(results['pred_coms'])
#%%
viz.plot_sim_matrix(coms,evaluation.adjusted_mutual_information)
plt.show()

viz.plot_com_properties_relation(coms, evaluation.size, evaluation.internal_edge_density)
plt.title('Internal Edge Density vs. Size')
plt.show()

viz.plot_com_properties_relation(coms, evaluation.size, evaluation.average_internal_degree)
plt.title('Internal Average Degree vs. Size')
plt.show()

viz.plot_com_stat(coms, evaluation.internal_edge_density)
plt.show()

#%%
df_nodes.to_csv('ais/nodes_from_cd.csv', index=False)
#%%

df_report = pd.DataFrame([list(nx_g),
                          [len(nx_g[node]) for node in nx_g.nodes()],
                          list(nx.degree_centrality(nx_g).values()),
                          nx.eigenvector_centrality(nx_g).values(),
                          nx.closeness_centrality(nx_g).values(),
                          nx.betweenness_centrality(nx_g).values()]
                         ).T
df_report.columns = ['Node', 'Targets', 'Degree',
                     'Eigenvector', 'Centrality', 'Betweenness']
df_report = (df_report.astype({'Degree':'float',
                               'Eigenvector':'float', 'Centrality':'float', 'Targets': 'int',
                               'Betweenness':'float'}).round(3))
df_report.hist(['Degree',  'Eigenvector', 'Centrality', 'Betweenness'])
plt.show()

df_report.boxplot(['Degree',  'Eigenvector', 'Centrality', 'Betweenness'])
plt.show()

print(df_report.sort_values('Betweenness'))

#%%

df_nodes.groupby('truth').count()['node'].sort_values(ascending=False).plot(kind='bar')
plt.title('Counts per Community')
plt.show()




#%% define functions
def return_principle_comp(graph):
    ## graph needs to be an ig graph
    comps = graph.decompose()
    pc_graph = comps[0]
    print(f'Original graph has {len(graph.vs())} nodes and {len(graph.es())} edges.'
          f'There are {len(comps)} components.  \n'
          f'The principle components has {round(len(pc_graph.vs())/len(graph.vs()),3)} of nodes and {round(len(pc_graph.es())/len(graph.es()),3)} of edges. ')
    return pc_graph



# Use the igraph version since it reindexes node numbers and is stricter than networkx
ig_g = return_principle_comp(graph)
# convert the pc back to a nx graph for plotting
# this version keeps the nx nodes as the same from original igraph
nx_g = nx.from_edgelist([(names[x[0]], names[x[1]])
                      for names in [ig_g.vs['name']] # simply a let
                      for x in ig_g.get_edgelist()], nx.Graph())
#%%
# this version renumbers the node ids starting at 0.
#ig_lcc = cdlib.utils.convert_graph_formats(nx_g, ig.Graph)

ig_lcc = ig.Graph()
ig_lcc.add_vertices(list(nx_g.nodes()))
for edge in nx_g.edges():
    ig_lcc.add_edge(ig_lcc.vs['name'].index(edge[0]), ig_lcc.vs['name'].index(edge[1]))

