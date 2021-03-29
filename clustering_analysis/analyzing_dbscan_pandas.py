#%%
import pandas as pd
import datetime

from gnact import utils, clust, network
import db_config


#%%


# create the engine to the database
engine = utils.connect_engine(db_config.colone_cargo_params, print_verbose=True)
#%% Define a rollup function to determine metrics for each iteration of each method.
def rollup_clustering(epsilons_km, min_samples, epsilons_time, method):
    method_dict = dict()
    for eps_km in epsilons_km:
        for min_samp in min_samples:
            for eps_time in epsilons_time:
                iteration_start = datetime.datetime.now()
                # Set the params_name based on the method
                if method in ['dbscan', 'optics']:
                    params_name = f"{method}_{str(eps_km).replace('.', '_')}_{min_samp}"
                elif method in ['hdbscan']:
                    params_name = f"{method}_{min_samp}"
                elif method in ['stdbscan', 'dynamic']:
                    params_name = f"{method}_{str(eps_km).replace('.', '_')}_{min_samp}_{eps_time}"
                else:
                    print("Method must be one of 'dbscan', 'optics', 'hdbscan', stdbscan', or dynamic segmentation.")
                    break
                print(f'Starting processing for {params_name}...')
                first_tick = datetime.datetime.now()
                df_clusts = clust.calc_clusts(df_posits, eps_km=eps_km, min_samp=min_samp,
                                              eps_time=eps_time, method=method)
                results = clust.calc_stats(df_clusts, df_stops, dist_threshold_km)
                lapse = datetime.datetime.now() - first_tick
                results['proc_time'] = lapse.total_seconds()
                method_dict[params_name] = results
                print(results)
    return method_dict

#%%
uid = ('636016432',)
# make the df from the data in the database for MSC Ashrui
df_posits = clust.get_uid_posits(uid, engine, end_time='2018-01-01')
df_sites = clust.get_sites_wpi(engine)

#%%
dist_threshold_km = 5
loiter_time_mins = 360
results_dict = {}

# manually create the missing site near Savannah
savannah_site = {'site_id':3, 'site_name': 'SAVANNAH_MANUAL_1', 'lat': 32.121167, 'lon':-81.130085,
               'region':'East_Coast'}
# add the site to the df_sites
df_sites = df_sites.append(savannah_site, ignore_index=True) # add savannah
# recompute the nearest neighbors
df_nn = clust.calc_nn(df_posits, df_sites)
# determine the "ground truth" for this sample
df_stops = network.calc_static_seg(df_posits, df_nn, df_sites,
                                   dist_threshold_km, loiter_time_mins)

#%%
method = 'dbscan'
# used in dbscan and stdbscan as eps, optics and hdbscan as max eps
epsilons_km = [1,3,5]
# used in all as minimum size of cluster
min_samples = [25,50,100,200,300,500,1000]
# if not stdbscan, leave as None if not using temporal clustering.  values in minutes
epsilons_time = [None]

method_dict = rollup_clustering(epsilons_km, min_samples, epsilons_time, method)
results_dict.update(method_dict)

#%%
method = 'optics'
# used in dbscan and stdbscan as eps, optics and hdbscan as max eps
epsilons_km = [5]
# used in all as minimum size of cluster
min_samples = [25,50,100,200,300,500,1000]
# if not stdbscan, leave as None if not using temporal clustering.  values in minutes
epsilons_time = [None]

method_dict = rollup_clustering(epsilons_km, min_samples, epsilons_time, method)
results_dict.update(method_dict)

#%%
method = 'hdbscan'
# used in dbscan and stdbscan as eps, optics and hdbscan as max eps
epsilons_km = [None]
# used in all as minimum size of cluster
min_samples = [25,50,100,200,300,500,1000]
# if not stdbscan, leave as None if not using temporal clustering.  values in minutes
epsilons_time = [None]

method_dict = rollup_clustering(epsilons_km, min_samples, epsilons_time, method)
results_dict.update(method_dict)
#%%
# method = 'stdbscan'
# # used in dbscan and stdbscan as eps, optics and hdbscan as max eps
# epsilons_km = [1,3,5]
# # used in all as minimum size of cluster
# min_samples = [25,50,100,200,300,500,1000]
# # if not stdbscan, leave as None if not using temporal clustering.  values in minutes
# epsilons_time = [240,360,480,600,720,1080]
#
# method_dict = rollup_clustering(epsilons_km, min_samples, epsilons_time, method)

#
# method = 'stdbscan'
# # used in dbscan and stdbscan as eps, optics and hdbscan as max eps
# epsilons_km = [3]
# # used in all as minimum size of cluster
# min_samples = [50,100,200,300]
# # if not stdbscan, leave as None if not using temporal clustering.  values in minutes
# epsilons_time = [360,600]
#
#
# method_dict = dict()
# for eps_km in epsilons_km:
#     for min_samp in min_samples:
#         for eps_time in epsilons_time:
#             iteration_start = datetime.datetime.now()
#             params_name = f"{method}_{str(eps_km).replace('.', '_')}_{min_samp}_{eps_time}"
#             print(f'Starting processing for {params_name}...')
#             first_tick = datetime.datetime.now()
#
#             df_clusts = pd.read_sql(f"""select c.{params_name} clust_id, pos.lat, pos.lon, pos.time
#             from ship_sample as pos, clustering_results as c
#             where c.id = pos.id
#             and {params_name} is not null""", engine)
#
#             results = clust.calc_stats(df_clusts, df_stops, dist_threshold_km)
#             lapse = datetime.datetime.now() - first_tick
#             results['proc_time'] = lapse.total_seconds()
#             method_dict[params_name] = results
#
#  results_dict.update(method_dict)
#%%
method = 'dynamic'
# used in dbscan and stdbscan as eps, optics and hdbscan as max eps
epsilons_km = [1,3,5]
# used in all as minimum size of cluster
min_samples = [None]
# if not stdbscan, leave as None if not using temporal clustering.  values in minutes
epsilons_time = [240,360,480,600,720,1080]

method_dict = rollup_clustering(epsilons_km, min_samples, epsilons_time, method)
results_dict.update(method_dict)

#%%
df_results = pd.DataFrame(results_dict).T

df_results = df_results.reset_index(drop=False).rename({'index':'method_full'},axis=1)
df_results['method_type'] = df_results.method_full.str.split('_').str.get(0)
df_results['params'] = df_results.method_full.str.split('_').str.slice(start=1).str.join('_')

df_results.to_csv(f'results_{uid[0]}.csv', index=False)
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

g = sns.scatterplot(df_results.recall, df_results.precision, hue=df_results.method_type, size=df_results.proc_time)
g.set_xticks(np.arange(len(df_results)))
g.set_xticklabels(df_results.params, rotation=60)
plt.show()

#%%
g = sns.scatterplot(df_results.recall, df_results.precision, hue=df_results.method_type, size=df_results.proc_time)
plt.title(f'Precision and Recall for UID {uid[0]} Various Clustering Methods \n Colored by Method, Sized by Processing Time')
plt.show()
#%%
from gnact import utils
with utils.connect_psycopg2(db_config.colone_cargo_params, print_verbose=False) as conn:
    with conn.cursor() as c:
        c.execute(f"""SELECT DISTINCT (name) from clustering_rollup;""")
        completed_cols = c.fetchall()


#%%




