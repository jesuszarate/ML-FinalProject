import matplotlib.pyplot as plt
import pandas as pd
import datetime, time
import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from ufo_data import UFOData
from scoring import Scoring
from species_identification import Identification

'''
First method of classifying the number of alien species visiting the US
We use DBScan to cluster groups of sightings that share a location and time period
Then we use the size of the clusters, along with other data,
to give a score for how credible each sighting is

Following that, we filter out non-credible sightings
and use only these data instances for a final k-means clustering using
values of K chosen based off of current Alien research.
For each # of alien species we try, we'll 
'''

def main():
    #load data
    names=['datetime', 'city','state', 'shape', 'duration(seconds)', 'comments','date posted', 'longitude', 'latitude']
    loader = UFOData(names, 'us')
    data = loader.encoded()

    #calculate credibility scores on data
    s = Scoring(data)
    scored_data = s.calc_scores()

    #Show all unclustered sightings that were reported as credible
    print("# rows = {}".format(len(scored_data.index)))
    #print(scored_data.loc[scored_data['dbscan_cluster'] == -1])

    #now calculate likely #'s of alien species
    p = Identification(scored_data)
    date_c = p.cluster_dates()
    shapes = p.cluster_ships()


print("running")
main()

#first read in data
"""
#/home/madelinm/School/ML/ML-FinalProject/ufo-scrubbed-geocoded-time-standardized.csv
names=['datetime', 'state','shape', 'duration(seconds)', 'comments','longitude', 'latitude']
loader = UFOData(names, 'us')
data = loader.encoded()

#start by clustering similar dates in similar areas
df_1_cols = ['latitude', 'longitude','datetime']
df_1 = data[['latitude', 'longitude','datetime']].apply(pd.to_numeric)

#print(df_1)

#We can't just straight up cluster by dates as ints
#that seems like a really bad idea
#it would end up 

#standardize
df_1_std = stats.zscore(df_1[df_1_cols])
print("standardized")
#print(df_1_std)

#Cluster the data
X = StandardScaler().fit_transform(df_1_std)

#1deg at 40deg latitude is 111034.61 meters
#set radius at 300 mt
eps = 260/111034.61 #The maximum distance between two samples for them to be considered as in the same neighborhood.

# data = df_merged[df_merged['uid']==uid][['lon','lat','venue_id']]
#df_1_std[['latitude', 'longitude','datetime']])
db = DBSCAN(eps=eps, min_samples=3, n_jobs=1).fit(X) # atleast 3 points in a cluster


#core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#core_samples_mask[db.core_sample_indices_] = True
data['dbscan_cluster'] =  db.labels_
#data['dbscan_core']    = core_samples_mask
print("# clusters: {}".format(len(set(data['dbscan_cluster']))))

#Just checking that it actually gave meaningful groupings
#Which it does!!!!
cluster1 = data.loc[data['dbscan_cluster'] == 1]
cluster1['datetime'] = cluster1['datetime'].apply(lambda X: datetime.datetime.fromtimestamp(int(X)).strftime('%Y-%m-%d %H:%M:%S'))
print(cluster1[names])
"""

'''
data = []

#cluster by date should really be split into clustering over years, months, etc
#we want to identify

#There has got to be a better way to do this?
for index, row in df.iterrows():
    lat = row['latitude']
    lon = row['longitude']
    date = ['datetime']
    data.append
    if is_number(lat) and is_number(lon):
        data.append([float(lon), float(lat), date])



        
data = np.array(data)
        
model = KMeans(n_clusters=20).fit(scale(data))
'''
#cluster on <lat, long, date>
#credibility score = 
#size of cluster
#time between sighting and report
#duration of sighting
#similar words in comments
#select only top sightings
#cluster with different k-means
#calculate entorpy with different #'s of clusters
#entropy = total likelihood that this is the correct # of species