import pandas as pd
import datetime, time
import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

#Things to do today
#
#Finish credibility score calculation

class Identification:
    '''
    Used for analyzing data to see how many alien species are likely to have visited earth.
    '''

    def __init__(self, df):
        self._data = df.copy()
        self.summary_cols = ['datetime','latitude', 'longitude', 'city', 'comments']

    def cluster_dates(self):
        """
        take 2 sightings to represent each cluster
        Then use that, and cluster on date
        See how many distinct clusters there are
        """
        data = self._data.copy()
        df = data.groupby('dbscan_cluster').head(2)

        temp_df = df['timestamp'].reshape(-1,1)
        X = StandardScaler().fit_transform(temp_df)
        eps = .038
         
        db = DBSCAN(eps=eps, min_samples=3, n_jobs=1).fit(X) # atleast 3 points in a cluster
        df['date_cluster'] =  db.labels_

        date_clusters = len(set(df['date_cluster']))

        print("# of date clusters: {}".format(date_clusters))

        return date_clusters

    def cluster_ships(self):
        """
        cluster sightings by UFO shape, and similar other sightings (in case people saw the ships differently)

        take different #'s of ships, and see how many instances belong in that cluster
        """
        df = self._data.copy()

        temp_df = df[['shape', 'dbscan_cluster']]
        X = StandardScaler().fit_transform(temp_df)
        eps = .038
         
        db = DBSCAN(eps=eps, min_samples=3, n_jobs=1).fit(X) # atleast 3 points in a cluster
        df['shape_cluster'] =  db.labels_

        shape_clusters = len(set(df['shape_cluster']))
        

        print("# of shape/cluster groupings: {}".format(shape_clusters))

        shape_counts = {}
        c_labels = set(df['shape_cluster'])

        #For each cluster, find the best shape and store the count
        for c in c_labels:
            c_df = df.loc[df['shape_cluster'] == c]
            count = len(c_df.index)
            item_counts = c_df['shape'].value_counts()
            UFO_shape = max(item_counts.keys(), key=(lambda key: item_counts[key]))
            if not UFO_shape in shape_counts:
                shape_counts[UFO_shape] = count
            else:
                shape_counts[UFO_shape] += count

        #Now have UFO shapes -> number of sightings they describe

        print("Total # of sightings = {}".format(len(df.index)))
        print(shape_counts)