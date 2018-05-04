import pandas as pd
import datetime, time
import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

class Scoring:
    '''
    Used for assigning each sighting a credibility score.
    
    Scores are calculated based off of the number of similar sightings,
    the time taken between the sighting and reporting it,
    and the duration of the sighting
    '''
    def __init__(self, df):
        self.data = df

    def score_helper(self, sighting, size_dict):
        '''
        Calculates the credibility score for this sighting.
        The credibility score is defined by a heuristic using:
        -duration of the sighting,
        -time between sighting and report,
        -# of sightings in the same area around same time (cluster size)
        '''
        cluster = sighting['dbscan_cluster']
        if cluster != -1:
            c_size = size_dict[cluster]
        else:
            #Cluster -1 holds all unclustered sightings
            c_size = 1

        #time since sighting before reporting, divided by num. seconds in 1 year
        wait_time = (sighting['posted_ts'] - sighting['timestamp']) / 31536000

        #average cluster has a size of 20
        #So the average sighting that's clustered will be at around 100 - years unreported
        #Unclustered sightings will only have a positive score if they're reported quickly
        return (c_size * 5) - wait_time 

    def calc_scores(self):
        '''
        Score calculation is done by first clustering on date and location
        then using the size of those clusters as input to a heuristic,
        along with other dimensions of the data, to get a score

        Here we make the assumption that sightings that are in the same location
        on similar dates are most likely sightings caused by the same event, or 
        a related series of events. This way, we ignore possible misconceptions
        or human error in labelling the shape of the UFO. 
        '''
        s_data = self.data.copy()

        cluster_cols = ['latitude', 'longitude','timestamp']
        summary_cols = ['datetime','latitude', 'longitude', 'city', 'comments']

        temp_df = s_data[cluster_cols]
        X = StandardScaler().fit_transform(temp_df)
        eps = .038

        
        db = DBSCAN(eps=eps, min_samples=3, n_jobs=1).fit(X) # atleast 3 points in a cluster
        s_data['dbscan_cluster'] =  db.labels_
        print("# clusters: {}".format(len(set(s_data['dbscan_cluster']))))

        #Checking for meaningful groups
        cluster1 = s_data.loc[s_data['dbscan_cluster'] == 1]
        #print(cluster1[summary_cols])

        c_sizes = {}
        c_labels = set(s_data['dbscan_cluster'])

        max_cluster = -1
        min_cluster = len(cluster1.index)
        count = 0
        size_sum = 0

        print("Getting cluster sizes")
        #loop through each cluster to build a dictionary of cluster sizes
        for c in c_labels:
            c_df = s_data.loc[s_data['dbscan_cluster'] == c]
            size = len(c_df.index)
            c_sizes[c] = size
            count+=1
            size_sum+= size
            if size > max_cluster:
                max_cluster = size
            if size < min_cluster:
                min_cluster = size
        #cluster stats
        avg = (size_sum - (max_cluster + min_cluster)) / (count - 2)
        print("\n\n*** CLUSTER STATS ***\n")
        print("Max Cluster Size = {}".format(max_cluster))
        print("Min Cluster Size = {}".format(min_cluster))
        print("Avg. Size = {}".format(avg))

        #Sanity check
        if c_sizes[1] != len(cluster1.index):
            print("*** Error ****\nCluster1 size = {}\nc_sizes[1] = {}".format(len(cluster1.index), c_sizes[1]))
        else:
            print("Sanity Check Passed")

        print("Scoring all data")
        #calculate scores
        s_data['credibility'] = s_data.apply(lambda x: self.score_helper(x, c_sizes), axis=1)
        #s_data['credibility'] = stats.zscore(s_data['credibility'])

        print("Filtering scores")

        #Set -2 as my lowest possible credible report
        return s_data.loc[(s_data['credibility'] > -2) & (s_data['dbscan_cluster'] != -1)]

    def Filter_scores():
        print("Not Implemented")