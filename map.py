# from mpl_toolkits.basemap import Basemap
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# if __name__ == '__main__':
#
#     # m = Basemap(projection='mill',
#     #             llcrnrlat=25,
#     #             llcrnrlon=-130,
#     #             urcrnrlat=50,
#     #             urcrnrlon=-60,
#     #             resolution='l')
#     #
#     # m.drawcoastlines()
#     # m.drawcountries(linewidth=2)
#     # m.drawstates(color='b')z
#
#
#     # setup Lambert Conformal basemap.
#     # set resolution=None to skip processing of boundary datasets.
#     # m = m = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90, llcrnrlon=-180,urcrnrlon=180,resolution='c')
#
#     m = Basemap(projection='ortho', resolution=None, lat_0=50, lon_0=-90)
#     m.bluemarble(scale=0.5)
#
#     # m.bluemarble()
#     # m.drawcoastlines()
#     # m.drawcountries(linewidth=1)
#     # m.drawstates(color='b')
#
#
#     # data = pd.read_csv('ufo-scrubbed-geocoded-time-standardized.csv',
#     #                    names=['date', 'city', 'state', 'country',
#     #                           'shape', 'size', 'duration', 'description',
#     #                           'sightingsDate', 'lon', 'lat'])
#     #
#     #
#     # for i in range(0, 1000):
#     #     xpt, ypt = m(float(data['lat'][i]), float(data['lon'][i]))
#     #
#     #     m.plot(xpt, ypt, 'r*', markersize=0.5)
#     #
#     # # for i in range(10000, 20000):
#     # #     xpt, ypt = m(float(data['lat'][i]), float(data['lon'][i]))
#     # #
#     # #     m.plot(xpt, ypt, 'r*', markersize=0.5)
#     # #
#     # # plt.show()
#     # # for i in range(20000, 30000):
#     # #     xpt, ypt = m(float(data['lat'][i]), float(data['lon'][i]))
#     # #
#     # #     m.plot(xpt, ypt, 'r*', markersize=0.5)
#     #
#     #
#     # plt.title('Basemap')
#     # plt.show()
#

import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
uid = 4902

#1deg at 40deg latitude is 111034.61 meters
#set radius at 300 mt
eps = 100#1000/111034.61 #The maximum distance between two samples for them to be considered as in the same neighborhood.

data = pd.read_csv('ufo-scrubbed-geocoded-time-standardized.csv',
                   names=['date', 'city', 'state', 'country',
                          'shape', 'size', 'duration', 'description',
                          'sightingsDate', 'longitude', 'latitude'])
data = data[data['country'] == 'us']

coords = data[['longitude', 'latitude']].astype(float)

# data = df_merged[df_merged['uid']==uid][['lon','lat','venue_id']]
# db = DBSCAN(eps=eps, min_samples=3, n_jobs=1).fit(np.array(data[['longitude','latitude']])) # atleast 3 points in a cluster
#
#
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# data['dbscan_cluster'] =  db.labels_
# data['dbscan_core']    = core_samples_mask
# print("# clusters: {}".format(len(set(data['dbscan_cluster']))))
#
# plt.style.use('seaborn-white')
# fig = plt.figure()
# fig.set_size_inches(20, 20)
#
# # empty = pd.DataFrame(columns=['longitude', 'latitude'])
# # geomap(empty, 13, 2, 'k', 0.1)
#
# ##############################
# plt.figure(figsize=(10, 10))
#
# # m = Basemap(projection='mill', llcrnrlat = 26, urcrnrlat = 50, llcrnrlon = -120, urcrnrlon = -65,
# #              resolution = 'h')
# m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
#             projection='lcc',lat_1=32,lat_2=45,lon_0=-95)
#
# # m = Basemap(projection='ortho', resolution=None, lat_0=50, lon_0=-90)
# m.bluemarble(scale=1)
# m.drawcoastlines()
# m.drawcountries()
# m.drawstates()

# for i in range(0, 1000):
# xpt, ypt = m(float(data['latitude'][i]), float(data['longitude'][i]))
# xpt = [float(data['latitude'][i]) for i in range(0, 1000)]
# ypt = [float(data['longitude'][i]) for i in range(0, 1000)]

# for i in range(0, 10000):
#         xpt, ypt = m(float(data['latitude'][i]), float(data['longitude'][i]))

#         m.plot(xpt, ypt, 'ro', markersize=0.5)

##############################

# unique_labels = sorted(set(data['dbscan_cluster']))
#
# for k in unique_labels:
#     xy = data[data['dbscan_cluster']==k]
#
#     # xy.head()
#
# #     xpt, ypt = m(float(data['latitude'][i]), float(data['longitude'][i]))
#
# #     for i in range(0, 10000):
# #         xpt, ypt = m(float(data['latitude'][i]), float(data['longitude'][i]))
# #
# #
# # #         m.plot(xpt, ypt, 'ro', markersize=0.5)
#
#     m.plot(xy['latitude'], xy['longitude'], 'kD' if k < 0 else 'o', markersize=8)
# plt.show()




from subprocess import check_output

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

df = data
df = df[df['country'] == 'us']


data = []

for index, row in df.iterrows():
    lat = row['latitude']
    lon = row['longitude']
    if is_number(lat) and is_number(lon):
        data.append([float(lon), float(lat)])

data = np.array(data)

# model = KMeans(n_clusters=20).fit(scale(data))

kms_per_radian = 6371.0088
epsilon = 20 / kms_per_radian
db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
cluster_labels = db.labels_
num_clusters = len(set(cluster_labels))
clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
print('Number of clusters: {}'.format(num_clusters))


plt.scatter(data[:, 0], data[:, 1], c=db.labels_.astype(float))
plt.show()

'''
data = coords[:20000]
plt.figure(figsize=(12,8))
# Map = Basemap(projection='mill', llcrnrlat = 26, urcrnrlat = 50, llcrnrlon = -120, urcrnrlon = 22,
#               resolution = 'h')
Map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=32,lat_2=45,lon_0=-95, resolution='i')
Map.drawcoastlines()
Map.drawcountries()
Map.drawstates()

for index, row in data.iterrows():
    xpt, ypt = Map(float(row['latitude']), float(row['longitude']))
    Map.plot(xpt, ypt, 'ro', markersize=0.5)

plt.show()

# m.plot(xpt, ypt, 'r*', markersize=0.5)

plt.title('Basemap')
plt.show()

# x, y = Map(list(data['longitude'].astype("float")), list(data['latitude'].astype(float)))
# Map.plot(x, y, 'go', markersize = 1, alpha = 0.5, color = 'blue')
#
# plt.show()
'''