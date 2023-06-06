import osmread
import osmnx as ox
import requests
import urllib3
urllib3.disable_warnings()
from pathlib import Path
import requests
from leuvenmapmatching.util.gpx import gpx_to_path
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching import visualization as mmviz
from leuvenmapmatching.map.inmem import InMemMap
import numpy as np
import matplotlib.pyplot as plt
import os

utn = ox.settings.useful_tags_node
oxna = ox.settings.osm_xml_node_attrs
oxnt = ox.settings.osm_xml_node_tags
utw = ox.settings.useful_tags_way
oxwa = ox.settings.osm_xml_way_attrs
oxwt = ox.settings.osm_xml_way_tags
utn = list(set(utn + oxna + oxnt))
utw = list(set(utw + oxwa + oxwt))
ox.settings.all_oneway = True
ox.settings.useful_tags_node = utn
ox.settings.useful_tags_way = utw

#place = "Shanghai, China"
cf = '["highway"~"motorway|trunk|primary|secondary|tertiary"]'  # 不下载 村镇道路|residential||unknown|unclassified|track|path|service
graph = ox.graph_from_bbox(40.5900, 40.5166, -3.4013, -3.5670,  network_type = 'all',
                        #custom_filter=cf,
                           simplify=False)
#G = ox.graph_from_place(place,
#                        network_type="drive",
#                        which_result=2,
#                        custom_filter=cf)
ox.plot_graph(graph, node_size=2, save=True, filepath="G11.png", dpi=800)
# Gfile 为保存到本地的路径
#ox.save_graphml(graph, 'graph.graphml')
#ox.io.save_graph_shapefile(G, filepath="/graph", encoding="utf-8", directed=False)


ox.save_graph_xml(graph , filepath='./graph4.xml')
#ox.save_graph_graphml(graph , filepath='./data/gra')
#osmnx.io.save_graphml(graph , filepath='./data/gra')
# 相对应的加载就是 load_graphml
#G = ox.load_graphml(Gfile)

# 构建网络

xml_file = Path(".") / "graph4.xml"

map_con = InMemMap("myosm", use_latlon=True, use_rtree=True, index_edges=True)
for entity in osmread.parse_file(str(xml_file)):
    if isinstance(entity, osmread.Way) and 'highway' in entity.tags:
        for node_a, node_b in zip(entity.nodes, entity.nodes[1:]):
            map_con.add_edge(node_a, node_b)
            # Some roads are one-way. We'll add both directions.
            map_con.add_edge(node_b, node_a)
    if isinstance(entity, osmread.Node):
        map_con.add_node(entity.id, (entity.lat, entity.lon))
map_con.purge()


f=open("RAW_GPS.txt")
line=f.readline()
track=[]
while line:
    info=line.split()
    info_raw=(float(info[2]),float(info[3]),float(info[0]))
    track.append(info_raw)
    line=f.readline()
f.close()

track1=track[0:623:5]#track1是三列（纬度 经度 时间）
track2=np.array(track1)#将列表list变成数组arry
#print (track2)
#X = track2[:,0]#只取出数组的第一列和第二列，不需要第三列（时间戳）
#Y = track2[:,1]
Z = track2[:,2]#留给加噪再抽样后加上时间戳
#print (X)
#plt.plot(X,Y,linestyle='',marker='.')
track_downsample=np.array([track2[:,0],track2[:,1]]).T#将数据第一列和第二列合并为新的轨迹数组
#print (track_downsample)
np.shape(track_downsample)
#print (track_downsample[1])

cov = np.array([[0.0000005, 0], [0, 0.0000005]])  # 协方差矩阵，影响高斯噪声的范围
a = np.empty((125, 2))  # 高斯噪音的均值点，也是数据抽样后的点。
x = np.empty((125, 5, 2))  # 定义三维函数来储存高斯加噪后的数据，即将42*2的数组扩充十倍

# mean[i]= np.empty((42, 2))
for i in range(125):
    a[i] = track_downsample[i]
    x[i] = np.random.multivariate_normal(a[i], cov, (5,), 'raise')#以a[i]为均值点，以cov为协方差

np.shape(x)
#print(x[1])
# print (x)
B = np.reshape(x, (-1, 2))#将x设为两列行数未知
# print(B)
print('B的维度：', B.shape)

#plt.scatter(B[:, 0], B[:, 1])
#plt.xlim(40.5, 40.6)
#plt.ylim(-3.56, -3.39)
#plt.show()
Z1=np.array([Z]).T
B1 = B[0:625:5]

Tr = np.hstack((B1, Z1))  # 给加噪后的坐标点后加上时间戳

# print(Tr)
tr= Tr[15:]#截取后110个点
Tr1 = tr.tolist()  # 数组转列表
# print(Tr1)
for i in range(85):  # 将列表转元组
    Tr1[i] = tuple(Tr1[i])

# create mapmatcher
matcher = DistanceMatcher(map_con,
                          max_dist=500,#meter Maximum distance from path (this is a hard cut, min_prob_norm should be better)
                          max_dist_init=1700,#Maximum distance from start location (if not given, uses max_dist)
                          min_prob_norm=0.001,#Minimum normalized probability of observations (ema)
                          non_emitting_length_factor=0.95,#Reduce the probability of a sequence of non-emitting states the longer it is. This can be used to prefer shorter paths. This is separate from the transition probabilities because transition probabilities are averaged for non-emitting states and thus the length is also averaged out.
                          obs_noise=50,#Standard deviation of noise
                          obs_noise_ne=50,#Standard deviation of noise for non-emitting states (is set to obs_noise if not given)
                          dist_noise=50,#Standard deviation of difference between distance between states and distance
                          max_lattice_width=20,
                          non_emitting_states=True)
# mapmatching
states, _ = matcher.match(track1, unique=False)
# plot the result
mmviz.plot_map(map_con, matcher=matcher,
               show_labels=False, show_matching=False,  show_graph=True,
               filename="outcccc5.png")
