# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 23:25:53 2022

@author: sabina
"""

import numpy as np

from MST import *
from MDS import *
from HelperForReadingData import *
from Errors import *
from Mapping import *
 

def convert_to_no_of_steps(steps_distances):
    return steps_distances * 2


### Creation of stress-free floor plan using personal interpretation ###
# no_of_samples = 123
# graph_samples = readCoordinates('Coordinates.txt', no_of_samples)
# samples_distances = floydWarshall(graph_samples, no_of_samples)

## Creation of stress-free-floor plan using Floyd Warshal algorithm directly on coordinates
samples_coordinates = readCoordinatesFromFile('Coordinates2.txt')
samples_distances = floydWarshall(samples_coordinates)

colors_sample = ['m'] * 24 + ['b'] * 18 + ['k'] * \
    45 + ['g'] * 36

stress_free_floor_pts = MDSAlgorithm(samples_distances)
plotMDS(stress_free_floor_pts, 121, '2D stress free floor plan', colors_sample, 1)
plotMDS3D(stress_free_floor_pts, 122, '3D stress free floor plan', colors_sample, 1)

### MAC Addresses ###
mac_addresses, no_of_steps = readAllMACAddressesFrom_XML('Sensor_readings.xml')
fingerprint_matrix = readWifiFingerprintsFrom_XML('Sensor_readings.xml', mac_addresses, no_of_steps)

# dissimilarity_matrix = cdist(fingerprint_matrix, fingerprint_matrix, metric='correlation')
# sample_fingerprint_matrix = floydWarshall(dissimilarity_matrix, len(dissimilarity_matrix))

colors_for_steps = ['m'] * 6 + ['k'] * 2 + ['b'] * 12 + ['k'] * \
    18 + ['g'] * 24 + ['k'] * 22 + ['m'] * 7
    
steps_coordinates = readCoordinatesFromFile('Step_coordinates.txt')
steps_distances = floydWarshall(steps_coordinates)
steps_distances_converted = convert_to_no_of_steps(steps_distances)

fingerprint_pts = MDSAlgorithm(steps_distances_converted)
plotMDS(fingerprint_pts, 121, '2D fingerprint space', colors_for_steps, 2)
plotMDS3D(fingerprint_pts, 122, '3D fingerprint space',  colors_for_steps, 2)

### MST using Prims and Kruskal algorithms + plots for stress free floor plan ###
mstPrims = MSTPrim(stress_free_floor_pts)
mstKruskal = MSTKruskal(stress_free_floor_pts)
plotMST(mstPrims, stress_free_floor_pts, 3, 121, "MST applyed on stress free floor plan using Prim", 'm-', colors_sample)
xPoints, yPoints = plotMST(mstKruskal, stress_free_floor_pts, 3, 122, "MST applyed on stress free floor plan  using Kruskal", 'y-', colors_sample)


### MST using Prims and Kruskal algorithms + plots for fingerprintspace ###
mstPrims_fp = MSTPrim(fingerprint_pts)
mstKruskal_fp  = MSTKruskal(fingerprint_pts)
plotMST(mstPrims_fp , fingerprint_pts, 4, 121, "MST applyed on fingerprint space using Prim", 'm-', colors_for_steps)
xPoints, yPoints = plotMST(mstKruskal_fp , fingerprint_pts, 4, 122, "MST applyed on fingerprint space using Kruskal", 'y-', colors_for_steps)

# # c = area_ratio(pts[42:87], pts)
corridor_area = 48
total_area = 116
area_ratio = corridor_area/total_area

# Corridor recognition for stress free floor plan
betweenness_centrality = computeBetweennessCentrality(stress_free_floor_pts, mstPrims)    
plotCorridors(stress_free_floor_pts, mstPrims, betweenness_centrality, area_ratio, "Corridor recognition applied on MST Prims for stress free floor plan", -8, 12, -15, 15)
rooms_nodes = [node for node, centrality in betweenness_centrality.items() if centrality < area_ratio]

# Corridor recognition for fingerprint space
betweenness_centrality_fp = computeBetweennessCentrality(fingerprint_pts, mstPrims_fp)
for node, centrality in betweenness_centrality_fp.items():
    print(f"Node {node}: Betweenness Centrality = {centrality}")
plotCorridors(fingerprint_pts, mstPrims_fp, betweenness_centrality_fp, area_ratio, "Corridor recognition applied on MST Prims for fingerprint", -30, 30, -15, 15)


rooms_nodes_fp = [node for node, centrality in betweenness_centrality_fp.items() if centrality < area_ratio]
corridor_nodes_fp = [node for node, centrality in betweenness_centrality_fp.items() if centrality > area_ratio]

recognized_room_colors_for_error = ['m'] * 7 + ['b'] * 14 + ['k'] * \
    4 + ['c'] * 1 +  ['k'] * 8 + ['c'] * 4 + ['g'] * 25 + ['c'] * \
    2 + ['k'] * 4 + ['c'] * 4 + ['k'] * 8 + ['c'] * 4 + ['m'] * 6

recognized_room_colors = ['m'] * 6  + ['c'] * 2  + ['b'] * 12  + ['c'] * 8 + ['g'] * 22 + ['c'] * 9 + ['m'] * 7
    
plotRooms(fingerprint_pts, rooms_nodes_fp, 'Recognized rooms and corridors', 7, recognized_room_colors)

plotRoomsAndCorridors(fingerprint_pts, 'Rooms and corridors highlighting the error', 8, recognized_room_colors_for_error)

total_rooms = 3
each_room_color = ["m", "g", "k", "b"]

rooms_pts = getCoordinatesFromNodes(stress_free_floor_pts, rooms_nodes)
rooms_fp_pts = getCoordinatesFromNodes(fingerprint_pts, rooms_nodes_fp)

plotRoomClusters(total_rooms, rooms_pts, each_room_color, "Kmeans clusters from stress free floor plan points - rooms")
all_cluster_fp_pts, labels_fp_room_for_error = plotRoomClusters(total_rooms, rooms_fp_pts, each_room_color, "Kmeans clusters from fingerprint space points - rooms")


# ref_points_room = getReferencePoints(centers_room, labels_room, rooms_pts)
# ref_points_fp_room = getReferencePoints(centers_fp_room, labels_fp_room, rooms_fp_pts)

# fingerprint_coords_transformed = roomLevelTransformationWithRefPoints(total_rooms, ref_points_room, ref_points_fp_room, labels_fp_room, rooms_fp_pts)
# plotFinalRooms(fingerprint_coords_transformed, 11, 'Transformed Fingerprint Coordinates', each_room_color)

fingerprint_coords_transformed = roomLevelTransformation(total_rooms, labels_fp_room_for_error, rooms_fp_pts)

plotFinalRooms(fingerprint_coords_transformed, 11, 'Transformed Fingerprint Coordinates', each_room_color)

newFingerprintCoordsTransformed = createTransformedFpPts(fingerprint_coords_transformed)
new_cluster_fp_pts = createTransformedFpPts(all_cluster_fp_pts)

avg_error_general, avg_error_general_perc, std_error_general = plotCDFError(new_cluster_fp_pts, newFingerprintCoordsTransformed, 12, 'Cumulative Distribution Function of mapping error')
avg_error_clusters, avg_error_clusters_perc, std_error_clusters = plotCDFErrorForClusters(all_cluster_fp_pts, fingerprint_coords_transformed, 13, 'Cumulative Distribution Function of mapping error - room clusters')
