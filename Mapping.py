# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 23:40:03 2023

@author: Sabina
"""

# Helper for corridor and room recognition + room level transformation

import networkx as nx
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np

total_rooms = 3
each_room_color = ["m", "g", "k", "b"]

stress_free_floor_door_pts = np.array([[[ 2.55913399,  7.57757523],
        [ 3.29809704,  8.8829818 ]],
        [[-5.34534356, -6.33351676],
        [-5.64825018, -7.88486557]],
        [[-5.34534356, -6.33351676],
        [-4.47496079, -6.82572059]]])

fp_doors = np.array([[[-17.0639428 ,   3.56252322],
        [-16.13795463,   3.17821624]],
        [[-13.91248149,   3.39258076],
        [-13.55053099,   4.33081636]],
        [[ 13.77607574,  -8.19366989],
        [ 15.64568853,  -8.90413432]]])

def computeBetweennessCentrality(points, mst):
    G = nx.Graph()

    for i, point in enumerate(points):
        G.add_node(i)

    for edge in mst:
        u, v, weight = edge
        G.add_edge(u, v, weight=weight)

    betweenness_centrality = nx.betweenness_centrality(G)
    return betweenness_centrality

def plotCorridors(points, mst, betweenness_centrality, threshold, title, xlimMin, xlimMax, ylimMin, ylimMax):

    G = nx.Graph()

    for i, point in enumerate(points):
        G.add_node(i, pos=point)

    for edge in mst:
        u, v, weight = edge
        G.add_edge(u, v, weight=weight)

    corridor_nodes = [node for node, centrality in betweenness_centrality.items() if centrality >= threshold]
    print("Number of nodes recognized as corridors: ", len(corridor_nodes))
    print("Nodes recognized as corridors: ", corridor_nodes)
    
    corridor_graph = G.subgraph(corridor_nodes)
   
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    plt.title(title)

    pos = nx.get_node_attributes(corridor_graph, 'pos')
    nx.draw(corridor_graph, pos, with_labels=False, node_color='black', node_size=50)
    
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    plt.xlim(xlimMin, xlimMax)
    plt.ylim(ylimMin, ylimMax)
    plt.show()
    

def plotRooms(points, selected_nodes, title, figNo, recognizedRoomColors):
    plt.figure(figNo)
    plt.scatter([x for x, _ in points], [y for _, y in points], color='black')
    selected_x = [points[node][0] for node in selected_nodes]
    selected_y = [points[node][1] for node in selected_nodes]
    plt.scatter(selected_x, selected_y, color=recognizedRoomColors)
    plt.title(title)
    plt.show()
    
def plotRoomsAndCorridors(points, title, figNo, colors):
    plt.figure(figNo)
    plt.scatter([x for x, _ in points], [y for _, y in points], color=colors)
    plt.title(title)
    plt.show()
 
def getCoordinatesFromNodes(pts, rooms_nodes):
    rooms_pts = np.array([pts[node] for node in rooms_nodes])
    return rooms_pts
    
def plotRoomClusters(total_rooms, rooms_pts, color, title, n_ref_points = 3):
    kmeans, centers, labels = getKmeans(total_rooms, rooms_pts)
    all_cluster_points = []
    fig, axs = plt.subplots(1, total_rooms, figsize=(6.4, 4.8))

    for i, label in enumerate(set(labels)):
        cluster_points = np.round(rooms_pts[labels == label], decimals = 2)
        all_cluster_points.append(cluster_points)
        print("cluster_points for cluster ", i, " are: ", cluster_points)
        
        centroid = centers[label]
        axs[i].scatter(cluster_points[:,0], cluster_points[:,1], c=each_room_color[i])
          
        axs[i].scatter(centroid[0], centroid[1], marker='D', c='r', s=50)
        
        dist_to_centroid = np.linalg.norm(cluster_points - centroid, axis=1)
        sorted_indices = np.argsort(dist_to_centroid) ##[::-1]
        ref_points = cluster_points[sorted_indices[:n_ref_points]]
        axs[i].scatter(ref_points[:, 0], ref_points[:, 1], c='r', marker='*', s=100)
        axs[i].set_title(f"Cluster {label}")

    fig.suptitle(title)
    return all_cluster_points, labels

def getKmeans(total_rooms, pts):
    kmeans = KMeans(n_clusters=total_rooms).fit(pts)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    return kmeans, centers, labels

def getReferencePoints(centers, labels, rooms_pts, n_ref_points=3):
    ref_points = []
    
    for i, label in enumerate(set(labels)):
        cluster = rooms_pts[labels == label]
        centroid = centers[label]     
        dist_to_centroid = np.linalg.norm(cluster - centroid, axis=1)
        sorted_indices = np.argsort(dist_to_centroid)
        ref_points.append(cluster[sorted_indices[:n_ref_points]])
    return np.array(ref_points)

def computeTransformationMatrix(ref_points, fingerprint_ref_points):
    A = np.vstack([fingerprint_ref_points.T, np.ones((1, 2))])
    B = np.vstack([ref_points.T, np.ones((1, 2))])
    M, res, rank, s = np.linalg.lstsq(A.T, B.T, rcond=None)
    T = np.vstack([M, [0, 0, 1]])
    T = reshapeTransformationMatrix(T)
    return T

# #reshape it to 18x2 matrix for np.dot fct
def reshapeTransformationMatrix(T):
    # add a column of ones
    ones = np.ones((T.shape[0], 1))
    matrix_with_ones = np.hstack((T, ones))
    
    # reshape to 18x2
    reshapedMatrix = matrix_with_ones.reshape((-1, 2))
    return reshapedMatrix
  
def roomLevelTransformationWithRefPoints(total_rooms, room_ref_points, fp_ref_points, labels, fingerprint_rooms_pts):
    fingerprints_transformed = []
    for room_id, label in enumerate(set(labels)):
        fingerprint_cluster_points = fingerprint_rooms_pts[labels == label]
        T = computeTransformationMatrix(room_ref_points[room_id], fp_ref_points[room_id])
      
        fingerprints_transformed.append(np.dot(T, np.hstack([fingerprint_cluster_points.T, np.ones((2, 1))])).T[:, :2][:-1])
       
        # fingerprint_coords_transformed_homog = np.hstack((fingerprint_rooms_pts, np.ones((len(fingerprint_rooms_pts), 1))))

        # fingerprint_coords_transformed_homog = np.dot(fingerprints_transformed, fingerprint_coords_homog.T).T

        # fingerprint_coords_transformed.append(fingerprint_coords_transformed_homog[:, :2] / fingerprint_coords_transformed_homog[:, 2].reshape((-1, 1)))
        # print("For room_id = ", room_id, "fingerprint_coords_transformed = ", len(fingerprint_coords_transformed), " | " , len(fingerprint_coords_transformed[0])
    # return fingerprint_coords_transformed
    print("room_level_transformation len: ", len(fingerprints_transformed[0]), " | ",len(fingerprints_transformed[1]), " | ",len(fingerprints_transformed[2]))
    return fingerprints_transformed

def roomLevelTransformation(total_rooms, labels, fingerprint_rooms_pts):
    fingerprints_transformed = []
    for room_id, label in enumerate(set(labels)):
        fingerprint_cluster_points = fingerprint_rooms_pts[labels == label]
        T = computeTransformationMatrix(stress_free_floor_door_pts[room_id], fp_doors[room_id])
      
        fingerprints_transformed.append(np.dot(T, np.hstack([fingerprint_cluster_points.T, np.ones((2, 1))])).T[:, :2][:-1])
       
        # fingerprint_coords_transformed_homog = np.hstack((fingerprint_rooms_pts, np.ones((len(fingerprint_rooms_pts), 1))))

        # fingerprint_coords_transformed_homog = np.dot(fingerprints_transformed, fingerprint_coords_homog.T).T

        # fingerprint_coords_transformed.append(fingerprint_coords_transformed_homog[:, :2] / fingerprint_coords_transformed_homog[:, 2].reshape((-1, 1)))
        # print("For room_id = ", room_id, "fingerprint_coords_transformed = ", len(fingerprint_coords_transformed), " | " , len(fingerprint_coords_transformed[0])
    # return fingerprint_coords_transformed
    print("room_level_transformation len: ", len(fingerprints_transformed[0]), " | ",len(fingerprints_transformed[1]), " | ",len(fingerprints_transformed[2]))
    return fingerprints_transformed
    
def  plotFinalRooms(fingerprint_coords_transformed, figNo, title, colors):
    plt.figure(figNo)
    for i in range(0, total_rooms):  
        plt.scatter(fingerprint_coords_transformed[i][:, 0], fingerprint_coords_transformed[i][:, 1], c = each_room_color[i])
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        
    
def createTransformedFpPts(fingerprint_coords_transformed):
    new_fingerprint_coords_transformed = []
    for cluster in fingerprint_coords_transformed:
        for fingerprint in cluster:
            new_fingerprint_coords_transformed.append(fingerprint)
    return np.array(new_fingerprint_coords_transformed)
   
