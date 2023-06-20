# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 22:43:30 2023

@author: Sabina
"""

# Helper functions for MDS

from sklearn.manifold import MDS
from matplotlib import pyplot as plt
import numpy as np

def MDSAlgorithm(distMatrix):
    mds = MDS(metric=True, dissimilarity='precomputed', random_state=0)
    pts = mds.fit_transform(distMatrix)
    return pts


def plotMDS(pts, plotNo, title, colors, figNo):
    ax = plt.figure(figNo).add_subplot(plotNo)
    ax.scatter(pts[:, 0], pts[:, 1], c=colors) 
    plt.title(title)
    plt.show()

def plotMDS3D(pts, plotNo, title, colors, figNo):
    ax = plt.figure(figNo).add_subplot(plotNo, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], 0, zdir='z', c=[color for color in colors])
    plt.title(title)
    plt.show()

def floydWarshallFromGraph(graph, noOfVertices):
    dist = list(map(lambda i: list(map(lambda j: j, i)), graph))
    for k in range (0, noOfVertices):
        for i in range (0, noOfVertices):
            for j in range (0, noOfVertices):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist

def floydWarshall(points):
    n = len(points)
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i, j] = getEuclidDistance(points[i], points[j])

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if distances[i, j] > distances[i, k] + distances[k, j]:
                    distances[i, j] = distances[i, k] + distances[k, j]
    return distances

def getEuclidDistance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

