# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 22:34:00 2023

@author: Sabina
"""

#Functions used for MST creation - Prims algorithm, Kruskal algorithm

from matplotlib import pyplot as plt
import heapq

def MSTPrim(points):
    n = len(points)
    mst = []
    visited = [False]*n
    distance = [float('inf')]*n
    parent = [None]*n
    distance[0] = 0
    heap = [(0, 0)]
    while heap:
        d, u = heapq.heappop(heap)
        visited[u] = True
        if parent[u] is not None:
            mst.append((u, parent[u], d))
        for v in range(n):
            if not visited[v]:
                d = getDistance(points[u][0], points[u][1], points[v][0], points[v][1])
                if d < distance[v]:
                    distance[v] = d
                    parent[v] = u
                    heapq.heappush(heap, (d, v))
    return mst

def MSTKruskal(points):
    n = len(points)
    result = []
    i, e = 0, 0
    graph = []
    for u in range(n):
        for v in range(u+1, n):
            d = getDistance(points[u][0], points[u][1], points[v][0], points[v][1])
            graph.append((u, v, d))
    graph = sorted(graph, key=lambda item: item[2])
    parent = [i for i in range(n)]
    rank = [0]*n
    while e < n - 1:
        u, v, w = graph[i]
        i = i + 1
        x = find(parent, u)
        y = find(parent, v)
        if x != y:
            e = e + 1
            result.append((u, v, w))
            union(parent, rank, x, y)
    return result

def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

def getDistance(x1, y1, x2, y2):
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

def plotMST(mst, pts, figNo, subplotNo, title, colorType, colors_for_nodes):

    xPoints = [p[0] for p in pts]
    yPoints  = [p[1] for p in pts]

    x1 = [pts[e[0]][0] for e in mst]
    y1 = [pts[e[0]][1] for e in mst]
    x2 = [pts[e[1]][0] for e in mst]
    y2 = [pts[e[1]][1] for e in mst]

    plt.figure(figNo).add_subplot(subplotNo)
    plt.title(title)
    
    plt.scatter(xPoints, yPoints, c = colors_for_nodes)

    for i in range(len(x1)):
        plt.plot([x1[i], x2[i]], [y1[i], y2[i]], colorType)

    plt.show()
    return xPoints, yPoints 