# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 23:30:50 2023

@author: Sabina
"""
# Helper for plotting the errors 
from matplotlib import pyplot as plt
import numpy as np

total_rooms = 3
each_room_color = ["m", "g", "k", "b"]

def plotCDFErrorForClusters(true_coords, pred_coords, figNo, title):
    avg_error = []
    std_error = []
    avg_error_perc = []
    
    cdf_each_cluster = []
    sorted_error_each_cluster = []
    
    plt.figure(figNo)
    plt.xlabel('Error (m)')
    plt.ylabel('Cumulative Probability')
    plt.title(title)
    plt.grid(True)
    
    for cluster_index in range(0, total_rooms): 
        true_coords_sorted_cluster, pred_coords_sorted_cluster = sort_clusters(true_coords[cluster_index], pred_coords[cluster_index])
        error = np.sqrt(np.sum((true_coords_sorted_cluster - pred_coords_sorted_cluster)**2, axis=1))
        
        sorted_error = np.sort(error)
        
        cdf = np.arange(1, len(sorted_error) + 1) / len(sorted_error)
        avg_error_perc.append(np.mean(cdf))
        avg_error.append(np.mean(error))
        std_error.append(np.std(error))
        cdf_each_cluster.append(cdf)
        sorted_error_each_cluster.append(sorted_error)
    
    for cluster_index in range(0, total_rooms):     
        plt.plot(sorted_error_each_cluster[cluster_index], cdf_each_cluster[cluster_index], color=each_room_color[cluster_index], label=f"Cluster {cluster_index}")
    plt.legend()
    plt.show()
    return avg_error, avg_error_perc, std_error

def sort_clusters(true_coords, pred_coords):
    true_coords_sorted_cluster = true_coords[np.argsort(true_coords[:, 0])]
    pred_coords_sorted_cluster = pred_coords[np.argsort(pred_coords[:, 0])]
    
    return true_coords_sorted_cluster, pred_coords_sorted_cluster

def plotCDFError(true_coords, pred_coords, figNo, title):
    true_coords_sorted_cluster, pred_coords_sorted_cluster = sort_clusters(true_coords, pred_coords)
    error = np.sqrt(np.sum((true_coords_sorted_cluster - pred_coords_sorted_cluster)**2, axis=1))
 
    sorted_error = np.sort(error)
    
    cdf = np.arange(1, len(sorted_error) + 1) / len(sorted_error)
    
    avg_error_perc = np.mean(cdf)
    avg_error = np.mean(error)
    std_error = np.std(error)
    plt.figure(figNo)
    plt.plot(sorted_error, cdf)
    plt.xlabel('Error (m)')
    plt.ylabel('Cumulative Probability')
    plt.title(title)
    plt.grid(True)
    plt.show()
    return avg_error, avg_error_perc, std_error