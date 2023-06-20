# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 23:22:23 2023

@author: Sabina
"""

# Helper for reading functions

from xml.dom import minidom
import math
import numpy as np

def readCoordinatesFromFile(fileName):
    with open(fileName, 'r') as file:
        lines = file.readlines()
        coordinates = []
        for line in lines:
            columns = line.strip().split('\t')
            if len(columns) == 3:
                x_value = float(columns[1])
                y_value = float(columns[2])
                coordinates.append((x_value, y_value))
    return coordinates

def readAllMACAddressesFrom_XML(fileName):
    mac_addresses = []

    file = minidom.parse(fileName)

    wr = file.getElementsByTagName('wr')
    for elem in wr:
        wifi_APs = elem.getElementsByTagName('r')
        for aps in wifi_APs:
            mac_addresses.append(aps.attributes['b'].value)

    mac_addresses = np.unique(np.array(mac_addresses))
    return mac_addresses.tolist(), len(wr)


def readWifiFingerprintsFrom_XML(fileName, macAddresses, noOfSteps):
    fingerprint_matrix = np.zeros((noOfSteps, len(macAddresses)))
    file = minidom.parse(fileName)
    wr = file.getElementsByTagName('wr')
    step_no = 0
    for elem in wr:
        if(step_no < noOfSteps):
            wifi_APs = elem.getElementsByTagName('r')
            for aps in wifi_APs:
                mac_address = aps.attributes['b'].value
                index_of_AP = macAddresses.index(mac_address)
                fingerprint_matrix[step_no][index_of_AP] = aps.attributes['s'].value
            step_no = step_no + 1
    return fingerprint_matrix


def readCoordinates(fileName, noOfPoints):
    inf = 999
    graph = np.zeros((noOfPoints, noOfPoints)) # noOfPoints = 123 
    x = []
    y = []
    with open(fileName) as f:
        contents = f.read()
        rows = contents.split('\n')
        for currentIndex in range(0, noOfPoints):
            for nextIndex in range(3, 13): # to see the connections from current row to OX, OY, Diagonals (if the case) and room connections
                
                currentRow = rows[currentIndex]
                currentRowValuesSplitted = currentRow.split('\t')
                
                refPoint = currentRowValuesSplitted[nextIndex] # rand refPoint = 3
                if(refPoint != ''):
                    
                    nextRow = rows[int(refPoint) - 1]
                    nextRowValuesSplitted = nextRow.split('\t')
              
                    x = [float(currentRowValuesSplitted[1]), float(nextRowValuesSplitted[1])]
                    y = [float(currentRowValuesSplitted[2]), float(nextRowValuesSplitted[2])]
                 
                    graph[currentIndex][int(refPoint)-1] = math.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
                    graph[int(refPoint)-1][currentIndex] = math.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
 
    for row in range(0, len(graph)):
        for col in range(0, len(graph)):
            if(graph[row][col] == 0 and row != col):
                graph[row][col] = inf
    f.close()
    return graph