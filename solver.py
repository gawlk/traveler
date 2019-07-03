#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---
# IMPORTS
# ---

import csv
import math
import numpy
import os
import random
import sys
import time

# ---
# FUNCTIONS
# ---

# ---
# GENERIC
# ---

'''
Return the distance between two points
'''
def distance(p1, p2):
    return math.sqrt((p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

'''
Get a distance from a database
If it doesn't exist, calculate it and add it to the database
'''
def distanceDatabase(p1, p2, db):
    d = db[p1[0] - 1][p2[0] - 1]

    # Save distance if doesn't already in the distances database
    if (d is None):
        d = distance(p1, p2)
        db[p1[0] - 1][p2[0] - 1] = d
        db[p2[0] - 1][p1[0] - 1] = d

    return d

'''
Return the list of points as a string with the point number 1 as the head and tail
and the total distance between all points
'''
def getResults(points):
    string = ""
    buffer = ""
    last = None
    total = 0
    for point in points:
        if (point[0] == 1):
            buffer = string
            string = ""
        if (last is not None):
            total += distance(last, point)
        string += str(point[0]) + "-"
        last = point
    string += buffer + "1"
    total += distance(last, points[0])

    return (string, total)

# ---
# PERMUTATION
# ---

'''
Return a tuple containing the path as a string and the total distance of the path
Arguments are the path as a list of tuples like so: (number, x, y)
The border argument is the first and the last point of the path
We assume here that it's number is '1'
All the calculated distances are stored in a database which reduces considerably the calculation time
'''
def calculatePermutation(border, path, distances):
    string = '1-'
    total = 0
    last = border

    for current in path:
        string += str(current[0]) + '-'
        total += distanceDatabase(last, current, distances)
        last = current

    total += distanceDatabase(last, border, distances)
    string += '1'

    return (string, total)

'''
Recursive function that calculates all the permutations based on the Heap's algorithm
It goes like so
1, 2, 3, 4, 5, 6
1, 2, 3, 4, 6, 5
1, 2, 3, 5, 4, 6
1, 2, 3, 5, 6, 4
1, 2, 3, 6, 5, 4
1, 2, 3, 6, 4, 5
...
'''
def _permute(l, from_i, to_i, results):
    if (from_i == to_i):
        results.append(l.copy())
    else:
        for i in range(from_i, to_i + 1):
            l[from_i], l[i] = l[i], l[from_i] # swap, no swap if i = from_i
            _permute(l, from_i + 1, to_i, results)
            l[from_i], l[i] = l[i], l[from_i] # undo the swap

'''
Mother function of _permute, mostly useless, just to make the call easier
'''
def permute(l):
    results = []
    _permute(l, 0, len(l) - 1, results)
    return results

'''
Return the best solution and total distance by comparing every possible path
'''
def solvePermutation(points):
    best = None
    border = points.pop(0)

    # Get all the permutations without the first point
    permutations = permute(points)

    # Create a database where we'll store all the calculated distances
    distances = [ [ None for i in range(len(points) + 1) ] for j in range(len(points) + 1) ]

    # Find the best path
    for permutation in permutations:
        result = calculatePermutation(border, permutation, distances)
        if (best is None or result[1] < best[1]):
            best = result

    return best

# ---
# NEAREST
# ---

'''
Return the solution and total distance using the the nearest point technique
We start with the point number 1 and connect it to his closest point and so on, until all the point are connected. Then, we connect the last point to the first one and return the results
'''
def solveNearest(points):
    border = points.pop(0)
    current = border
    string = '1-'
    total = 0

    # While haven't treated all the points
    while points:
        # Choose the closest point to the current one
        bestDistance = None
        for point in points:
            d = distance(current, point)
            if (bestDistance is None or d < bestDistance):
                bestDistance = d
                bestPoint = point

        # Add to path and remove from list
        string += str(bestPoint[0]) + '-'
        total += bestDistance
        current = bestPoint
        points.remove(bestPoint)
    
    # Don't forget to add one more time the first point to the path
    string += '1'
    total += distance(current, border)

    return (string, total)

# ---
# NEAREST IMPROVED
# ---

'''
For each point in points apply the nearest point algorithm
We save the calculated distances to save some calculation time (which works quite well!)
If a path happens to be shorter than the best path, replace it
'''
def solveNearestImproved(points):
    distances = [ [ None for i in range(len(points)) ] for j in range(len(points)) ]
    best = None
    
    for current in points:
        tmpPoints = points.copy()
        tmpCurrent = current
        path = []
        total = 0

        # While haven't treated all the points
        while tmpPoints:
            # Choose the closest point to the current one
            bestDistance = None
            for point in tmpPoints:
                d = distanceDatabase(tmpCurrent, point, distances)

                # Choose the closest point to the current one
                if (bestDistance is None or d < bestDistance):
                    bestDistance = d
                    bestPoint = point

            path.append(bestPoint)
            tmpCurrent = bestPoint
            total += bestDistance
            tmpPoints.remove(bestPoint)
        
        if (best is None or total < best[1]):
            best = (path.copy(), total)

    return getResults(best[0])

# ---
# NEAREST INSERTION
# ---

'''
Return the solution and total distance using the the nearest point technique
We create here a cycle and add the nearest point of the remaining list of points to it until there is no point left
The point will be connected to the closest point from the cycle
'''
def solveNearestInsertion(points):
    done = [ points.pop(0) ]

    while points:
        # Find the city closest to the current cycle
        best = None
        for current in done:
            for point in points:
                d = distance(current, point)
                if (best is None or d < best[2]):
                    best = (current, point, d)

        # Find the second city from the cycle closest to the new city without direct crossing
        index = done.index(best[0])
        distanceNeighborL = distance(done[index - 1], best[1])
        distanceNeighborR = distance(done[(index + 1) % len(done)], best[1])
        distanceCurrentL = distance(done[index - 1], best[0])
        distanceCurrentR = distance(done[(index + 1) % len(done)], best[0])
        if (distanceNeighborL + best[2] + distanceCurrentR < distanceCurrentL + best[2] + distanceNeighborR):
            newIndex = index
        else:
            newIndex = (index + 1) % len(done)
        done.insert(newIndex, best[1])

        points.remove(best[1])

    return getResults(done)

# ---
# FARTHEST INSERTION
# ---

'''
Return the solution and total distance using the the farthest point technique
We create here a cycle and add the farthest point to it by breaking the closest edge from the cycle and connecting the 2 points of that edge to the farthest point

Time is very similar with NearestInsertion but it has much better results
'''
def solveFarthestInsertion(points):
    distances = [ [ None for i in range(len(points)) ] for j in range(len(points)) ]
    done = [ points.pop(0) ]

    while points:
        farthestPoint = None

        # Find the farthest city to the current cycle
        for current in done:
            for point in points:
                d = distanceDatabase(current, point, distances)
                if (farthestPoint is None or d > farthestPoint[1]):
                    farthestPoint = (point, d)
        farthestPoint = farthestPoint[0]

        # Find the closest edge of the cycle to the farthest city
        last = None
        best = None
        for current in done:
            if (last is not None):
                d1 = distanceDatabase(last, farthestPoint, distances)
                d2 = distanceDatabase(current, farthestPoint, distances)
                d3 = distanceDatabase(last, current, distances)
                d = d1 + d2 - d3

                if (best is None or d < best[2]):
                    best = (last, current, d)
            last = current

        if (last is None):
            last = done[0]

        d1 = distanceDatabase(last, farthestPoint, distances)
        d2 = distanceDatabase(done[0], farthestPoint, distances)
        d3 = distanceDatabase(last, done[0], distances)
        d = d1 + d2 - d3
        if (best is None or d < best[2]):
            best = (last, done[0], d)

        # Connect the farthest city to the cycle
        done.insert(done.index(best[0]) + 1, farthestPoint)
        points.remove(farthestPoint)

    return getResults(done)

# ---
# RANDOM NODE INSERTION
# ---

'''
Return the solution and total distance using the node insertion technique
We take a random point from the list and put it back at the best possible place and we do this until there are no more changes for a certain amount of time
'''
def solveRandomNodeInsertion(points):
    same = 0
    sameMax = len(points) * 2
    distances = [ [ None for i in range(len(points)) ] for j in range(len(points)) ]

    if (len(points) > 3):
        # While there was a change not too long ago
        while (same < sameMax):
            # Take random point out of the group
            index = random.randint(0, len(points) - 1)
            point = points.pop(index)

            # Find the best edge of the group to which it can be connected
            last = None
            best = None
            for current in points:
                if (last is not None):
                    d1 = distanceDatabase(last, point, distances)
                    d2 = distanceDatabase(point, current, distances)
                    d3 = distanceDatabase(last, current, distances)
                    d = d1 + d2 - d3
                    if (best is None or d < best[2]):
                        best = (last, current, d)
                last = current

            d1 = distanceDatabase(last, point, distances)
            d2 = distanceDatabase(point, points[0], distances)
            d3 = distanceDatabase(last, points[0], distances)
            d = d1 + d2 - d3
            if (d < best[2]):
                best = (last, points[0], d)

            # Put it back between the 2 points that made the closest edge to the point
            points.insert(points.index(best[0]) + 1, point)

            # If the point changed place, reset the counter of the last happening change, if not, increase the counter
            if (index != points.index(best[0]) + 1):
                same = 0
            else:
                same += 1

    return getResults(points)

# ---
# ANT COLONY OPTIMIZATION
# ---

'''
Choose the next city an ant should go to
pheromone: list of pheromones on link to other cities
dist: list of distances to other cities
visited: list of visited cities
'''
def pickCity(pheromone, distances, visited):
    alpha = 1 # pheromone's coefficient
    beta = 3 # distances coefficient

    # Copy list because we'll apply lots of shit on it
    row = pheromone.copy()

    # Pheromone array put all the visited cities to 0
    # in order to skip them
    for city in visited:
        row[city] = 0

    # Calculate all the values using the formula
    # (pheromone ** alpha) * (1 / (distances ** beta))
    for i in range(len(row)):
        if (distances[i] < 0):
            row[i] = 0
        else:
            row[i] = (row[i] ** alpha) * (1 / (distances[i]) ** beta)

    # Generate probabilities for each city
    row_sum = 0
    for value in row:
        row_sum += value

    # Make total of array equal to one
    probabilities = row
    for i in range(len(probabilities)):
        probabilities[i] /= row_sum

    # Return one random city using probabilities
    return numpy.random.choice(len(probabilities), p = probabilities)

'''
Generate a path
Start each time at the first city of the list
'''
def genPath(distances, pheromones):
    path = []
    start = 0 # Start at the first city of the list
    current = start

    visited = [ start ]

    for i in range(len(distances) - 1):
        city = pickCity(pheromones[current], distances[current], visited)

        path.append((current, city))
        visited.append(city)
        current = city

    path.append((current, start)) # close the loop

    return path

'''
Return the total distance of a path
'''
def genPathDist(path, distances):
    total_dist = 0
    for link in path:
        if (link[0] != link[1]):
            total_dist += distances[link[0]][link[1]]
    return total_dist

'''
Generate all paths (= one path per ant) and return them
'''
def genAllPaths(n_ants, distances, pheromones):
    all_paths = []

    for i in range(n_ants):
        path = genPath(distances, pheromones)

        all_paths.append((path, genPathDist(path, distances)))

    return all_paths

def spreadPheromones(all_paths, distances, pheromones):
    for path, dist in all_paths:
        for move in path:
            if (distances[move[0]][move[1]] > 0):
                pheromones[move[0]][move[1]] += 1.0 / distances[move[0]][move[1]]

def runAntColony(decay, n_iterations, n_ants, distances, pheromones):
    shortest_path = None
    all_time_shortest_path = None

    for i in range(n_iterations):
        all_paths = genAllPaths(n_ants, distances, pheromones)

        spreadPheromones(all_paths, distances, pheromones)

        shortest_path = min(all_paths, key = lambda x: x[1])

        if (all_time_shortest_path == None or shortest_path[1] < all_time_shortest_path[1]):
            all_time_shortest_path = shortest_path

        # Decay all
        for i in range(len(pheromones)):
            for j in range(len(pheromones)):
                pheromones[i][j] *= decay

    return shortest_path

def solveAntColonyOptimization(points):
    # Init matrix of distances
    distances = [ [ -1 for i in range(len(points)) ] for j in range(len(points)) ]
    for point1 in points:
        for point2 in points:
            if (point1 != point2):
                distances[point1[0] - 1][point2[0] - 1] = distance(point1, point2)

    # Init values for the algorithm
    n_ants = len(points)
    n_iterations = int(len(points) / 2) + 5

    value = 1 / len(points)
    pheromones = [ [ value for i in range(len(points)) ] for j in range(len(points)) ]
    decay = 1

    # Get the shortest path
    shortestPath = runAntColony(decay, n_iterations, n_ants, distances, pheromones)

    # Get results
    solution = ""
    total = 0
    for point in shortestPath[0]:
        total += distances[point[0]][point[1]]
        solution += str(point[0] + 1) + "-"
    solution += "1"

    return (solution, total)

# ---
# MAIN FUNCTIONS
# ---

'''
Algorithm choice menu
'''
def menu():
    print()
    print("Hi !")
    print()
    print("There are multiple algorithms available, please choose one:")
    print("1. Permutation")
    print("2. Nearest")
    print("3. Nearest improved")
    print("4. Nearest insertion")
    print("5. Farthest insertion")
    print("6. Random node insertion")
    print("7. Ant colony optimization")
    print()

    try:
        choice = int(input("Choice: "))
        if (choice <= 0 and choice > 6):
            choice = 1
    except:
        choice = 0

    print()
    return choice

'''
Return an array of (number, x, y) tuples
'''
def parse(filename):
    points = []

    file = open(filename, 'r+')

    reader = csv.reader(file, delimiter = ',')
    i = 1
    for row in reader:
        if (len(row) > 1):
            points.append((i, float(row[0]), float(row[1])))
            i += 1

    return points

# ---
# MAIN
# ---

def main():
    if (len(sys.argv) < 2):
        print('ERROR: Filename is missing')
        return
    elif (not os.path.isfile(sys.argv[1])):
        print('ERROR: File not found')
        return
    else:
        filename = sys.argv[1]
        a = filename.split(".")
        if (len(a) < 2 or a[1] != "csv"):
            print('ERROR: Wrong file, must be a CSV !')
            return

    choice = menu()

    while (choice != 0):
        points = parse(filename)

        start = time.time()

        if (choice == 1):
            print("Calculate permutation...")
            (solution, total) = solvePermutation(points)
        elif (choice == 2):
            print("Calculate nearest...")
            (solution, total) = solveNearest(points)
        elif (choice == 3):
            print("Calculate nearest improved...")
            (solution, total) = solveNearestImproved(points)
        elif (choice == 4):
            print("Calculate nearest insertion...")
            (solution, total) = solveNearestInsertion(points)
        elif (choice == 5):
            print("Calculate farthest insertion...")
            (solution, total) = solveFarthestInsertion(points)
        elif (choice == 6):
            print("Calculate random node insertion...")
            (solution, total) = solveRandomNodeInsertion(points)
        else:
            print("Calculate ant colony optimization...")
            (solution, total) = solveAntColonyOptimization(points)

        end = time.time()

        print()
        print('solution: ' + solution)
        print('distance:', total)
        print('time:', end - start, 'secondes')

        if (os.path.isfile("library/plot.py")):
            os.system("python3 library/plot.py " + filename + ' ' + solution)

        choice = menu()

# ---
# LAUNCH
# ---

main()