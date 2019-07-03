import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import re

def plot():
    #Parse file
    x = []
    y = []
    size = 0
    with open(filename, 'r+') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        for row in reader:
            if (len(row) > 1):
                x.append(float(row[0]))
                y.append(float(row[1]))
            else:
                size = int(row[0])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    #Plot the points
    ax.scatter(x, y, c = '#000000', alpha = 0.5)

    for i in range(size):
        ax.annotate(str(i + 1), (x[i], y[i]))

    #Add solution lines to the plot
    currX = x[0]
    currY = y[0]

    for point in solution.split('-'):
        nextX = x[int(point) - 1]
        nextY = y[int(point) - 1]
        ax.plot([currX,nextX], [currY,nextY], 'r-', lw = 2)
        currX = nextX
        currY = nextY

    #Show plot
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_aspect('equal')
    plt.show()

def usage():
    print("Mauvais nombre d'argument:")
    print("usage: python pvcPlot.py <data.csv> <1-x-x-x-x-1>")

def wrongSol():
    print("Solution non acceptee:")
    print("La solution doit etre au format 1-x-x-x-x-1")

#Regex solution pattern
pattern = re.compile("^1\-(\d+\-)+1$")

#Arguments control
if(len(sys.argv) != 3):
    print(len(sys.argv))
    usage()
else:
    filename = sys.argv[1]
    solution = sys.argv[2]
    if(pattern.match(solution)):
        plot()
    else:
        wrongSol()
