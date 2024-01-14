from matplotlib import pyplot as plt
import csv
import numpy

data = [[int(x) for x in row] for row in csv.reader(open("all_moves.csv"))]

data = data[0:]

BIN_SIZE = 5
NUM_BINS = 21

data_hist = numpy.array([[sum(i*BIN_SIZE <= x < (i+1)*BIN_SIZE for x in row)
               for i in range(NUM_BINS)] for row in data])

data_hist = data_hist.T

ax = plt.axes()

print(data_hist)
plt.imshow(data_hist, cmap="Reds", origin='lower')

plt.savefig("plot_hist.png")