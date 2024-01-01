from matplotlib import pyplot as plt
import csv

data = [[int(x) for x in row] for row in csv.reader(open("all_moves.csv"))]

ax = plt.axes()

plt.violinplot(data, showmeans=True)
plt.savefig("plot.png")