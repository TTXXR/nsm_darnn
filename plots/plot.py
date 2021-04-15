import matplotlib.pyplot as plt
import pandas as pd

name = "2layergru"
data = pd.read_csv(name+".txt", header=None)
plt.plot(data.index, data[1], 'y*-')
plt.plot(data.index, data[2], 'r--')
# plt.plot(data.index, data[3], 'm--')
plt.savefig(name+"_plot.png")
plt.show()