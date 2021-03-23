import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("plots/tt_test.txt", header=None)
plt.plot(data.index, data[1], 'y*-')
plt.plot(data.index, data[2], 'r--')
plt.plot(data.index, data[3], 'm--')
plt.show()