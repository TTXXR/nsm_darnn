import pandas as pd
import os
import numpy as np
# df = pd.read_csv("/home/rr/Downloads/nsm_data/Train/Input/0.txt", sep=' ',header=None)
# print(df)
# print(df.shape) # 100 * 5307
# print(len(df.columns)) # 100 * 5307
# a=(1,)
# print(type(a))

# df = pd.read_csv("/home/rr/Downloads/nsm_data/Train/Label/0.txt", sep=' ',header=None)
# print(df)
# print(df.shape) # 100 * 618
# a = np.array([1,2,3])
# print(a.shape[0])
# a = [0,0,0]
a = np.zeros((3,3))
b = [1 for i in a if i.all()!=0]
print(type(b))