import pandas as pd
import os

# df = pd.read_csv("/home/rr/Downloads/nsm_data/Train/Input/0.txt", sep=' ',header=None)
# print(df)
# print(df.shape) # 100 * 5307
# print(len(df.columns)) # 100 * 5307
# a=(1,)
# print(type(a))

# df = pd.read_csv("/home/rr/Downloads/nsm_data/train.txt", header=None)
# print(df)
# print(df.shape) # 100 * 618


# root_path = "/home/rr/Downloads/nsm_data/train.txt"
# df = pd.read_csv("/home/rr/Downloads/nsm_data/train.txt",index_col=0)
# print(df.shape)
# df.to_csv("input.txt",header=None,index=None)

input_data = pd.DataFrame()
label_data = pd.DataFrame()

root_path = "/home/rr/Downloads/nsm_data/"

path_list = os.listdir(root_path + "Train/Input/")
path_list.sort(key=lambda x: int(x[:-4]))
n = 0
for f in path_list:
    print(f)
    data1 = pd.read_csv(root_path + "Train/Input/" + f, sep=' ', header=None)
    input_data = input_data.append(data1, ignore_index=True)

    data2 = pd.read_csv(root_path + "Train/Label/" + f, sep=' ', header=None)
    label_data = label_data.append(data2, ignore_index=True)
    n = n + 1

    # data = pd.concat([input_data, label_data], axis=1)
    # data.to_csv(root_path + "Raw/"+f)

    if n % 50 == 0:
        input_data.to_csv(root_path + str(n) + "input.txt", header=None, index=None)
        label_data.to_csv(root_path + str(n) + "label.txt", header=None, index=None)

    if n > 1500:
        break
#
# input_data.to_csv(root_path+"Raw/",header=None,index=None)
# label_data.to_csv(root_path+"label.txt",header=None,index=None)

# print(input_data.shape)
