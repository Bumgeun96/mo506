import matplotlib.pyplot as plt
import matplotlib.patches as p
import numpy as np
import pickle

center_x = []
center_y = []
hor = []
ver = []
reward = []
with open("./pickle/reward.pickle",'rb') as fr:
    for _ in range(100000):
        try:
            data = pickle.load(fr)
            center_x.append(data[0])
            center_y.append(data[1])
            hor.append(data[2])
            ver.append(data[3])
            reward.append(data[4])
        except EOFError:
            break
img_color = plt.imread('./ex/ex1.jpg')
center_x = np.array(center_x[1:])
center_y = np.array(center_y[1:])
reward = np.array(reward[1:])
# print(center_y)
fig,ax = plt.subplots(figsize=(10,10))

# ax.tick_params(left=True,labelleft=True,
#                right=False,labelright=False,
#                bottom=False,labelbottom=False,
#                top=True, labeltop=True)
# plt.imshow(img_color)
# plt.plot(center_x,center_y)
plt.plot(reward)
# rec = p.Rectangle((center_x[-1]-hor[-1]/2,center_y[-1]-ver[-1]/2), hor[-1],ver[-1],fill=False)
# plt.gca().add_patch(rec)
plt.show()