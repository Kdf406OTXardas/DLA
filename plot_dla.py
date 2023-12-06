import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.pyplot import plot, show, draw, figure, cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EDGE_FIELD = 60
END_POROSITY = 3

df = pd.read_csv(f"/home/natkachov/datasets/export_from_dla_cuda/dla_cuda_{EDGE_FIELD}_{100 - END_POROSITY}.csv")
# df.reset_index(drop=True)
print(df)
df.head()
df['status_point'] = df['property'].astype(int)
df = df[df['property'] == 1.0]
print(df)
print(len(df))
df.head()
x = df['x']
y = df['y']
z = df['z']

df_3d = df[['x', 'y', 'z', 'status_point']]
color_points = df['status_point']

# Create a 3D scatter plot with Seaborn
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=color_points, cmap='viridis', marker='o')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.title('DLA')
plt.show()

sns.pairplot(df_3d, hue='status_point')
plt.show()
