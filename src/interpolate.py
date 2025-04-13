import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ball_positions.csv')
print(df.head())

df_interpolated = df.interpolate(method='polynomial', order= 2)
df_interpolated.to_csv("interpolated_ball_positions.csv", index=False)
print(df_interpolated.head(10))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(df_interpolated['X'], df_interpolated['Y'], df_interpolated['Z'], marker='o', linestyle='-', color='b', label='Interpolated Path')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('3D Ball Path')
plt.legend()
plt.show()




