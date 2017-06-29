# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt

x = [10, 20, 30, 40, 50, 60, 70, 80]
gsp = [3, 2, 16, 44, 63, 1091, 6253, 90089]
spade = [1, 1, 1, 1, 1, 1, 2, 2]

plt.figure(figsize=(8, 4))
plt.plot(x, gsp, label="gsp", color="red", linewidth=2)
plt.plot(x, spade, color='blue', label="spade")
plt.xlabel("number of data")
plt.ylabel("time(ms)")
plt.title("gsp and spade")
plt.legend()
plt.show()