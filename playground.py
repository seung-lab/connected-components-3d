import numpy as np

print(np.int64(10).itemsize)

# x = [0,1,2,3,4,-5]
# b = list(filter(lambda a: a > 0, x))
# print(b)

# a = np.ones((3,3))
# a[2,2] = -2
# a[2,1]= -3
# print(a)
#
# a[a<0] = a[a<0] - 20
# print(a)
# b=a>1
# a[b]=0
# print(a)
# labels_filled = np.ones((10,10,10))
# labels = np.zeros((10,10,10))
#
# new = np.zeros(labels.shape)
#
# print(new.shape)
#
# wholes = np.subtract(labels_filled,labels)

# print(wholes)

# print("max_label is: " + str(np.max(wholes)))
# print("min_label is: " + str(np.min(wholes)))

#
# associated_comp = {}
#
# associated_comp[1]=2
# associated_comp[5]=2
# associated_comp[8]=1
# associated_comp[9]=0
# associated_comp[2]=0
#
# print(associated_comp)
# filtered = {k: v for k, v in associated_comp.items() if v != 0}
# print(filtered)
# # import sys
#
# a = np.uint8(5)
# b = np.uint16(5)
#
# print(sizeof a)
# print(sys.getsizeof(b))

#
# import numpy as np
# x = np.linspace(1,100,100)
#
# print(x[::2])
# print(x[::3])
# print(x[10:40:6])


# arr = [np.zeros((),dtype=np.uint8) for _ in range(5)]
#
# arr[0] = np.append(arr[0],5)
# arr[1] = np.append(arr[1],6)
# arr[1] = np.append(arr[1],7)
# print(arr[0])
# print(arr[1])
# print(arr)

# adj_comp = np.ones((3,1))*-2
# test.append(1)
# print(test)
# test.append(2)

# import numpy as np
#
# a = np.array([[1, 2],
#            [3, 4]])
# b = np.array([1,1])
#
# print("a: " + str(a))
# print("b: " + str(b))
#
# print(str(tuple(b)))
# print(str(a[b]))


# a = np.repeat(2,6)
# print(a)
#
# print(np.max(np.absolute([5,0])))

# from scipy.spatial import ConvexHull, convex_hull_plot_2d
# points = np.random.rand(30, 2)   # 30 random points in 2-D
# hull = ConvexHull(points)
#
# import matplotlib.pyplot as plt
# plt.plot(points[:,0], points[:,1], 'o')
# print(points)
# print("Simplices: ")
# print(hull.simplices)
# for simplex in hull.simplices:
#     plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
#
# plt.show()
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# def randrange(n, vmin, vmax):
#     '''
#     Helper function to make an array of random numbers having shape (n, )
#     with each number distributed Uniform(vmin, vmax).
#     '''
#     return (vmax - vmin)*np.random.rand(n) + vmin
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# n = 100
#
# # For each set of style and range settings, plot n random points in the box
# # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
#     xs = randrange(n, 23, 32)
#     ys = randrange(n, 0, 100)
#     zs = randrange(n, zlow, zhigh)
#     ax.scatter(xs, ys, zs, c=c, marker=m)
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# plt.show()
