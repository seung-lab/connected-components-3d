

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
