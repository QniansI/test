# # 数据操作
import tensorflow as tf
import numpy as np
# x = tf.constant(range(12))
# reshape函数把行向量x的形状改为(3, 4)
# X = tf.reshape(x, (3, 4))
# print(X)
# 各元素为0，形状为(2, 3, 4)的张量
# tf.zeros((2,3,4))
# 创建各元素为1的张量
# y = tf.ones((2, 3, 4))
# print(y)
# 通过Python的列表（list）指定需要创建的tensor中每个元素的值
# Y = tf.constant([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
# print(Y)
# 随机生成tensor中每个元素的值。形状为(3, 4)的tensor。它的每个元素都随机采样于均值为0、标准差为1的正态分布
# z = tf.random.normal(shape=[3, 4], mean=0, stddev=1)
# print(z)
# 元素加法
# x = tf.constant(range(12))
# X = tf.reshape(x, (3, 4))
# Y = tf.constant([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
# print(X)
# print(Y)
# print(X+Y)
# 元素除法 X/Y,元素乘法 X*Y
# 元素做指数运算
# tf.cast()函数的作用是执行 tensorflow 中张量数据类型转换
# Y = tf.constant([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
# Y = tf.cast(Y, tf.float32)
# print(Y)
# # tf.exp()计算e的x次方
# print(tf.exp(Y))
# 使用matmul函数做矩阵乘法。
# 下面将X与Y的转置做矩阵乘法。由于X是3行4列的矩阵，Y转置为4行3列的矩阵，
# 因此两个矩阵相乘得到3行3列的矩阵
# x = tf.constant(range(12))
# X = tf.reshape(x, (3, 4))
# Y = tf.constant([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
# print(tf.matmul(X, tf.transpose(Y)))
# concat()在行上（维度0，即形状中的最左边元素）和列上（维度1，即形状中左起第二个元素）连结两个矩阵
# print(tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1))
# 使用条件判断式可以得到元素为0或1的新的tensor
# print(tf.equal(X, Y))
# reduce_sum()对tensor中的所有元素求和得到只有一个元素的tensor
# print(tf.reduce_sum(X))
# X = tf.cast(X, tf.float32)
# print(tf.norm(X))
# 广播机制
# 当对两个形状不同的tensor按元素运算时，可能会触发广播（broadcasting）机制：
# 先适当复制元素使这两个tensor形状相同后再按元素运算
# A = tf.reshape(tf.constant(range(3)), (3, 1))
# B = tf.reshape(tf.constant(range(2)), (1, 2))
# print(A, B)
# print(A+B)
# 索引
# tensor的索引从0开始逐一递增
# 依据左闭右开指定范围的惯例，它截取了矩阵X中行索引为1和2的两行
# x = tf.constant(range(12))
# X = tf.reshape(x, (3, 4))
#print(X)
# # 行索引截取
# print(X[1:3])
# # 列索引截取
# print(X[:, 1:3])
# tf.Variable(X)变量初始化
# X = tf.Variable(X)
# # 为该元素重新赋值
# print(X)
# print(X[1, 2].assign(9))
# # 截取一部分元素，并为它们重新赋值。在下面的例子中，我们为行索引为1的每一列元素重新赋值
# # print(tf.ones(X[1:2, :].shape, dtype=tf.int32)*12)
# print(X[1:2, :].assign(tf.ones(X[1:2, :].shape, dtype=tf.int32)*12))
# 运算的内存开销
# Y = tf.constant([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
# Y = tf.Variable(Y)
# X = tf.Variable(X)
# before = id(Y)
# # Y = Y + X
# # print(id(Y) == before)
# # 指定结果到特定内存，我们可以使用前面介绍的索引来进行替换操作。
# # 先通过zeros_like创建和Y形状相同且元素为0的tensor，记为Z。
# # 接下来，我们把X + Y的结果通过[:]写进Z对应的内存中
# # Z = tf.Variable(tf.zeros_like(Y))
# # before = id(Y)
# # Y[:].assign(X + Y)
# # print(id(Y) == before)
# # 还是为X + Y开了临时内存来存储计算结果
# Y.assign_add(X)
# print(id(Y) == before)
# tensor 和 NumPy 相互变换
# P = np.ones((2, 3), dtype=int)
# # NumPy实例变换成tensor实例。
# D = tf.constant(P)
# print(D)
# # 再将NDArray实例变换成NumPy实例
# np.array(D)
# print(np.array(D))