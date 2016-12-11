import numpy as np
import theano
import theano.tensor as T

n = 10
m = 4
p = 3
q = 2
x = T.fmatrix("x")  # shape = n, m
W = theano.shared(np.random.random((m, p)).astype(np.float32), name="W")
V = theano.shared(np.random.random((p, m, m)).astype(np.float32), name="V")
b = theano.shared(4 * np.ones((p,)).astype(np.float32), name="b")

z1 = T.tensordot(x, V, axes=[[1], [2]])  # n,m + p,m.m = n,p,m
z2 = T.batched_tensordot(x, z1, axes=[[1], [2]])  # n,m + n,p,m = n,p
y = T.dot(x, W) + b.dimshuffle(['x',0]) +z2
f = theano.function([x], outputs=y)

_x = np.random.random((n, m)).astype(np.float32)
_y = f(_x)

print "X"
print _x
print "Y"
print _y

alpha = theano.shared(np.float32(1e-3), name="alpha")
beta = theano.shared(np.float32(1), name="beta")

Q = theano.shared(np.random.random((m, p, q)).astype(np.float32), name="Q") #p,m,q
V2 = T.batched_tensordot(Q, Q, axes=[[2],[2]]) # p,m,q+p,m,q = p,m,m
V3 = beta*(T.eye(m,m,0, dtype='float32').dimshuffle(['x',0,1])*alpha+V2)
z1 = T.tensordot(x, V3, axes=[[1], [2]])  # n,m + p,m.m = n,p,m
z2 = T.batched_tensordot(x, z1, axes=[[1], [2]])  # n,m + n,p,m = n,p
y = T.dot(x, W) + b.dimshuffle(['x',0]) +z2
f = theano.function([x], outputs=y)

print "X2"
print _x
print "Y2"
print _y


#V2 = T.tensordot(V, V, axes=[[1,2],[1,0]]) # m,p,m + m,p,m = m,p,m

_A = np.random.uniform(-1,1,(5,2)).astype(np.float32)
print np.dot(_A, np.transpose(_A))

f=theano.function([x], outputs=T.dot(x, x.dimshuffle([1,0])))
print f(_A)

f=theano.function([x], outputs=T.tensordot(x, x, axes=[[1],[1]]))
print f(_A)

#fz = theano.function([x], outputs=z2)
#print fz(_x).shape
