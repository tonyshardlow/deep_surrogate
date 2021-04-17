"""
Train a deep surrogate model to approximate the parametric solution of the Fin equation 
u''(x) + u'(x)/x - Bi(x)u(x) = 0 
over a domain where Bi(x) is represented by a polynomial of degree up to 15 with 
coefficients lying within a suitable domain
"""

import numpy as np
import tensorflow as tf
from dense_net import *
from matplotlib import pyplot as plt

np.random.seed(0)
tf.set_random_seed(0)


def train(iterations,learning_rate): # main training loop
    for i in np.arange(iterations):
        xl = sample_xl(xlim,sizel)
        xr = sample_xr(xlim,sizer)
        x = sample_x(xlim,sizex)
        coeff, coeffl, coeffr = sample_coeff(coefflim,sizex,sizel)
        tf_dict = {lr:learning_rate , xl_pl:xl , x_pl:x , xr_pl:xr , \
                   coeffl_pl:coeffl, coeff_pl:coeff, coeffr_pl:coeffr}
        sess.run(train_op,tf_dict)
        if i % 10 ==0:
            loss_eval = sess.run(printout,tf_dict)
            print('iteration: ',i, '.  learning rate: ', learning_rate,\
                          '.  loss: ', loss_eval)
 
def sample_x(xlim,sizex):  #function to sample from domain
    x = np.random.uniform(xlim[0],xlim[1],size=[sizex,1])
    return x

def sample_xl(xlim,sizel): #function to sample from left boundary
    xl = np.ones([sizel,1])*xlim[0]
    return xl

def sample_xr(xlim,sizer): #function to sample from right boundary
    xr = np.ones([sizer,1])*xlim[1]
    return xr


def sample_coeff(coefflim,sizex,sizel):  #function to sample from parameter space
    coeff  = np.random.uniform(coefflim[0],coefflim[1],size=[sizex,n])/scale
    coeffl = coeff[:sizel,:]
    coeffr = coeff[:sizer,:]
    return coeff, coeffl, coeffr




sizel = 500 # no. of samples from left boundary per iteration
sizer = 500 # no. of samples from right boundary
sizex = 6000 # no. of samples inside domain
n=16 # no. of polynomial terms in Bi(x|\theta)
      

# Define placeholders to feed sampled training points into
lr   = tf.placeholder(dtype = "float32", shape = [])
loss_pl   = tf.placeholder(dtype = "float32", shape = [])

xl_pl  = tf.placeholder(dtype = "float32", shape = [sizel,1])
coeffl_pl  = tf.placeholder(dtype = "float32", shape = [sizel,n])

x_pl  = tf.placeholder(dtype = "float32", shape = [sizex,1])
coeff_pl  = tf.placeholder(dtype = "float32", shape = [sizex,n])

xr_pl  = tf.placeholder(dtype = "float32", shape = [sizer,1])
coeffr_pl  = tf.placeholder(dtype = "float32", shape = [sizer,n])

X = tf.concat([x_pl,coeff_pl],axis=1)
Xl = tf.concat([xl_pl,coeffl_pl],axis=1)
Xr = tf.concat([xr_pl,coeffr_pl],axis=1)


# set up neural network to approximate the solution
no_layers = 4
nodes_per_layer = 45
layers   = [n+1] + no_layers*[nodes_per_layer] + [1]
net = dense_net(layers)


# solution domain
xlim = [0.3,1.]

# parameter domain - 'scale' is a vector used to scale the solution domain of 
# the coefficients. The solution domain for coefficient i is coefflim/scale[i]  
coefflim = [-5.,20]
leading_coef_dom = np.array([1,1,2,4])
scale = np.hstack([leading_coef_dom,np.cumprod(2*np.ones(n-leading_coef_dom.shape[0]))*2])


#evaluate differential operator terms on sample points
u  = net.evaluate(X)
u_x = tf.gradients(u,x_pl)[0]
u_xx = tf.gradients(u_x,x_pl)[0]

#evaluate network at boundary value sample points
ul = net.evaluate(Xl)
ur = net.evaluate(Xr)

# define polynomial approximation of Biot number 
Bi  = sum(tf.reshape(coeff_pl[:,i],[-1,1])*x_pl**i for i in range(n))

# construct loss function
pinn = (u_xx + u_x/x_pl - Bi*u)**2
lbc = (ul-0.2)**2
rbc = (ur-1)**2

loss = tf.reduce_mean(pinn) + 10*tf.reduce_mean(lbc) + 10*tf.reduce_mean(rbc)


# prints total value of loss function to screen, as well as individual loss terms
printout  = [loss,tf.reduce_mean(pinn), tf.reduce_mean(lbc), tf.reduce_mean(rbc)]

# get gradients of loss and clip by global norm
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
gradients, variables = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 2.)
train_op = optimizer.apply_gradients(zip(gradients, variables))



# start tf session and train network
sess = tf.Session()
sess.run(tf.global_variables_initializer())

train(iterations = 30000,learning_rate = 0.001)
train(iterations = 150000, learning_rate = 0.0001)
train(iterations = 300000, learning_rate = 0.00001)
train(iterations = 20000, learning_rate = 0.000001)


###########################################################################################################
# Section below creates solution plots
###########################################################################################################

#Run this section plot solutions for a specified Bi(x|\theta)  (default Bi=10)
#Bi(x|\theta) can be specified through its polynomial coefficients as a list
#'leading_coefs' of up to 16 values, if less than 16 coefficients are given 
#then any higher degree terms are automatically set to have coefficient zero

leading_coefs = [10] #<<--set polynomial coefficients of Bi(x|\theta) here


n_x = 1000
xlin = np.linspace(xlim[0],xlim[1],n_x)
xx = xlin.reshape(-1,1).astype(np.float32)
leading_coefs = np.array(leading_coefs)
coef = (np.hstack([leading_coefs,np.zeros(n-leading_coefs.shape[0])])*np.ones([n_x,n])).astype(np.float32)
yy = sess.run(net.evaluate(np.hstack([xx,coef])))

fig = plt.figure()
plt.plot(xx,yy)
plt.xlabel(r"$x$")
plt.ylabel(r"$u(x)$")
plt.show()






