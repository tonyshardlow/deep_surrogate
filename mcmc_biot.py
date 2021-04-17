"""
Perform MCMC using multiple chains in parallel to infer posterior distribution of Biot number
using a deep surrogate model to accelerate the evaluation of the likelihood function
"""

import numpy as np
import tensorflow as tf
from dense_net import *
from matplotlib import pyplot as plt
from scipy.stats import uniform
import arviz as az

np.random.seed(0)
tf.set_random_seed(0)

n=16 #number of polynomial terms


# load pre-trained deep surrogate model for MCMC
no_layers = 4
nodes_per_layer = 45
layers   = [n+1] + no_layers*[nodes_per_layer] + [1]
net = dense_net(layers)
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, 'biot')




#read in data
n_dat  = 30
x_dat_read = np.genfromtxt('x_dat.csv', delimiter=',').reshape([-1,1]).astype('float32')[0:30] 
y_dat_read = np.genfromtxt('y_dat.csv', delimiter=',').reshape([-1,1]).astype('float32')[0:30]



n_chains = 5 # number of chains to run in parallel


# Define the domain of the prior distribution (this is the same as the solution domain of the network)
coefflim = [-5.,20]
leading_coef_dom = np.array([1,1,2,4])
scale = np.hstack([leading_coef_dom,np.cumprod(2*np.ones(n-leading_coef_dom.shape[0]))*2])

coeff_dif = coefflim[1]-coefflim[0]
coef_shift = np.abs(coefflim[0])


# placeholder to pass parameters proposed by MCMC
coeff_pl  = tf.placeholder(dtype = "float32", shape = [n_dat*n_chains,n])
coeffhist_pl  = tf.placeholder(dtype = "float32", shape = [n_chains,])
coeff_forplot = tf.placeholder(dtype = "float32", shape = [100,n])




# stack n_chains copies of the data into one array
x_dat = x_dat_read
y_dat = y_dat_read
for i in range(1,n_chains):
    x_dat = np.vstack([x_dat,x_dat_read])
    y_dat = np.vstack([y_dat,y_dat_read])

# evaluate stacked data for MCMC parameters
mean_pl  = net.evaluate(tf.concat([x_dat,coeff_pl],axis=1)) 


def prior_coef(coef): #function to evaluate prior pdf of given parameters
    pri = np.sum(np.log(coeff_dif/scale))*np.ones([n_chains])
    for j in range(n_chains):
        if np.amax(abs((coef[j]*scale/coeff_dif) + coef_shift/coeff_dif - 0.5 ))>0.5:
            pri[j] = np.log(0)
    return pri

def prior_sig(sig): #function to evaluate prior pdf for sigma^2
    return np.log(uniform(loc=0.,scale=1).pdf(sig))

#function to calculate log likelihood of the data for each markov chain
def log_likelihood(coef,sig,x_dat,y_dat): 
    coefstack = coef[0,:]*np.ones([n_dat,1])
    sigstack = sig[0]*np.ones([n_dat,1])
    for i in range(1,n_chains):
        coefstack = np.vstack([coefstack,coef[i,:]*np.ones([n_dat,1])])
        sigstack = np.vstack([sigstack,sig[i]*np.ones([n_dat,1])])
    coef = coefstack
    sig = sigstack
    mean = sess.run(mean_pl,{coeff_pl:coef})
    log_lik = np.zeros([n_chains])
    for i in range(n_chains):
        log_lik[i] = np.sum(-np.log(np.sqrt(2* np.pi*sig[n_dat*i:n_dat*i+n_dat])) - ((y_dat[n_dat*i:n_dat*i+n_dat]-mean[n_dat*i:n_dat*i+n_dat])**2) / (2*sig[n_dat*i:n_dat*i+n_dat]**2))
    return log_lik



#function to generate metropolis hastings proposal for each chain
def propose_point(coef,sig):
    coef_1 = coef + np.random.normal(0,0.8,[n_chains,n])/scale
    sig_1 = sig + np.random.normal(0,0.0005,[n_chains])
    return coef_1, sig_1

#function to decide whether to accept or reject for each chain
def accept(coef,coef_1,sig,sig_1,x_dat,y_dat):
    out = np.zeros([n_chains])
    lik = log_likelihood(coef,sig,x_dat,y_dat)
    lik_1 = log_likelihood(coef_1,sig_1,x_dat,y_dat)
    log_post0 = prior_coef(coef) + prior_sig(sig) + lik
    log_post1 = prior_coef(coef_1) + prior_sig(sig_1) + lik_1
    check = log_post1>log_post0
    for i in range(n_chains):
        if check[i]:
            out[i] = True
        else:
            accept_p = np.random.uniform(0,1)
            out[i] = ( accept_p < np.exp(log_post1[i]-log_post0[i]) )
    return out


# set the initial state of each markov chain to Bi=0 and sigma=0.2
sig_init = np.ones([n_chains])*0.2
coef_init = np.zeros([n_chains,n])



iterations = 100000 
n_samples = iterations*n_chains

#initialise the chains to the initial state
coef = coef_init
sig = sig_init

#lists to store posterior samples
coef_accepted = []
coef_sample = []   
sig_accepted = []
sig_sample = []   


#variables to help with displaying acceptance rate
old_accepted = 0
new_accepted = 0


#main MCMC routine. Applies metropolis hastings using multiple markov chains in 
#parallel to sample from the posterior distribution of the parameter space of Bi(x|\theta).
for i in range(1,iterations):
    coef_1, sig_1 =  propose_point(coef,sig) # sample proposal points
    if i % 100 == 99:
        print('iteration: ', i+1 , '.  acceptance rate from last 100 = ', (len(np.array(coef_accepted)[:,1])-old_accepted)/(100*n_chains))
        old_accepted = len(np.array(coef_accepted)[:,1])
    update = accept(coef,coef_1,sig,sig_1,x_dat,y_dat) #decide whether to accept proposals
    for j in range(n_chains):
        if update[j]:
            coef[j,:] = coef_1[j,:]
            sig[j] = sig_1[j]
            coef_accepted.append(coef_1[j,:])
            sig_accepted.append(sig_1[j])
            coef_sample.append(coef_1[j,:])
            sig_sample.append(sig_1[j])
        else:
            coef_sample.append(coef[j,:])
            sig_sample.append(sig[j])


# delete first half of the samples due to the burn in period
coef_exburnin = np.array(coef_sample[round(n_samples/2):])
sig_exburnin = np.array(sig_accepted[round(n_samples/2):])



###########################################################################################################
# Section below creates output posterior plots 
###########################################################################################################

xx = np.linspace(0.3,1,100).astype(np.float32)
true = 18*np.exp(xx-0.3)    
funsamples = np.zeros([coef_exburnin.shape[0],100])
funsamples = sum(coef_exburnin[:,i].reshape([-1,1])*xx**i for i in range(n))
funmean = np.mean(funsamples,axis = 0)
funci0  = np.quantile(funsamples, 0.025,axis=0)
funci1  = np.quantile(funsamples, 0.975,axis=0)
coefmean = (np.mean(coef_exburnin,axis=0)*np.ones([100,1])).astype(np.float32)
solcoefmean = sess.run(net.evaluate(np.hstack([xx.reshape([-1,1]),coefmean]))).reshape([-1])



fig, axs = plt.subplots(1, 2)


axs[1].plot(xx,funmean,label=r"Inferred $\tilde{Bi}(x|\bf{\theta})$", color='royalblue')
axs[1].plot(xx,true,label=r"True $Bi(x)$", color='tomato')
axs[1].fill_between(xx, funci1, funci0, facecolor='blue', alpha=0.3)
axs[1].legend(bbox_to_anchor=(0.05, 0.95), loc='upper left', borderaxespad=0.)


plt.plot(xx,funmean,label='Posterior Mean', color='royalblue')

axs[0].scatter(x_dat,y_dat,s=6.4,alpha = 1, label = 'Data points', color='tomato')
axs[0].plot(xx,solcoefmean, label = r"Fitted $u(x)$", color='royalblue')
axs[0].legend(bbox_to_anchor=(0.05, 0.95), loc='upper left', borderaxespad=0.)


fig.set_size_inches(11, 5)

fig.subplots_adjust(top=0.985,
bottom=0.155,
left=0.07,
right=0.96,
hspace=0.305,
wspace=0.2)

axs[0].set_xlabel(r"$x$")
axs[0].set_ylabel(r"$u(x)$")

axs[1].set_xlabel(r"$x$")
axs[1].set_ylabel(r"$Bi(x)$")


'''
###########################################################################################################
# Section below visualises the prior distribution and shows the true Bi
###########################################################################################################


exp18 = np.exp(xx-0.3)*18

pri_coef  = np.random.uniform(coefflim[0],coefflim[1],size=[1000,n])/scale
pri_coef = pri_coef.astype(np.float32)
pri_bi_samples = sum(pri_coef[:,i]*xx.reshape([-1,1])**i for i in range(n))

fig = plt.figure()
#plt.plot(xx,exact)
plt.plot(xx,pri_bi_samples,alpha=0.03,color='g')
plt.plot(xx,exp18,alpha=0.5, label = 'prior samples',color='g')
plt.plot(xx,exp18, label = 'True Bi')
fig.legend(bbox_to_anchor=(0.15, 0.85), loc='upper left', borderaxespad=0.)
plt.xlabel(r"$x$")
plt.ylabel(r"$Bi(x)$")


plt.show()
'''
