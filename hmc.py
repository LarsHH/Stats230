import numpy as np
import scipy.stats
import scipy.linalg
import time
import random
import cPickle
from load import mnist
from scipy import stats


# TO DO
# wrap parameters into list to make more parsimonious

def autocorr(x, t=1):
    return np.corrcoef(np.array([x[0:len(x)-t], x[t:len(x)]]))

def sigmoid(X):
    return 1/(1 + np.exp(-X))

def decoder(z, w2, b2, w3, b3):
    h2 = np.tanh(np.dot(z, w2) + b2)
    y = sigmoid(np.dot(h2, w3) + b3)
    return y

#### Log posterior density
def log_posterior(z, X, w3, b3, w2, b2):
    # Log prior z ~ N(0, I)
    #log_p_z = -0.5 * np.dot(np.dot((z - mu_z).T, np.diag(1./sigma_z)), (z - mu_z))
    log_p_z = - 0.5 * np.dot(z.T, z)

    # Decoder from Auto-encoder i.e. gets the probabilities in the Bernoulli likelihood
    y = decoder(z, w2, b2, w3, b3)

    # Log Bernoulli likelihood
    log_p_x_z = np.sum(X * np.log(y) + (1-X) * np.log(1-y))

    # Add to get the posterior
    log_p_z_x = log_p_z + log_p_x_z
    return log_p_z_x


#### Hamiltonian
def hamiltonian(z, X, w3, b3, w2, b2, p):
    return - log_posterior(z, X, w3, b3, w2, b2) + 0.5 * np.dot(p.T, p)


#### Force
def force(z, X, w3, b3, w2, b2):
    y = decoder(z, w2, b2, w3, b3) # 784x1
    h2 = np.tanh(np.dot(z, w2) + b2) # 500x1
    #y * (1-y) # (784,)
    #(1-h2**2) # (500,)
    dy_dz = np.atleast_2d(y * (1-y)).T * np.dot(w3.T, np.atleast_2d(1-h2**2).T * w2.T)

    #dy_dz = y * (1-y) * np.dot(w3.T , (1-h2**2) * w2.T # 784 x dim_z , w3=500x784, w2=dim_zx500
#   784xdim_z = 781x1 784x1 784x500 500x1 500xdim_z
    dlog_p_x_z_dz = np.sum(np.atleast_2d((X - y)/(y * (1-y))).T * dy_dz, axis=0)
    dlog_p_z_dz = z
    dlog_p_z_x_dz = dlog_p_x_z_dz + dlog_p_z_dz
    return -dlog_p_z_x_dz

####
def hmc(X, max_iter, dim_z, z_start, epsilon, leap_frog_steps, burn_in=0):
    print "Sampling from HMC\nIterations={0} (burn in={1})\nz dimension={2}\nepsilon={3}\nleap frog steps={4}".format(max_iter, burn_in, dim_z, epsilon, leap_frog_steps)
    max_iter += burn_in
    z_sample = np.zeros((max_iter, dim_z))
    p_sample = np.zeros((max_iter, dim_z))
    iteration = 0
    sample_count = 0
    print_freq = 100

    z_sample[0, :] = z_start


    while iteration < (max_iter-2):

        # Print every print_freq iterations
        if not iteration%print_freq:
            print "...iteration {0}".format(iteration)

        # Sample new momentum
        p_sample[iteration,:] = np.random.randn(dim_z)

        # Calculate current Hamiltonian
        H_null = hamiltonian(z_sample[iteration, :], X, w3, b3, w2, b2, p_sample[iteration, :])
        z_step = z_sample[iteration,:]
        p_step = p_sample[iteration,:]

        for i in range(1, leap_frog_steps):
            p_half_step = p_step - 0.5 * epsilon * force(z_step, X, w3, b3, w2, b2)
            z_step_one = z_step + epsilon * p_half_step
            p_step_one = p_half_step - 0.5 * epsilon * force(z_step_one, X, w3, b3, w2, b2)
            p_step = p_step_one
            z_step = z_step_one

        H_one = hamiltonian(z_step, X, w3, b3, w2, b2, p_step)

        alpha = np.min([1, np.exp(- H_one + H_null)])

        # Acceptance step
        if np.random.rand(1) < alpha:
            z_sample[iteration + 1, :] = z_step
            p_sample[iteration + 1, :] = p_step
            sample_count += 1
        else:
            z_sample[iteration + 1, :] = z_sample[iteration, :]
            p_sample[iteration + 1, :] = p_sample[iteration, :]
        iteration += 1

    print "Acceptance rate = {0}".format( sample_count/float(iteration))

    # Save sample
    filename = "./samples/z_sample_{0}_{1}.pkl".format(dim_z, time.asctime( time.localtime(time.time()) ).replace(" ", "_").replace(":","_"))
    f = open(filename, 'wb')
    cPickle.dump(z_sample[burn_in:], f)
    f.close()

    #### Effective Sample Size
    for i in range(z_sample.shape[1]):
        z_autocorr = autocorr(z_sample[burn_in:, i])
        z_ess = z_sample.shape[0] * (1-z_autocorr)/ (1+ z_autocorr)
        print "ESS in dimension {0} is {1}".format(i, z_ess)

    return z_sample[burn_in:]

def grid_search(epsilons, leap_frog_steps):
    for L in leap_frog_steps:
        for e in epsilons:
            z_sample = hmc(X=trX[1], max_iter=500, dim_z=dim_z, z_start=np.zeros((1, dim_z)), epsilon=e, leap_frog_steps=L)




#### Load parameters from VAE
dim_z = 2

f = open("./params/params_z{0}.pkl".format(dim_z), 'rb')
params = cPickle.load(f)
f.close()

#### Parameters
# [w_mu, w_sigma, w1, w2, w3, b_mu, b_sigma, b1, b2, b3]
w2 = params[3]
w3 = params[4]
b2 = params[8]
b3 = params[9]

#### Loading data
trX, teX, trY, teY = mnist(onehot=True)
trX = trX[:1000]
teX = teX[:1000] # Let's only consider first 1000 observations for now
trX = np.round(trX, 0)
teX = np.round(teX, 0)




#### Obtaining a z sample
z_sample = hmc(X=trX[1], max_iter=100, dim_z=dim_z, z_start=np.zeros((1, dim_z)), epsilon=0.01, leap_frog_steps=10)

#grid_search([0.001, 0.01, 0.05, 0.075, 0.1], [10])

q = stats.gaussian_kde(z_sample.T)

z_sample_two = hmc(X=trX[1], max_iter=100, dim_z=dim_z, z_start=np.zeros((1, dim_z)), epsilon=0.01, leap_frog_steps=10)

print q(z_sample_two.T).shape


denominator = np.zeros([z_sample_two.shape[0],])

for i in range(z_sample_two.shape[0]):
    denominator[i] = np.exp(log_posterior(z_sample_two[i], trX[1], w3, b3, w2, b2))

marg_likeli_1 = np.mean(q(z_sample_two.T) / denominator)

print marg_likeli_1
