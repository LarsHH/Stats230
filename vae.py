import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor.nlinalg as nlinalg
import numpy as np
from load import mnist
import cPickle

srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def sigmoid(X):
    return 1/(1 + T.exp(-X))

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X


# Model set up
def model(epsilon, X, w_mu, b_mu, w_sigma, b_sigma, w1, b1, w2, b2, w3, b3):

    h = T.tanh(T.dot(X, w1) + b1)

    # Note: mu and sigma are both vectors
    mu = T.dot(h, w_mu) + b_mu

    sigma = 0.5 * (T.dot(h, w_sigma) + b_sigma)


    # Note: elementwise product of sigma with epsilon because sigma is a vector
    z = mu + T.exp(sigma) * epsilon

    # hidden layer
    h2 = T.tanh(T.dot(z, w2) + b2)

    # output layer (/last hidden layer)
    y = sigmoid(T.dot(h2, w3) + b3)

    return y, mu, sigma


# Some dimension
dim_X = 784
dim_h = 500
dim_z = 2


# This is the placeholder for our data matrix while we set up the model
X = T.fmatrix()

# Initializing parameters (all shared variables)
# initialize weights as N(0,0.01) - see init weights for details
w1 = init_weights((dim_X, dim_h))
b1 = theano.shared(floatX(np.zeros(dim_h,)))

w_mu = init_weights((dim_h, dim_z))
b_mu = theano.shared(floatX(np.zeros(dim_z,)))

w_sigma = init_weights((dim_h, dim_z))
b_sigma = theano.shared(floatX(np.zeros(dim_z,)))

w2 = init_weights((dim_z, dim_h))
b2 = theano.shared(floatX(np.zeros(dim_h,)))

w3 = init_weights((dim_h, dim_X))
b3 = theano.shared(floatX(np.zeros(dim_X,)))
# initialize biases as zero





# Random number generation within Theano
epsilon = T.fmatrix()

# Setting up model
y, mu, sigma = model(epsilon, X, w_mu, b_mu, w_sigma, b_sigma, w1, b1, w2, b2, w3, b3)

# Log likelihood for decoder p(x|z)
#log_lik = T.mean(X * T.log(y) + (1 - X) * T.log(1 - y))
log_lik = -T.nnet.binary_crossentropy(y, X).sum()

# KL Divergence
D_KL = 0.5 * T.sum(1 + 2*sigma - mu**2 - T.exp(2*sigma))

# Total cost ( signs correct??? )
L = log_lik + D_KL

# List of parameters to be updated (add to this if you need new parameter)
params = [w_mu, w_sigma, w1, w2, w3, b_mu, b_sigma, b1, b2, b3]

# Update Rule - good for now
updates = RMSprop(-L, params, lr=0.001)

# Compiling functions / setting up computational graph
train = theano.function(inputs=[X, epsilon], outputs=L, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
predict = theano.function(inputs=[X, epsilon], outputs=L, allow_input_downcast=True, on_unused_input='ignore')


# Loading data
trX, teX, trY, teY = mnist(onehot=True)
# binarizing data
trX = floatX(np.round(trX, 0))
teX = floatX(np.round(teX, 0))
# variable to adjust batch size
n_epochs = 10
batchsize = 100
samples_seen = 0
cost_this_epoch = 0
samples_this_epoch = 0
train_history = np.zeros((2, n_epochs))
test_history = np.zeros((2, n_epochs))

# Training : just iterate over training set and plug into train function to make updates
for i in range(n_epochs):
    samples_this_epoch = 0
    cost_this_epoch = 0
    for start, end in zip(range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
        e = np.random.normal(0, 1, [batchsize, dim_z])
        cost = train(trX[start:end], e)
        cost_this_epoch += cost
        samples_this_epoch += batchsize

    samples_seen += samples_this_epoch
    train_history[0, i] = samples_seen
    train_history[1, i] = cost_this_epoch/float(samples_this_epoch)
    print "Training: Samples seen={0}, Avg Cost per Sample={1}".format(train_history[0, i], train_history[1, i])
    e = np.random.normal(0, 1, [teX.shape[0], dim_z])
    test_history[0, i] = samples_seen
    test_history[1, i] = predict(teX, e)/float(teX.shape[0])
    print "Test: Samples seen={0}, Avg Cost per Sample={1}".format(test_history[0, i], test_history[1, i])


# Save the weights
f = open("./params/params_z{0}.pkl".format(dim_z), 'wb')
cPickle.dump([param.get_value() for param in params], f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

# Save the results
f = open("./results/train_history_z{0}.pkl".format(dim_z), 'wb')
cPickle.dump(train_history, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

f = open("./results/test_history_z{0}.pkl".format(dim_z), 'wb')
cPickle.dump(test_history, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
