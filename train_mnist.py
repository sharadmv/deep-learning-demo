from tensorflow.examples.tutorials.mnist import input_data
from deepx.nn import *
from deepx.loss import *
from deepx.optimize import *

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

deepnet = Vector(784) >> Tanh(200) >> Tanh(200) >> Softmax(10)
rmsprop = RMSProp(deepnet, CrossEntropy())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    print rmsprop.train(batch_xs, batch_ys, 1)

print "Accuracy: ",
Xtest, ytest = mnist.test.images, mnist.test.labels
results = deepnet.predict(Xtest)
print float(sum(results.argmax(axis=1) == ytest.argmax(axis=1))) / Xtest.shape[0]
