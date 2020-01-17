import numpy as np

one_hot = np.zeros([4,3])
y = np.array([1,2,1,0])
one_hot[np.arange(y.size),y] = 1

# print(one_hot)
# print (y.reshape(-1,1))
dic = {}

for i in range(1,5):
    dic["hans"+str(i)] = i

print (dic)

x = np.array([[1],[2],[3],[4]])
y = x.reshape((1,4)) 
z = y.reshape(x.shape)
print (z)

        # # Compute the loss with L2 regularization

        # logits = np.exp(scores)
        # sumProb = np.sum(logits, axis = 1, keepdims = True)
        # softmax = logits / sumProb
        # log_softmax = np.log(softmax)
        # sample_loss = log_softmax [range(N),y]

        # loss = -1 * np.sum(sample_loss)
        # loss = loss / N + 0.5 * self.reg * np.sum(W1 * W1)  +  0.5 * self.reg * np.sum(W2 * W2)



        # # Compute the backward gradient
        # d_A2 = softmax.copy()
        # d_A2[range(N), y] -= 1
        
        # grads['W2'] = H1.T.dot(d_A2) / N + (self.reg * W2)
        # grads['b2'] = np.ones(N).dot(d_A2) / N

        # d_relu_local = H1.copy()
        # d_relu_local = np.where (d_relu_local>0, 1, 0)
        # d_relu_upstream = np.dot(d_A2, W2.T)
        # d_relu = d_relu_local * d_relu_upstream

        # grads['W1'] = X.T.dot(d_relu) / N + (self.reg * W1)
        # grads['b1'] = d_relu.sum(0) / N

###########################################################################

        # W1, b1 = self.params['W1'], self.params['b1']
        # W2, b2 = self.params['W2'], self.params['b2']
        # N,_ = X.shape

        # # Foward pass of firsr layer
        # A1 = X.dot(W1) + b1

        # # # Activation of A1
        # H1 = np.maximum (A1,0)

        # # Foward pass of second layer
        # A2 = H1.dot(W2) + b2
        
        # scores = A2
# dic = {1:1,2:2,3:'fx'}
# for key in dic.values():
#     print(key)
for i in range(5,2,-1):
    print(i)