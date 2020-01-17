"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    sample_num = X.shape[0]  
    class_num = W.shape[1]
    # Computing the cross entropy loss 

    for sample_index in range(sample_num):
        scorse = np.dot(X[sample_index,:], W) # Input of current data sample mul the weight metrix, get a score matrix of shape (1 * C)
        pred_class = y[sample_index] # The prediction label of current data sample
        exp_pred_prob = np.exp(scorse[pred_class]) # Get the score of the predicted class
        exp_sum_prob = np.sum(np.exp( scorse ) ) # Get the sum of scores of all of the classes
        softmax =  exp_pred_prob / exp_sum_prob # Get the softmax result of current data sample
        loss -= np.log(softmax) # get the loss function
    
    # Computing the cross gradient

        for class_index in range(class_num):
            if class_index == pred_class:
                dW [:,class_index] += (softmax - 1) * X[sample_index,:].transpose() # If the class index is the same with the pre_class.
            else:
                softmax_class = np.exp(scorse[class_index]) / exp_sum_prob
                dW [:,class_index] += softmax_class  * np.transpose(X[sample_index,:]) # If the class index is different with the pre_class.

    loss = loss / sample_num + 0.5 * reg * np.sum(W * W) # add regularization!
    dW = dW / sample_num + reg/sample_num * W  # add regularization!

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    # pass
    # Compute the loss
    class_num = W.shape[1]
    sample_num = X.shape[0]
    
    scores = np.dot(X, W) # Each row is the scores of 1 sample w.r.t. 10 classes.
    exp_scores = np.exp(scores)
    exp_scores_sum = np.sum( exp_scores, axis = 1, keepdims = True) # Sum the exp scores for each sample
    softmax_matrix = exp_scores / exp_scores_sum # return a softmax matirx
    pred_score = softmax_matrix[range(sample_num),y]

    loss = np.sum(-np.log(pred_score))
    loss = loss / sample_num + 0.5 * reg * np.sum(W * W)

    # Compute the gradient
    
    softmax_matrix[range(sample_num),y] -= 1
    dW = np.dot(X.transpose(),softmax_matrix)
    dW = dW / sample_num + reg / sample_num * W 

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7, 5e-5]
    regularization_strengths = [2.5e4, 5e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################

    # pass
    learning_rates_modi = [ i for i in np.linspace(learning_rates[0], learning_rates[1], num=2)]
    regularization_strengths_modi = [ i for i in np.linspace(regularization_strengths[0], regularization_strengths[1], num=2)]

    for rate in learning_rates_modi:
        for regl in regularization_strengths_modi:
            classifier = SoftmaxClassifier()
            classifier.train(X_train, y_train, learning_rate=rate, reg=regl, num_iters=6000)

            y_train_pred = classifier.predict(X_train)
            train_acc = np.mean(y_train_pred == y_train) 

            y_val_pred = classifier.predict(X_val)
            val_acc = np.mean(y_val_pred == y_val) 

            results[(rate,regl)] = (train_acc, val_acc) 
    
            if val_acc > best_val:
                best_val = val_acc
                best_softmax = classifier


    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
