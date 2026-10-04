import numpy as np

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Compute the mean squared error using gradient descent.

    Args:
        y (np.array): target values
        tx (np.array): input features
        initial w (np.array): initial weights
        max iters (int): maximum number of iterations
        gamma (float): learning rate

    Returns:
        w (np.array): optimized weights
        loss (float): final loss value
    """
    w = initial_w
    for n_iter in range(max_iters):
        e = y - tx.dot(w)
        grad = -tx.T.dot(e) / len(y)
        w -= gamma * grad
    loss = np.mean(e ** 2) / 2
    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """""
    Compute the mean squared error using stochastic gradient descent.

    """""
    w = initial_w.copy()

    for n_iter in range(max_iters):
        i = np.random.randint(len(y))
        e = y[i] - tx[i].dot(w)
        grad = -tx[i] * e
        w -= gamma * grad
    loss = np.mean((y - tx.dot(w)) ** 2) / 2
    return w, loss

def least_squares(y, tx): 
    
    """
    Compute the least squares solution using the normal equation.

    Args:
        y (np.array): target values
        tx (np.array): input features

    Returns:
        w (np.array): optimized weights
        loss (float): final loss value
    """
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = np.mean((y - tx.dot(w)) ** 2) / 2
    return w, loss


def ridge_regression(y, tx, lambda_) :
    """
    Compute the ridge regression solution using the normal equation.

    Args:
        y (np.array): target values
        tx (np.array): input features
        lambda_ (float): regularization parameter

    Returns:
        w (np.array): optimized weights
        loss (float): final loss value
    """
    N, D = tx.shape
    w = np.linalg.solve(tx.T.dot(tx) + 2 * N * lambda_ * np.eye(D), tx.T.dot(y))
    loss = np.mean((y - tx.dot(w)) ** 2) / 2 + lambda_ * np.sum(w ** 2)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Compute the logistic regression solution using gradient descent.
    y is between 0 and 1.

    Args:
        y (np.array): target values
        tx (np.array): input features
        initial w (np.array): initial weights
        max iters (int): maximum number of iterations
        gamma (float): learning rate

    Returns:
        w (np.array): optimized weights
        loss (float): final loss value
    """
    w = initial_w
    for n_iter in range(max_iters):
        pred = 1 / (1 + np.exp(-tx.dot(w)))
        pred = np.clip(pred, 1e-15, 1 - 1e-15)      # Clip predictions to avoid log(0) and numerical instability
        e = pred - y
        grad = tx.T.dot(e) / len(y)
        w -= gamma * grad
    loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    return w, loss
   

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Compute the regularized logistic regression solution using gradient descent.
    y is between 0 and 1. The regularization term is added to the loss function and the gradient.

    Args:
        y (np.array): target values
        tx (np.array): input features
        lambda_ (float): regularization parameter
        initial w (np.array): initial weights
        max iters (int): maximum number of iterations
        gamma (float): learning rate

    Returns:
        w (np.array): optimized weights
        loss (float): final loss value
    """
    w = initial_w
    for n_iter in range(max_iters):
        pred = 1 / (1 + np.exp(-tx.dot(w)))
        pred = np.clip(pred, 1e-15, 1 - 1e-15)      # Clip predictions to avoid log(0) and numerical instability
        e = pred - y
        grad = tx.T.dot(e) / len(y) + 2 * lambda_ * w
        w -= gamma * grad
    loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred)) + lambda_ * np.sum(w ** 2)
    return w, loss
    




