import numpy as np

def zero_pad(arr, length):
    zeros = np.zeros((length,))
    zeros[0:arr.shape[0]] = arr
    return zeros

def quadratic_kappa(pred_theirs, pred_ours):
    num_ratings = 9
    y_theirs = pred_theirs.data.numpy().squeeze()
    y_ours = pred_ours.data.numpy().squeeze()

    # convert scores to ints in the range [0, 9]
    their_int = np.round(y_theirs * num_ratings).astype(int)
    ours_int = np.round(y_ours * num_ratings).astype(int)

    n = num_ratings + 1

    similar = np.zeros((n, n))
    for i, j in zip(their_int, ours_int):
        similar[i][j] += 1

    expected = np.outer(zero_pad(np.bincount(their_int), n), zero_pad(np.bincount(ours_int), n))

    expected = expected * (np.sum(similar)/ np.sum(expected))

    temp = np.tile(np.arange(n), (n, 1))
    weights = (temp - temp.T) ** 2 / (num_ratings**2)
    quad_kappa = 1 - (np.sum(np.multiply(weights, similar)) / np.sum(np.multiply(weights, expected)))

    return quad_kappa