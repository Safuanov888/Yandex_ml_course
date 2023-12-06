import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    length = data.shape[1]
    # находим максимальный собственный вектор и максимальное собственное значение
    eigenvector_max = np.ones(length)
    for i in range(num_steps):
        eigenvector_max = data.dot(eigenvector_max)
        eigenvector_max /= np.sqrt(np.sum(eigenvector_max ** 2))
    eigenvalue_max = np.sqrt(np.sum(data.dot(eigenvector_max) ** 2)) / np.sqrt(np.sum(eigenvector_max ** 2))

    return float(eigenvalue_max), eigenvector_max