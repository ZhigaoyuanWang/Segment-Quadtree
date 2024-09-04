import numpy as np

def sum_grid(array, N):
    T, X, Y = array.shape
    grid_x_size = X // N
    grid_y_size = Y // N
    result = np.zeros((T, N, N))

    for t in range(T):
        for i in range(N):
            for j in range(N):
                x_start = i * grid_x_size
                x_end = (i + 1) * grid_x_size if i < N - 1 else X
                y_start = j * grid_y_size
                y_end = (j + 1) * grid_y_size if j < N - 1 else Y
                
                result[t, i, j] = array[t, x_start:x_end, y_start:y_end].sum()
    
    return result

# Example usage
T, X, Y = merged_array.shape
array = merged_array
N = 32  # Example number of grid divisions

result = sum_grid(array, N)
print("Result shape:", result.shape)