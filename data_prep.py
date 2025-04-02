import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

def generate_data(n_samples=1000, seed=42, noise_std=0.1):
    # Set the random seed for reproducibility
    np.random.seed(seed)
    
    # Generate x1 and x2 uniformly in [-10, 10]
    x = np.random.uniform(-6, 6, (n_samples, 2))
    x1 = x[:, 0]
    x2 = x[:, 1]
    
    # Define the 20 functions (a mix of linear and nonlinear)
    f1  = x1
    f2  = x2
    f3  = np.sin(x1)
    f4  = np.sin(x2)
    f5  = np.cos(x1)
    f6  = np.cos(x2)
    f7  = x1 * x2
    f8  = x1 + x2
    f9  = x1 - x2
    f10 = np.tanh(x1)
    f11 = np.tanh(x2)
    f12 = np.square(x1)
    f13 = np.square(x2)
    f14 = np.sqrt(np.abs(x1) + 1)
    f15 = np.sqrt(np.abs(x2) + 2)
    f16 = np.log(np.abs(x1) + 3)
    f17 = np.log(np.abs(x2) + 4)
    f18 = np.exp(-np.square(x1))
    f19 = np.exp(-np.square(x2))
    f20 = np.sin(x1 * x2)
    
    # Stack to form a (n_samples x 20) data matrix
    data = np.stack([f1, f2, f3, f4, f5, f6, f7, f8, f9, 
                     f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20], axis=1)
    
    # Add Gaussian noise to the data
    noise = np.random.normal(0, noise_std, data.shape)
    noisy_data = data + noise
    
    return noisy_data, x

def split_data(data, ratios=(0.7, 0.15, 0.15), seed=42):
    """
    Randomly samples data to split into train, development, and test sets based on provided ratios.
    """
    np.random.seed(seed)
    n = data.shape[0]
    # Generate a random permutation of indices
    indices = np.random.permutation(n)
    
    train_end = int(ratios[0] * n)
    dev_end = train_end + int(ratios[1] * n)
    
    train_data = data[indices[:train_end]]
    dev_data = data[indices[train_end:dev_end]]
    test_data = data[indices[dev_end:]]
    
    return train_data, dev_data, test_data

def save_data(train, dev, test, x, filename='data.npz'):
    """
    Saves the training, development, and test sets along with the original x values to a file.
    """
    np.savez(filename, train_data=train, dev_data=dev, test_data=test, x=x)
    print(f"Data saved to {filename}")

def main():
    # Generate noisy data
    noisy_data, x = generate_data(n_samples=1000, seed=42, noise_std=0.1)
    
    # Randomly sample and split the data into train, dev, and test sets
    train_data, dev_data, test_data = split_data(noisy_data, ratios=(0.7, 0.15, 0.15), seed=42)
    
    # Save the split data along with the original x values
    save_data(train_data, dev_data, test_data, x, filename='data.npz')
    
    # Optional: Visualize one of the features in a 3D plot (for demonstration)
    # Here we plot f8 (x1 + x2) vs. f7 (x1 * x2) vs. f20 (sin(x1*x2))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(noisy_data[:, 7], noisy_data[:, 6], noisy_data[:, 19],
                    c=noisy_data[:, 19], cmap='viridis', marker='o', s=50)
    ax.set_xlabel('$X_{l1}$ (Sum: x1+x2)')
    ax.set_ylabel('$X_{l2}$ (Product: x1*x2)')
    ax.set_zlabel('$f_{20}$ (sin(x1*x2))')
    ax.set_title('3D Scatter Plot of Selected Features')
    fig.colorbar(sc, shrink=0.5, aspect=5)
    plt.show()

if __name__ == "__main__":
    main()
