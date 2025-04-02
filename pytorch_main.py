import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.special import erf

# -------------------------------
# 1) Load the autoencoder model with pre-trained weights
# -------------------------------
def load_autoencoder_model(weights_file="autoencoder_weights.pt"):
    from vae_model_pytorch import AE 
    model = AE()
    state_dict = torch.load(weights_file, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# -------------------------------
# 2) Load the training data from data.npz
# -------------------------------
def load_train_data(filename="data.npz"):
    data = np.load(filename)
    train_data = data["test_data"].astype(np.float32)
    return train_data

# -------------------------------
# RBF Kernel and Expected Improvement
# -------------------------------
def rbf_kernel(X1, X2, length_scale):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    sqdist = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :])**2, axis=2)
    return np.exp(-0.5 * sqdist / (length_scale**2))

def expected_improvement(mu, sigma, f_best, xi=0.01):
    sigma = np.maximum(sigma, 1e-9)
    Z = (mu - f_best - xi) / sigma
    phi = (1.0 / np.sqrt(2*np.pi)) * np.exp(-0.5 * Z**2)
    Phi = 0.5 * (1 + erf(Z / np.sqrt(2)))
    return (mu - f_best - xi) * Phi + sigma * phi

# -------------------------------
# 3) f1..f20 using (z1, z2)
# -------------------------------
def compute_f_values(z1, z2):
    f1  = z1
    f2  = z2
    f3  = np.sin(z1)
    f4  = np.sin(z2)
    f5  = np.cos(z1)
    f6  = np.cos(z2)
    f7  = z1 * z2
    f8  = z1 + z2
    f9  = z1 - z2
    f10 = np.tanh(z1)
    f11 = np.tanh(z2)
    f12 = np.square(z1)
    f13 = np.square(z2)
    f14 = np.sqrt(np.abs(z1) + 1)
    f15 = np.sqrt(np.abs(z2) + 2)
    f16 = np.log(np.abs(z1) + 3)
    f17 = np.log(np.abs(z2) + 4)
    f18 = np.exp(-np.square(z1))
    f19 = np.exp(-np.square(z2))
    f20 = np.sin(z1 * z2)
    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
                     f11, f12, f13, f14, f15, f16, f17, f18, f19, f20])

# -------------------------------
# 4) Objective: decode (z1,z2) -> 20D and compare with computed f_i(z1,z2)
# -------------------------------
class LatentObjectiveFunctionLS:
    def __init__(self, model, x_target):
        """
        model: a PyTorch AE model with .decoder that maps (1,2) -> (1,20).
        x_target: a numpy array of shape (20,) representing the target.
        """
        self.model = model
        

    def fun_test(self, z):
        """
        Evaluate the sum of squared errors:
          SSE = sum_i (f_i(z1, z2) - x_recon[i])^2
        where:
          - f_i are computed from z1,z2 via compute_f_values,
          - x_recon is the 20D output decoded from z.
        """
        z1, z2 = z[0], z[1]
        # Compute the 20 function values from (z1,z2)
        f_vals = compute_f_values(z1, z2)
        
        # Decode z -> 20-D output (x_recon)
        z_np = np.array([[z1, z2]], dtype=np.float32)  # shape: (1,2)
        z_tensor = torch.tensor(z_np)
        self.model.eval()
        with torch.no_grad():
            x_recon_tensor = self.model.decoder(z_tensor)
        x_recon = x_recon_tensor.cpu().numpy()[0]  # shape: (20,)
        
        # Compute SSE between f_vals and x_recon
        diff = f_vals - x_recon
        return float(np.sum(diff**2))

# -------------------------------
# 5) Initialize surrogate using training data latent representations
# -------------------------------
def init_surrogate_from_train_data(obj_func, model, n_samples=None):
    """
    Builds initial surrogate training data from latent representations of training examples.
    Returns:
      X_train: A numpy array of latent vectors.
      y_train: A numpy array of objective function values.
    """
    train_data = load_train_data("data.npz")
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    with torch.no_grad():
        # Assuming model.forward returns (decoded, encoded)
        _, z_train = model(train_tensor)
    latent_candidates = z_train.cpu().numpy()  # shape: (num_train, latent_dim)
    
    # Optionally subsample if n_samples is specified.
    if n_samples is not None and n_samples < latent_candidates.shape[0]:
        indices = np.random.choice(latent_candidates.shape[0], n_samples, replace=False)
        latent_candidates = latent_candidates[indices]
    
    y_train = np.array([obj_func.fun_test(x) for x in latent_candidates])
    return latent_candidates, y_train

# -------------------------------
# 6) Bayesian Optimization Loop
# -------------------------------
def latent_bo_optimizer_iterative(obj_func, bounds, n_iters=30, n_init=5, model=None):
    """
    Runs Bayesian Optimization in a 2D latent space.
    - obj_func: an object with a fun_test(z) method.
    - bounds: [[z1_min, z1_max], [z2_min, z2_max]].
    - n_iters: number of BO iterations.
    - n_init: number of initial points from training data to build the surrogate.
    - model: autoencoder model used to compute latent representations.
    Returns: best_z, best_obj, history.
    """
    d = len(bounds)
    # Initialize surrogate using latent representations from training data.
    X_train, y_train = init_surrogate_from_train_data(obj_func, model, n_samples=n_init)
    
    # Initialize history with best objective so far.
    history = [np.min(y_train)]
    
    alpha = 1e-6
    length_scale = 0.5
    
    for it in range(n_iters):
        # Generate candidate points (random sampling within bounds)
        num_candidates = 100
        candidates = np.zeros((num_candidates, d))
        for j in range(d):
            candidates[:, j] = np.random.uniform(bounds[j][0], bounds[j][1], num_candidates)
        
        # Build covariance matrix for current training data.
        K = rbf_kernel(X_train, X_train, length_scale) + alpha**2 * np.eye(len(X_train))
        L = np.linalg.cholesky(K)
        
        mu_candidates = []
        sigma_candidates = []
        for cand in candidates:
            k_star = rbf_kernel(X_train, cand.reshape(1, -1), length_scale).flatten()
            v = np.linalg.solve(L, k_star)
            mu_pred = v.T.dot(np.linalg.solve(L, y_train))
            sigma_pred = np.sqrt(max(0, 1 - np.dot(v, v)))
            mu_candidates.append(mu_pred)
            sigma_candidates.append(sigma_pred)
        
        mu_candidates = np.array(mu_candidates)
        sigma_candidates = np.array(sigma_candidates)
        
        f_best = np.argmin(y_train)
        EI = expected_improvement(mu_candidates, sigma_candidates, f_best, xi=0.01)
        best_idx = np.argmax(EI)
        
        x_next = candidates[best_idx]
        y_next = obj_func.fun_test(x_next)
        
        # Append new observation.
        X_train = np.vstack([X_train, x_next])
        y_train = np.append(y_train, y_next)
        
        current_best = np.argmin(y_train)
        history.append(y_next)
        
        print(f"Iteration {it+1}: Proposed z = {x_next}, Objective = {y_next}")
        print(f"Current best objective = {current_best} (over {len(y_train)} samples)\n")
    
    best_overall_idx = np.argmin(y_train)
    best_z = X_train[best_overall_idx]
    best_obj = y_train[best_overall_idx]
    return best_z, best_obj, history

# -------------------------------
# 7) Main Example
# -------------------------------
def main():
    # A) Load training data.
    train_data = load_train_data("data.npz")
    print(f"Train data shape: {train_data.shape}")
    # Use the first row as the target for demonstration.
    x_target = train_data[0]
    
    # B) Load the autoencoder model.
    ae_model = load_autoencoder_model("autoencoder_weights.pt")
    
    # C) Create the objective function.
    ls_obj = LatentObjectiveFunctionLS(ae_model, x_target)
    
    # D) Define bounds for z1 and z2.
    bounds_latent = [[-5, 5], [-5, 5]]
    
    # E) Run Bayesian Optimization, passing the model.
    n_iters = 100
    best_z, best_obj, history = latent_bo_optimizer_iterative(ls_obj, bounds_latent, n_iters=n_iters, n_init=150, model=ae_model)
    
    print("\nOptimization completed.")
    print(f"Best latent point found: {best_z}")
    print(f"Best objective value: {best_obj}")
    
    # F) Plot the objective over iterations.
    plt.figure(figsize=(8, 5))
    plt.plot(history, marker='o', linestyle='-', color='b')
    plt.xlabel('BO Iteration')
    plt.ylabel('Least-Squares Objective')
    plt.title('Objective Value over BO Iterations')
    plt.grid(True)
    plt.show()
    
    # G) Decode the best latent point to compare.
    z1, z2 = best_z[0], best_z[1]
    z_tensor = torch.tensor([[z1, z2]], dtype=torch.float32)
    ae_model.eval()
    with torch.no_grad():
        x_recon_tensor = ae_model.decoder(z_tensor)
    x_recon_opt = x_recon_tensor.cpu().numpy()[0]
    
    final_f_vals = compute_f_values(z1, z2)
    print("\nFinal f-values from best latent point:")
    print(final_f_vals)
    print("\nDecoded output from best latent point:")
    print(x_recon_opt)
    print("\nTarget vector:")
    print(x_target)
    
    sse = np.sum((final_f_vals - x_recon_opt)**2)
    print("\nFinal SSE between f-values and decoded output:", sse)

if __name__ == "__main__":
    main()
