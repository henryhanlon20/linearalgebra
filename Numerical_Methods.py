
"""
The Michaelis-Menten equation describes the rate of enzymatic reactions as a function of substrate concentration.
Estimating the parameters Vmax (maximum reaction rate) and Km (Michaelis constant) is essential for understanding enzyme kinetics.

**Michaelis-Menten Equation**:
   v = (Vmax * [S]) / (Km + [S])

We will also be using a linearization of the Michaelis-Menten equation for implementation of the least squares method. 
2. **Lineweaver-Burk Linearization**:
   1/v = (Km/Vmax)(1/[S]) + (1/Vmax)

We will generate semi-random experimental data of reaction velocities as a function of concentration of substate [S]

Then, we will fit the data using two methods. 
1: Solving the normal equation using QR decomposition to generate least squares fit
2: Estimate non-linear equations directly, using the newton-raphson method and QR decomposition 
"""
import numpy as np
import matplotlib.pyplot as plt

#Generate Simulated Data with noise
def simulate_data(Vmax_true, Km_true, S_data, noise_level):
    v_data_true = (Vmax_true*S_data)/(Km_true+S_data)
    # Add Gaussian noise
    v_data = v_data_true + np.random.normal(0, noise_level, len(S_data))
    return v_data

#Least Squares Estimation
#Inputs: S_data (array) with concentration information, v_data (array) with reaction speed information 
#Returns: Estimated v_max and estimated Michaelis constant for comparison against true. 
def least_squares_estimation(S_data, v_data):
    inv_S = 1 / S_data
    inv_v = 1 / v_data
    y=inv_v
    X = np.column_stack((inv_S, np.ones_like(inv_S)))
    Q, R = np.linalg.qr(X)
    beta = np.linalg.solve(R, np.dot(Q.T, y))
    Vmax_est= 1 / beta[1]
    Km_est= beta[0]*Vmax_est
    return Vmax_est, Km_est

#Newton Raphson Estimation
#Inputs: S_data (array) with concentration information, v_data (array) with reaction speed information, 
# initial_guess (1x2 array with guess for vmax and km), tol and max_iter to control number of iterations 
# and measure convergence, true values for plotting 
#Returns: params (array) the most recent guess, params_history (matrix) a list of all parameters for each iteration,
#errors: a list of errors after each step. 

def newton_raphson_method(S_data, v_data, initial_guess, Vmax_true, Km_true, tol=1e-6, max_iter=100):
    # 1. Initialize variables
    n_data = len(S_data)
    params = initial_guess.copy()
    errors = []
    params_history = [params.copy()]

    # 2. Define function vector and Jacobian
    def compute_function_vector(params):
        F=v_data-(params[0]*S_data)/(params[1]+S_data)
        return F

    def compute_jacobian_matrix(params):
        J=np.column_stack((-S_data/(params[1]+S_data),params[0]*(S_data)/(params[1]+S_data)**2))
        return J
        
    # 3. Newton-Raphson Iteration
    for iteration in range(max_iter):
        F = compute_function_vector(params)
        J = compute_jacobian_matrix(params)
        Q, R = np.linalg.qr(J)
        delta = np.linalg.solve(R, np.dot(Q.T, F))
        params -= delta
        error = np.linalg.norm(F)
        errors.append(error)
        params_history.append(params.copy())
        if error < tol:
            break
        pass
    # 4. Return results
    return params, errors, params_history

def plot_results(S_data, v_data, Vmax_ls, Km_ls, Vmax_nr, Km_nr):
    # Generate fine substrate concentration data for plotting
    S_fit = np.linspace(np.min(S_data), np.max(S_data), 200)

    # Compute fitted velocities for both methods
    v_fit_ls = (Vmax_ls * S_fit) / (Km_ls + S_fit)
    v_fit_nr = (Vmax_nr * S_fit) / (Km_nr + S_fit)

    # Plot experimental data and fitted curves
    plt.figure(figsize=(10, 6))
    plt.scatter(S_data, v_data, color='blue', label='Experimental Data')
    plt.plot(S_fit, v_fit_ls, color='red', linestyle='--', label='Least Squares Fit')
    plt.plot(S_fit, v_fit_nr, color='green', linestyle='-', label='Newton-Raphson Fit')
    plt.xlabel('Substrate Concentration [S]')
    plt.ylabel('Reaction Velocity v')
    plt.title('Comparison of Least Squares and Newton-Raphson Methods')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_residuals(S_data, v_data, Vmax_ls, Km_ls, Vmax_nr, Km_nr):
    # Compute residuals for least squares method
    v_pred_ls = (Vmax_ls * S_data) / (Km_ls + S_data)
    residuals_ls = v_data - v_pred_ls

    # Compute residuals for Newton-Raphson method
    v_pred_nr = (Vmax_nr * S_data) / (Km_nr + S_data)
    residuals_nr = v_data - v_pred_nr

    # Plot residuals
    plt.figure(figsize=(10, 6))
    plt.scatter(S_data, residuals_ls, color='red', label='Least Squares Residuals')
    plt.scatter(S_data, residuals_nr, color='green', label='Newton-Raphson Residuals', marker='x')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Substrate Concentration [S]')
    plt.ylabel('Residuals')
    plt.title('Residuals for Both Methods')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_convergence(errors):
    iterations = np.arange(1, len(errors) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, errors, marker='o', color='purple')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Update Norm')
    plt.title('Convergence of Newton-Raphson Method')
    plt.grid(True)
    plt.show()

def plot_parameter_trajectory(params_history, Vmax_true, Km_true):
    params_array = np.array(params_history)
    iterations = np.arange(len(params_history))

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, params_array[:, 0], marker='o', label='Vmax Estimate')
    plt.plot(iterations, params_array[:, 1], marker='s', label='Km Estimate')
    # Include true values as horizontal lines
    plt.axhline(Vmax_true, color='red', linestyle='--', label='True Vmax')
    plt.axhline(Km_true, color='green', linestyle='--', label='True Km')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Estimates')
    plt.title('Parameter Trajectory in Newton-Raphson Method')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Step 1: Simulate experimental data
    Vmax_true = 100.0
    Km_true = 50.0
    n_points = 20
    S_data = np.linspace(5, 200, n_points)
    np.random.seed(0)
    noise_level = 5.0
    v_data = simulate_data(Vmax_true, Km_true, S_data, noise_level)

    # Step 2: Method 1 - Least Squares Estimation
    Vmax_ls, Km_ls = least_squares_estimation(S_data, v_data)
    print(f"Least Squares Estimates:\nVmax = {Vmax_ls:.2f}, Km = {Km_ls:.2f}")

    # Step 3: Method 2 - Newton-Raphson Method
    initial_guess = np.array([80.0, 40.0])
    params_nr, errors_nr, params_history = newton_raphson_method(S_data, v_data, initial_guess, Vmax_true, Km_true)
    Vmax_nr, Km_nr = params_nr
    print(f"Newton-Raphson Estimates:\nVmax = {Vmax_nr:.2f}, Km = {Km_nr:.2f}")

    # Step 4: Compare and Visualize Results
    plot_results(S_data, v_data, Vmax_ls, Km_ls, Vmax_nr, Km_nr)

    # Plot residuals
    plot_residuals(S_data, v_data, Vmax_ls, Km_ls, Vmax_nr, Km_nr)

    # Plot convergence of Newton-Raphson
    plot_convergence(errors_nr)

    # Plot parameter trajectory in Newton-Raphson
    plot_parameter_trajectory(params_history, Vmax_true, Km_true)

if __name__ == "__main__":
    main()