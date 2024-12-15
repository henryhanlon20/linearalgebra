import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Demonstration of Power Method and QR Method

#Parameters
N = 15        # Number of floors in the building
m = 1000.0    # Mass of each floor (kg)
k = 5e6       # Stiffness between floors (N/m)

#Function for Creation of Mass Matrix
#Diagonal matrix with elements corresponding to the mass of a floor. 
#Inputs: N (number of floors), m (Mass of each floor [kg])
#Returns: M, an NxN matrix with diagonal elements of "m"
def create_mass_matrix(N, m):
    M=np.zeros((N,N))
    np.fill_diagonal(M,m)
    return M

#Function for Creation of Stiffness Matrix
#Tridiagonal matrix with elements corresponding to the linkages between floors. 
#Inputs: N (number of floors), k (stiffness between floors [N/m])
#Returns: K, an NxN matrix with elements k, 2k and -k depending on geometry
def create_stiffness_matrix(N, k):
    K=np.zeros((N,N))
    np.fill_diagonal(K,2*k)
    for i in range(N-1):
        K[i,i+1]=-k
        K[i+1,i]=-k
    K[0,0]=k
    K[N-1,N-1]=k
    return K

# Create mass and stiffness matrices
M = create_mass_matrix(N, m)
K = create_stiffness_matrix(N, k)

#Now we convert the equation of motion to a standard eigenvalue problem. 
# Kv=λMv
# A=(M^-1)*K
# Av=λv 

A=np.linalg.inv(M)@K

#Define function for computing the dominant eigenvalue and eigenvector using the Power Method 
#Inputs: A (NxN matrix) formed by M^-1*k, num_iterations: the max number of iterations, tol: the tolerance for evaluating convergence
#Returns: eigenvalue: the largest eigenvalue of A (λ_1), eigenvector: the vector corresponding to λ_1
def power_method(A, num_iterations=1000, tol=1e-10):
    x=np.random.rand(A.shape[0])
    for i in range(num_iterations):
        x_new=A@x
        x_new=x_new/np.linalg.norm(x_new)
        if np.linalg.norm(x_new-x)<tol:
            break
        x=x_new
    eigenvalue=x_new.T@A@x_new
    eigenvector=x_new
    return eigenvalue,eigenvector

# Compute the dominant eigenvalue and eigenvector
lambda_max, v_max = power_method(A)

#Define function for computing the all eigenvalues and eigenvectors using the QR method 
#Inputs: A (NxN matrix) formed by M^-1*k, num_iterations: the max number of iterations, tol: the tolerance for evaluating convergence
#Returns: eigenvalues: a diagonal matrix containing the eigenvalues, eigenvectors: a matrix containing orthogonal columns 
def qr_algorithm(A, tol=1e-10, max_iterations=1000):
    Ak=A.copy()
    eigenvalues=[]
    V=np.eye(A.shape[0])
    for j in range(max_iterations):
        Q,R=np.linalg.qr(Ak)
        Ak=R@Q
        V=V@Q
        if np.linalg.norm(np.diag(Ak)-np.diag(A))<tol:
            break
    eigenvalues=np.diag(Ak)
    return eigenvalues,V

# Compute all eigenvalues and eigenvectors
eigenvalues_qr, eigenvectors_qr = qr_algorithm(A)

# The eigenvalues are λ = ω^2, so we take the square root to get natural frequencies ω
# The eigenvectors correspond to the mode shapes

# Power Method results
natural_frequency_pm = np.sqrt(lambda_max)
mode_shape_pm = v_max

# QR Algorithm results
natural_frequencies_qr = np.sqrt(np.abs(eigenvalues_qr))

# Sort the eigenvalues and corresponding eigenvectors in ascending order
idx = np.argsort(natural_frequencies_qr)
natural_frequencies_qr = natural_frequencies_qr[idx]
eigenvectors_qr = eigenvectors_qr[:, idx]


#Visualization

floors = np.arange(1, N+1)

# Plotting the dominant mode shape from Power Method
plt.figure(1,figsize=(8, 6))
plt.plot(mode_shape_pm, floors, marker='o')
plt.gca().invert_yaxis()
plt.title(f'Dominant Mode Shape from Power Method\nFrequency = {natural_frequency_pm:.2f} rad/s')
plt.xlabel('Amplitude')
plt.ylabel('Floor Number')
plt.grid(True)
#plt.show()

# Plotting mode shapes from QR Algorithm

# 2D Plots of the first few mode shapes
num_modes_to_plot = 5

plt.figure(2,figsize=(15, 8))
for i in range(num_modes_to_plot):
    plt.subplot(1, num_modes_to_plot, i+1)
    plt.plot(eigenvectors_qr[:, i], floors, marker='o')
    plt.gca().invert_yaxis()
    plt.title(f'Mode {i+1}\nFreq = {natural_frequencies_qr[i]:.2f} rad/s')
    plt.xlabel('Amplitude')
    if i == 0:
        plt.ylabel('Floor Number')
    plt.grid(True)
plt.tight_layout()
#plt.show()

# 3D Plot of Mode Shapes
fig = plt.figure(3,figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# meshgrid for time and floor
time_steps = 200
t = np.linspace(0, 2*np.pi, time_steps)
X, Y = np.meshgrid(t, floors)

# mode to visualize in 3D
mode_to_visualize = 1  # Change this to visualize different modes

# displacement for each time step
Z = np.zeros_like(X)
for i in range(len(floors)):
    Z[i, :] = eigenvectors_qr[i, mode_to_visualize - 1] * np.sin(natural_frequencies_qr[mode_to_visualize - 1] * t)

# Plot the surface
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_title(f'3D Visualization of Mode {mode_to_visualize}')
ax.set_xlabel('Time')
ax.set_ylabel('Floor Number')
ax.set_zlabel('Amplitude')
ax.view_init(elev=30, azim=135)  # Adjust view angle
plt.show()

#Results

print("Natural Frequencies (rad/s) from QR Algorithm:")
for i, freq in enumerate(natural_frequencies_qr):
    print(f"Mode {i+1}: {freq:.2f}")

print(f"\nDominant Natural Frequency from Power Method: {natural_frequency_pm:.2f} rad/s")

