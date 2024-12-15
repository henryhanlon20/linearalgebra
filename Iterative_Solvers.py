import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Solving steady-state 2D heat conduction equation over a rectangular domain:
#     ∂²T/∂x² + ∂²T/∂y² = 0

# Boundary Conditions:
#     - Left edge (x=0):      T = T_left
#     - Right edge (x=Lx):    T = T_right
#     - Bottom edge (y=0):    T = T_bottom
#     - Top edge (y=Ly):      T = T_top

#Parameters
# Domain dimensions
Lx = 1.0       # Length in x-direction (meters)
Ly = 1.0       # Length in y-direction (meters)

# Boundary conditions
T_left = 100.0    # Temperature at x=0 (°C)
T_right = 50.0    # Temperature at x=Lx (°C)
T_bottom = 75.0   # Temperature at y=0 (°C)
T_top = 0.0       # Temperature at y=Ly (°C)

# Numerical parameters
nx = 20          # Number of grid points in x-direction (excluding boundaries)
ny = 20          # Number of grid points in y-direction (excluding boundaries)
dx = Lx / (nx + 1)  # Grid spacing in x-direction
dy = Ly / (ny + 1)  # Grid spacing in y-direction

# Generate grid points
x = np.linspace(dx, Lx - dx, nx)
y = np.linspace(dy, Ly - dy, ny)
X, Y = np.meshgrid(x, y)

# Total number of unknowns
N = nx * ny

#Define Coefficient Matrix A

# Initialize the coefficient matrix 'A' and right-hand side vector 'b'
A = np.zeros((N, N))
b = np.zeros(N)

# Mapping from 2D grid to 1D vector
def index(i, j):
    return i * nx + j

# Construct 'A' and 'b'
for i in range(ny):
    for j in range(nx):
        k = index(i, j)
        A[k, k] = -2.0 / dx**2 - 2.0 / dy**2  # Center node

        # Left neighbor
        if j > 0:
            A[k, index(i, j - 1)] = 1.0 / dx**2
        else:
            # Boundary at x=0
            b[k] -= (1.0 / dx**2) * T_left

        # Right neighbor
        if j < nx - 1:
            A[k, index(i, j + 1)] = 1.0 / dx**2
        else:
            # Boundary at x=Lx
            b[k] -= (1.0 / dx**2) * T_right

        # Bottom neighbor
        if i > 0:
            A[k, index(i - 1, j)] = 1.0 / dy**2
        else:
            # Boundary at y=0
            b[k] -= (1.0 / dy**2) * T_bottom

        # Top neighbor
        if i < ny - 1:
            A[k, index(i + 1, j)] = 1.0 / dy**2
        else:
            # Boundary at y=Ly
            b[k] -= (1.0 / dy**2) * T_top

#Gaussian Elimination 
def gaussian_elimination(A, b):

    #We know that A has dimension NxN and b has dimension N
    N=A.shape[0]
    #Forward Elimination
    for k in range(A.shape[0]-1): #for all rows except the last one
      #Identify the row with the largest pivot
      pivot_row = np.argmax(np.abs(A[k:,k]))+k
      #Now see if that row is our starting row
      if pivot_row!=k:
        A[[k,pivot_row]]=A[[pivot_row,k]] #swap rows of A
        b[[k,pivot_row]]=b[[pivot_row,k]] #swap rows of b
      for i in range(k+1,N):
        factor=A[i,k]/A[k,k] #compute the required factor
        A[i,k:]=A[i,k:]-factor*A[k,k:] #conduct the row operation on A
        b[i]=b[i]-factor*b[k] #conduct the row operation on b

    #Back Substitution
    x=np.zeros(N) #Create the solution vector
    for k in range(N-1,-1,-1):
      x[k]=(b[k]-np.dot(A[k,k+1:],x[k+1:]))/A[k,k]
    return x

#Jacobi Method
def jacobi_method(A, b, tol=1e-6, max_iterations=10000):
    N=A.shape[0]
    x=np.zeros(N) #here is my first guess.
    D=np.diag(A)
    R=A-np.diag(D)
    error_norms=[]

    for iteration in range(max_iterations):
        x_old=x.copy()
        x=(b-np.dot(R,x_old))/D
        error_norm = np.linalg.norm(x-x_old, ord=np.inf)
        error_norms.append(error_norm)

        if error_norm<tol:
            break
    return x, error_norms

#Gauss-Seidel 
def gauss_seidel_method(A, b, tol=1e-6, max_iterations=10000):
    N=A.shape[0]
    x=np.zeros(N)
    D=np.diag(A)
    R=A-np.diag(D)
    U=np.triu(A,k=1) #take the upper triangular part excluding main diagonal
    L=np.tril(A,k=-1) #take the lower triangular part excluding the main diagonal
    error_norms=[]

    for iteration in range(max_iterations):
        x_new=np.copy(x)
        for i in range(N):
          sum1=sum(A[i][j]*x_new[j] for j in range (i))
          sum2=sum(A[i][j]*x[j] for j in range(i+1,N))
          x_new[i]=(b[i]-sum1-sum2)/A[i][i]
        error_norm = np.linalg.norm(x_new-x, ord=np.inf)
        error_norms.append(error_norm)
        x=x_new

        if error_norm<tol:
            break
    return x, error_norms

#Successive Over-Relaxation 
def sor_method(A, b, omega=1.5, tol=1e-6, max_iterations=10000):
    N=A.shape[0]
    x=np.zeros(N)
    D=np.diag(A)
    R=A-np.diag(D)
    U=np.triu(A,k=1) #take the upper triangular part excluding main diagonal
    L=np.tril(A,k=-1) #take the lower triangular part excluding the main diagonal
    error_norms=[]

    for iteration in range(max_iterations):
        x_new=np.copy(x)
        for i in range(N):
          sum1=sum(A[i][j]*x_new[j] for j in range (i))
          sum2=sum(A[i][j]*x[j] for j in range(i+1,N))
          x_new[i]=(1-omega)*x[i]+omega*(b[i]-sum1-sum2)/A[i][i]
        error_norm = np.linalg.norm(x_new-x, ord=np.inf)
        error_norms.append(error_norm)
        x=x_new

        if error_norm<tol:
            break
    return x, error_norms

# Solve using Gaussian Elimination
A_ge = A.copy()
b_ge = b.copy()
T_vec = gaussian_elimination(A_ge, b_ge)

# Solve using Jacobi Method
T_jacobi_vec, errors_jacobi = jacobi_method(A, b)

# Solve using Gauss-Seidel Method
T_gs_vec, errors_gs = gauss_seidel_method(A, b)

# Solve using SOR Method
omega = 1.8  # Optimal relaxation factor (may need tuning)
T_sor_vec, errors_sor = sor_method(A, b, omega=omega)

#Make Solution Vectors 2D 

T = T_vec.reshape((ny, nx))
T_jacobi = T_jacobi_vec.reshape((ny, nx))
T_gs = T_gs_vec.reshape((ny, nx))
T_sor = T_sor_vec.reshape((ny, nx))

# Add boundary temperatures
def add_boundaries(T_inner):
    T_full = np.zeros((ny + 2, nx + 2))
    T_full[1:-1, 1:-1] = T_inner
    # Left and right boundaries
    T_full[:, 0] = T_left
    T_full[:, -1] = T_right
    # Bottom and top boundaries
    T_full[0, :] = T_bottom
    T_full[-1, :] = T_top
    return T_full

T_full = add_boundaries(T)
T_jacobi_full = add_boundaries(T_jacobi)
T_gs_full = add_boundaries(T_gs)
T_sor_full = add_boundaries(T_sor)

#Temperature Distribution Plotting Function 
#Function to plot temperature distributions side by side
def plot_temperature_comparison(T_list, method_names, title='Temperature Distribution Comparison'):
     num_methods = len(T_list)
     X_full, Y_full = np.meshgrid(np.linspace(0, Lx, nx + 2), np.linspace(0, Ly, ny + 2))

     # Create subplots for contour plots
     fig, axes = plt.subplots(1, num_methods, figsize=(5 * num_methods, 6))

     for i, T in enumerate(T_list):
         ax = axes[i] if num_methods > 1 else axes
         cp = ax.contourf(X_full, Y_full, T, 20, cmap='jet')
         fig.colorbar(cp, ax=ax)
         ax.set_title(method_names[i])
         ax.set_xlabel('x (m)')
         ax.set_ylabel('y (m)')
         ax.axis('equal')

     plt.suptitle(title)
     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
     plt.show()

     # Surface plots
     if Axes3D is not None:
         fig = plt.figure(figsize=(5 * num_methods, 6))
         for i, T in enumerate(T_list):
             ax = fig.add_subplot(1, num_methods, i + 1, projection='3d')
             surf = ax.plot_surface(X_full, Y_full, T, cmap=cm.jet)
             fig.colorbar(surf, ax=ax)
             ax.set_title(method_names[i])
             ax.set_xlabel('x (m)')
             ax.set_ylabel('y (m)')
             ax.set_zlabel('Temperature (°C)')
         plt.suptitle(title)
         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
         plt.show()
     else:
         print("3D plotting is not available in your Matplotlib installation.")

# Plot temperature distributions side by side
T_methods = [T_full, T_jacobi_full, T_gs_full, T_sor_full]
method_names = ['Gaussian Elimination', 'Jacobi Method', 'Gauss-Seidel Method', f'SOR Method (ω={omega})']
plot_temperature_comparison(T_methods, method_names, title='Temperature Distribution Comparison')

#Convergence Plot

plt.figure(figsize=(10, 6))
plt.semilogy(errors_jacobi, 'r-', label='Jacobi Method')
plt.semilogy(errors_gs, 'g-', label='Gauss-Seidel Method')
plt.semilogy(errors_sor, 'b-', label=f'SOR Method (ω={omega})')
plt.xlabel('Iteration Number')
plt.ylabel('Error Norm (Infinity Norm)')
plt.title('Convergence of Iterative Methods')
plt.legend()
plt.grid(True)
plt.show()