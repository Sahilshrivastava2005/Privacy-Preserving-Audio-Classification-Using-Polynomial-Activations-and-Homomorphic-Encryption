import numpy as np
from scipy.optimize import minimize

def compute_pse(p, x, y):
    """
    Step 2: Define Percent Squared Error (PSE)
    Calculates the PSE for a given set of polynomial coefficients.
    """
    # Evaluate polynomial: p0*x^n + p1*x^(n-1) + ... + pn
    p_x = np.polyval(p, x)
    
    # Compute the squared percentage error for each point
    # Note: Adding a tiny epsilon to the denominator to prevent division by zero if y contains 0
    epsilon = 1e-10
    squared_errors = ((p_x - y) / (y + epsilon)) ** 2
    
    # Calculate the mean and multiply by 100
    pse = np.mean(squared_errors) * 100
    return pse

def optimize_polynomial_coefficients(x, y, n):
    """
    Step 1 & 3: Construct System and Optimize
    Finds the polynomial coefficients that minimize the PSE.
    
    Parameters:
    x (array-like): x data points
    y (array-like): y data points
    n (int): highest polynomial degree
    
    Returns:
    numpy.ndarray: Optimized coefficients [p0, p1, ..., pn]
    """
    x = np.array(x)
    y = np.array(y)
    
    # Step 1: Standard least squares to get a good initial guess for the optimizer
    # This solves the standard Vandermonde system for absolute error
    initial_guess_p = np.polyfit(x, y, n)
    
    # Step 3: Optimize Polynomial Coefficients
    # We search for p that minimizes the PSE
    result = minimize(
        fun=compute_pse, 
        x0=initial_guess_p, 
        args=(x, y), 
        method='BFGS' # A robust optimization algorithm
    )
    
    if not result.success:
        print("Warning: Optimization may not have converged.")
        print(result.message)
        
    return result.x

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Sample data points (x_i, y_i)
    x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Let's create some y data (e.g., roughly y = 2x^2 + 3x + 1)
    y_data = np.array([6.1, 14.8, 28.2, 44.9, 66.0])
    
    # Degree of the polynomial
    degree_n = 2
    
    # Run the algorithm
    optimized_p = optimize_polynomial_coefficients(x_data, y_data, degree_n)
    
    # Calculate the final PSE to see how well it performed
    final_pse = compute_pse(optimized_p, x_data, y_data)
    
    print(f"Target Degree (n): {degree_n}")
    print(f"Optimized Coefficients [p0, p1, ..., pn]:\n{optimized_p}")
    print(f"Final Percent Squared Error (PSE): {final_pse:.4f}%")