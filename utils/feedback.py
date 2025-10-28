import numpy as np

def calculate_deviation(user_angles, ref_angles):
    """Calculates the average angular deviation between the user and reference poses."""
    user_np = np.array(user_angles)
    ref_np = np.array(ref_angles)
    
    # Calculate the absolute difference for each angle
    error = np.abs(user_np - ref_np)
    
    # Return a simple score (e.g., 100 minus average error)
    avg_error = np.mean(error)
    score = max(0, 100 - avg_error)
    return score