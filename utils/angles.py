import numpy as np

def calculate_angle(a, b, c):
    """
    Calculates the angle at point b between vectors ba and bc in 2D.
    This is more robust for video analysis where Z-depth is inferred and noisy.
    """
    # MODIFIED: Convert to 2D numpy arrays by taking only x and y (index 0 and 1)
    a = np.array([a[0], a[1]]) 
    b = np.array([b[0], b[1]]) 
    c = np.array([c[0], c[1]]) 
    
    # The rest of the logic remains the same
    ba = a - b
    bc = c - b
    
    # Handle potential division by zero
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba == 0 or norm_bc == 0:
        return 0.0 # Or handle as an error/default case

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    
    # Clip the value to handle floating point inaccuracies
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    
    return angle