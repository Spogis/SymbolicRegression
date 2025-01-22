import sympy as sp
import math

def calculate_y(x):
    y = 2.21949495614222*x**2 - 5.3670266836839*x + 3.21636591215306
    return float(y)

if __name__ == "__main__":
    # Test the function with example values
    x = 0.0  # Example values
    result = calculate_y(x)
    print(f"y = {result}")
