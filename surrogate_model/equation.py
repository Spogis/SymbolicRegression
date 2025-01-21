import sympy as sp
import math

def calculate_y(x):
    y = 2.21966004397058*x**2 - 5.37115099850075*x + 3.22095213396296
    return float(y)

if __name__ == "__main__":
    # Test the function with example values
    x = [0.0]  # Example values
    result = calculate_y(x)
    print(f"Result for input values: y = {result}")
