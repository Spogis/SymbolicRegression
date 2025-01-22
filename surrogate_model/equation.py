import sympy as sp
import math

def calculate_y(x):
    y = 2.1994387*x**2 - 5.2970787319326*x + 3.16002384592155
    return float(y)

if __name__ == "__main__":
    # Test the function with example values
    x = [0.0]  # Example values
    result = calculate_y(x)
    print(f"Result for input values: y = {result}")
