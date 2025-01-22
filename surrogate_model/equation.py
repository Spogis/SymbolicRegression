import sympy as sp
import math

def calculate_y(x):
    y = 2.56022041817836 - 5.25991474204195*x
    return float(y)

if __name__ == "__main__":
    # Test the function with example values
    x = 0.0  # Example values
    result = calculate_y(x)
    print(f"y = {result}")
