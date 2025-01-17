import sympy as sp
import math

def calculate_y(Re, Pr):
    equation = 0.03500041*math.sqrt(Pr**0.666656920762096*Re)
    return float(equation)

if __name__ == "__main__":
    # Test the function with example values
    Re_test, Pr_test = [1] * 2  # Example values
    result = calculate_y(Re_test, Pr_test)
    print(f"Result for input values: y = {result}")
