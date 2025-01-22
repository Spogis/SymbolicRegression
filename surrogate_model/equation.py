import sympy as sp
import math

def calculate_Nu(Re, Pr):
    Nu = 0.0349978389621261*math.sqrt(Pr**0.6667161*Re)
    return float(Nu)

if __name__ == "__main__":
    # Test the function with example values
    Re, Pr = [239696.96969696973, 3.941666666666667]  # Example values
    result = calculate_Nu(Re, Pr)
    print(f"Nu = {result}")
