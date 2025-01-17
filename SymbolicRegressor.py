import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import random
from pysr import PySRRegressor

random.seed(42)
np.random.seed(42)

# 1. Load the dataset
data = pd.read_excel("datasets/nusselt.xlsx")

# 2. Define y (dependent variable) and X (independent variables)
y = data.iloc[:, 0].values  # The first column is always the dependent variable
X = data.iloc[:, 1:].values  # The other columns are the independent variables

# 3. Configure the PySR model
model = PySRRegressor(
    populations=8,  # Assuming we have 4 cores, this means 2 populations per core, so one is always running
    niterations=1000,  # Generations between migrations
    unary_operators=["exp", "log", "log10", "sqrt"],  # Unary operators
    binary_operators=["+", "-", "*", "/", "^"],  # Binary operators
    elementwise_loss="loss(x, y) = (x - y)^2",  # Loss function
    verbosity=1,  # Verbosity level
)

# 4. Fit the model to the data
print("Training the model...")
model.fit(X, y)

# 5. Get the best generated equation
best_equation_sympy = model.sympy()
print("Generated equation:", best_equation_sympy)

# 6. Simplify and rename variables in the equation
# Variable names based on dataset columns
variables = sp.symbols(data.columns[1:])  # The independent variables
simplified_equation = sp.simplify(best_equation_sympy.subs(dict(zip(["x" + str(i) for i in range(len(variables))], variables))))

# Generate LaTeX and display
latex_equation = sp.latex(simplified_equation)

# Display the equations in the console
print("Simplified equation with renamed variables:", simplified_equation)
print("Simplified equation in LaTeX:", latex_equation)

# 7. Save the simplified equation as an image
# Create the figure with the adjusted size
plt.figure(figsize=(10,8))
plt.text(
    0.5,
    0.5,
    f"${latex_equation}$",
    fontsize=18,
    ha="center",
    va="center",
)
plt.axis("off")
plt.savefig("assets/BestEquation.png")
plt.show()

# 8. Generate predicted values
y_pred = model.predict(X)

# 9. Create a DataFrame with real and predicted values
results_df = pd.DataFrame({
    "Real": y,
    "Predict": y_pred
})

# 10. Calculate absolute and relative error
results_df["Absolute_Error"] = np.abs(results_df["Real"] - results_df["Predict"])
results_df["Relative_Error_%"] = 100 * results_df["Absolute_Error"] / results_df["Real"]

# 11. Save the DataFrame to an Excel file
results_df.to_excel("datasets/Real_vs_Predict.xlsx", index=False)

# 12. Display a preview of the DataFrame
print(results_df.head())

# 13. Delete 'hall_of_fame' files generated by PySR
hall_of_fame_files = glob.glob("hall_of_fame*")
for file in hall_of_fame_files:
    try:
        os.remove(file)
        print(f"File {file} deleted successfully.")
    except Exception as e:
        print(f"Error deleting {file}: {e}")

# 14. Generate the Python file with the function for the equation
def generate_equation_file(equation, variables):
    equation_str = sp.pycode(equation)

    # Create the Python file that will contain the function to calculate y
    with open("equations/equation.py", "w") as f:
        f.write('import sympy as sp\n')
        f.write('import math\n\n')
        f.write(f'def calculate_y({", ".join(variables)}):\n')
        f.write(f'    equation = {equation_str}\n')
        f.write('    return float(equation)\n\n')

        f.write('if __name__ == "__main__":\n')
        f.write('    # Test the function with example values\n')
        f.write(f'    {", ".join([var + "_test" for var in variables])} = [1] * {len(variables)}  # Example values\n')
        f.write('    result = calculate_y(' + ", ".join([var + "_test" for var in variables]) + ')\n')
        f.write('    print(f"Result for input values: y = {result}")\n')

    print("File 'equation.py' generated successfully!")

# Generate the file with the equation
generate_equation_file(simplified_equation, data.columns[1:])
