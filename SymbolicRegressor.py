import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import random
from sklearn.metrics import r2_score
from pysr import PySRRegressor

random.seed(42)
np.random.seed(42)

# 1. Load the dataset
data = pd.read_excel("datasets/quadratic.xlsx")

# Limpeza dos nomes das colunas
data.columns = (
    data.columns
    .str.strip()  # Remove espaços extras no início e no final
    .str.replace(r'[^a-zA-Z0-9_]', '', regex=True)  # Remove caracteres inválidos, mantendo letras, números e '_'
    .str.replace(' ', '_')  # Substitui espaços por '_'
)

# 2. Define y (dependent variable) and X (independent variables)
y = data.iloc[:, 0].values  # The first column is always the dependent variable
X = data.iloc[:, 1:].values  # The other columns are the independent variables

temp_directory = "temp_pysr_files"
if not os.path.exists(temp_directory):
    os.makedirs(temp_directory)

# 3. Configure the PySR model
model = PySRRegressor(
    populations=10,
    niterations=100,
    unary_operators=["exp", "log", "sqrt"],  # Unary operators
    binary_operators=["+", "-", "*", "/", "^"],  # Binary operators
    early_stop_condition=(
        "stop_if(loss, complexity) = loss < 1e-2 && complexity < 10"
        # Stop early if we find a good and simple equation
    ),
    elementwise_loss="myloss(x, y) = sum(abs.(x .- y) ./ abs.(x)) ",  # MAPE

    temp_equation_file=True,
    tempdir=temp_directory,
    delete_tempfiles=False,
    verbosity=1,  # Verbosity level

    warm_start=False,  # Não utilizar o modelo da rodada anterior
)

# 4. Fit the model to the data
print("Training the model...")
model.fit(X, y)

# 5. Get the best generated equation
best_equation_sympy = model.sympy()
print("Generated equation:", best_equation_sympy)

# Access the generated surrogate_model and apply early_stop_condition
equations_df = model.equations_

# Apply the filter for early stopping condition (loss < 1e-2 and complexity < 10)
filtered_equations = equations_df[
    (equations_df["loss"] < 1e-2) & (equations_df["complexity"] < 10)
    ]

# If there are surrogate_model that meet the early stop condition, choose the one with the lowest loss
if not filtered_equations.empty:
    best_filtered_equation = filtered_equations.sort_values("loss").iloc[0]
    best_equation_sympy = sp.sympify(best_filtered_equation["equation"])  # Convert string to sympy equation
    print("Using filtered equation with the lowest loss and complexity:")
    print(best_equation_sympy)
else:
    print("No equation satisfies the early stop condition.")
    print("Using the best generated equation:")
    print(best_equation_sympy)

# 6. Simplify and rename variables in the equation
# Variable names based on dataset columns
variables = sp.symbols(data.columns[1:])  # The independent variables
simplified_equation = sp.simplify(best_equation_sympy.subs(dict(zip(["x" + str(i) for i in range(len(variables))], variables))))

# Generate LaTeX and display
latex_expression = sp.latex(simplified_equation)
independent_variables_name=data.columns[0]
latex_equation = f"{independent_variables_name}={latex_expression}"

# Display the surrogate_model in the console
print("Simplified equation with renamed variables:", simplified_equation)
print("Simplified equation in LaTeX:", latex_equation)

# 7. Save the simplified equation as an image
# Create the figure with the adjusted size
plt.figure(figsize=(12,5))
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
r2 = r2_score(y, y_pred)
print(f"r2 score= {r2}")

# 10. Calculate absolute and relative error
results_df["Absolute_Error"] = np.abs(results_df["Real"] - results_df["Predict"])
results_df["Relative_Error_%"] = 100 * results_df["Absolute_Error"] / results_df["Real"]

# 11. Save the DataFrame to an Excel file
results_df.to_excel("datasets/Real_vs_Predict.xlsx", index=False)

# 12. Display a preview of the DataFrame
print(results_df.head())

# 13. Generate the Python file with the function for the equation
def generate_equation_file(equation, variables, independent_variable, valores_primeira_linha):
    equation_str = sp.pycode(equation)

    # Create the Python file that will contain the function to calculate y
    with open("surrogate_model/equation.py", "w") as f:
        f.write('import sympy as sp\n')
        f.write('import math\n\n')
        f.write(f'def calculate_{independent_variable}({", ".join(variables)}):\n')
        f.write(f'    {independent_variable} = {equation_str}\n')
        f.write(f'    return float({independent_variable})\n\n')

        f.write('if __name__ == "__main__":\n')
        f.write('    # Test the function with example values\n')
        f.write(f'    {", ".join([var for var in variables])} = {list(valores_primeira_linha)}  # Example values\n')
        f.write(f'    result = calculate_{independent_variable}(' + ", ".join([var for var in variables]) + ')\n')
        f.write('    print(f"Result for input values: y = {result}")\n')

    print("File 'equation.py' generated successfully!")


# Generate the file with the equation
valores_primeira_linha = data.iloc[0, 1:].values
generate_equation_file(simplified_equation, data.columns[1:], data.columns[0], valores_primeira_linha)
