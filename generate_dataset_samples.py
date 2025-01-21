import numpy as np
import pandas as pd

# Função para gerar o dataset
def generate_dataset(equation_type, a, b, c, noise_level=0.1, num_points=100, output_file='quadratic.xlsx'):

    x = np.linspace(0, 10, num_points)

    if equation_type == 'linear':
        y = a * x + b
    elif equation_type == 'quadratic':
        y = a*x**2 +b*x + c
    elif equation_type == 'power':
        y = a**x + b

    # Adiciona ruído
    noise = np.random.normal(0, noise_level, size=num_points)
    y_noisy = y + noise

    # Cria um DataFrame
    df = pd.DataFrame({'y': y_noisy, 'x': x})

    # Salva o DataFrame em Excel
    df.to_excel(output_file, index=False)
    print(f"Dataset gerado e salvo em '{output_file}'.")

# Input do usuário
equation_type = "power"
noise_level = 0.001
num_points = 100
#a, b, c = -5.26, 2.58, 0.00           #linear
#a, b, c = 2.22, -5.37, 3.22            #quadratic
a, b, c = 3.1525, -7.15, 0.00            #power

output_file = "datasets/power.xlsx"

# Gera o dataset
generate_dataset(equation_type, a, b, c, noise_level, num_points, output_file)
