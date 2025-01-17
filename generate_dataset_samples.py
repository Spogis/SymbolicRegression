import numpy as np
import pandas as pd

# Função para gerar o dataset
def generate_dataset(equation_type, noise_level=0.1, num_points=100, output_file='quadratic.xlsx'):
    """
    Gera um dataset com dados de x, y e ruído baseado no tipo de equação especificada.

    Parameters:
        equation_type (str): Tipo de equação ('linear', 'quadratic', 'sine', etc.)
        noise_level (float): Nível de ruído a ser adicionado
        num_points (int): Número de pontos no dataset
        output_file (str): Nome do arquivo de saída em Excel
    """
    x = np.linspace(0, 10, num_points)

    if equation_type == 'linear':
        y = 2 * x + 1
    elif equation_type == 'quadratic':
        y = x**2 - 5 * x + 4
    elif equation_type == 'sine':
        y = np.sin(x + 2.0)
    elif equation_type == 'exponential':
        y = np.exp(0.3 * x)
    else:
        raise ValueError("Tipo de equação não suportado. Escolha entre 'linear', 'quadratic', 'sine', ou 'exponential'.")

    # Adiciona ruído
    noise = np.random.normal(0, noise_level, size=num_points)
    y_noisy = y + noise

    # Cria um DataFrame
    df = pd.DataFrame({'y': y_noisy, 'x': x})

    # Salva o DataFrame em Excel
    df.to_excel(output_file, index=False)
    print(f"Dataset gerado e salvo em '{output_file}'.")

# Input do usuário
equation_type = "sine"
noise_level = 0.001
num_points = 100
output_file = "datasets/sine.xlsx"

# Gera o dataset
generate_dataset(equation_type, noise_level, num_points, output_file)
