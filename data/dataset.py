import os
import pandas as pd

input_folder = 'data'
output_folder = 'data_processed'

for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)
    output_filename = os.path.splitext(filename)[0] + '.txt'
    output_path = os.path.join(output_folder, output_filename)

    df = pd.read_csv(input_path)
    smiles_series = df['SMILES']
    smiles_series.to_csv(output_path, index=False, header=False)
    print(f"Processed: {filename}")


