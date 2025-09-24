import os
import csv

output_dir = 'output'
data_dir = 'csv_processed'
result_csv = 'result_600.csv'

results = []

for folder_index in range(1, 23):
    folder_path = os.path.join(output_dir, str(folder_index))
    csv_filename = f"split_{folder_index + 1:03d}.csv"
    csv_path = os.path.join(data_dir, csv_filename)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data_rows = list(reader)

    molecule_index = 0

    txt_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.txt')],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    for txt_file in txt_files:
        txt_path = os.path.join(folder_path, txt_file)
        with open(txt_path, 'r') as f:
            for line in f:
                try:
                    pred = float(line.strip())
                except ValueError:
                    continue

                if molecule_index >= len(data_rows):
                    break

                if pred > 0.600:
                    row = data_rows[molecule_index]
                    smiles = row.get('SMILES', '')
                    # cluster = row.get('Cluster', '')
                    results.append((smiles, pred))

                molecule_index += 1

results.sort(key=lambda x: x[1], reverse=True)
print(f"Total results with prediction > 0.600: {len(results)}")

with open(result_csv, 'w', newline='') as out_f:
    writer = csv.writer(out_f)
    writer.writerow(['smiles', 'prediction'])
    writer.writerows(results)
