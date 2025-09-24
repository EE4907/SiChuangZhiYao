import os
import csv

output_dir = 'output'
data_dir = 'csv_processed'
result_csv = 'result_998.csv'

results = []

for folder_index in range(1, 23):
    folder_path = os.path.join(output_dir, str(folder_index))
    csv_filename = f"split_{folder_index + 1:03d}.csv"
    csv_path = os.path.join(data_dir, csv_filename)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
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

                if pred > 0.998:
                    row = data_rows[molecule_index]
                    row_with_pred = [pred] + row
                    results.append(row_with_pred)

                molecule_index += 1

results.sort(key=lambda x: x[0], reverse=True)

with open(result_csv, 'w', newline='', encoding='utf-8') as out_f:
    writer = csv.writer(out_f)
    writer.writerow(['prediction'] + header)
    writer.writerows(results)

print(f"Total results with prediction > 0.998: {len(results)} written to {result_csv}")