# HCAR1 Antagonist Discovery Platform

This repository contains the codebase for an integrated AI-driven pipeline designed for the discovery of novel HCAR1 (Hydroxycarboxylic Acid Receptor 1) antagonists. The project combines multiple state-of-the-art computational methods, including deep learning-based affinity prediction and molecular docking, to efficiently screen large compound libraries.

![1760149814563](image/readme/1760149814563.png)Repository Structure

```
├── data/					# Directory for raw and processed datasets

│   ├── csv_processed/		# Processed csv files of data

│   ├── data/				# The original data

│   ├── featured_dataset/	# Processed data of transformerCPI

│   └── data_processed/		# Processed txt files of data

├── GeminiMol/			# Module for molecular similarity and agonist exclusion

├── TEFDTA/				# Module for deep learning-based affinity prediction

│   └── main_final.py		# Main script for training and prediction

└── transformerCPI/			# Module for transformer-based affinity prediction

       ├── GPCR/
   
       ├── main.py			# Main script for running predictions
   
       │   └── mol_featurizer.py	# Script for featurizing molecular data
   
       └── model/				# Directory for saved model weights
```

## Key Components

### 1. TEFDTA - Affinity Prediction

This module is based on the TEFDTA model[1], which uses a deep learning architecture to predict binding affinity between compounds (SMILES) and proteins (FASTA sequences).

To train the model, run:

```
python TEFDTA/main_final.py --name tefdta_model
```

The model will be saved as  `TEFDTA/model/tefdta_model`.

To perform the affinity prediction, run:

```
python TEFDTA/main_final.py --mode predict \
                            --name tefdta_results \
                            --smiles_file path/to/smiles_list.txt \
                            --protein_fasta path/to/target_protein.fasta
```

* `--name`: Name of the output file.
* `--smiles_file`: A text file containing one SMILES string per line.
* `--protein_fasta`: A file containing the target protein's sequence in FASTA format.

The results will be saved as `TEFDTA/result/tefdta_results`.

### 2. transformerCPI - Affinity Prediction

This module is based on the tranformerCPI [2] model, which provides the other affinity prediction method using a Transformer-based architecture on pre-featurized molecular and protein data.

To featurize the data, run:

```
python transformerCPI_SICBC\GPCR\mol_featurizer.py
```

The results will be saved at `data/featured_dataset`.

To perform the affinity prediction, run:

```

python transformerCPI_SICBC\GPCR\main.py

```

The results will be saved at `transformerCPI/output`.

### 3. GeminiMol - Similarity Search for Agonist Exclusion

This module is based on the GeminiMol [3] model, which is used for molecular comparison and analysis. Its `PharmProfiler.py` tool is critical for ensuring selectivity by filtering out compounds that are highly similar to known HCAR1 agonists, thus reducing the risk of generating unintended activators.

Please refer to the dedicated documentation for the GeminiMol module for detailed usage instructions.

### 4. Others

Some other programs are used to process data and integrate final results.

`GeminiMol\Search_Molecular.ipynb`: used to cluster molecules for docking.

`data\dataset.py`: extracting SMILES as txt from the original data.

`data\filter_csv.py`: filtering invalid or unused smiles from the original data.

## Setup and Installation

### 1. Create and activate environment

```
conda create -n hcar_ai python=3.9
conda activate hcar_ai
```

### 2. Install dependencies

The necessary packages are below:

```
numpy==1.21.5
pandas==1.3.5
scipy==1.10.1
rdkit==2021.03.2
gensim>=3.4.0
chembl_webresource_client.new_client

torch==1.13.1+cu116
torchvision==0.14.1+cu116

dgl-cu116==1.1.1
dglgo==0.0.2
dgllife==0.3.2

matplotlib
six
scikit-learn
oddt
autogluon==0.8.1
tqdm
```

## License

The original code written by our team for this project is licensed under the MIT License.

This project incorporates third-party components under their own licenses:

- The **TransformerCPI** model code is licensed under the Apache License 2.0.
- The **GeminiMol** model code is licensed under the Academic Free License v3.0.

Please look in the folders to see their individual licenses or see the full content of the licenses at the main folder.

## Contributors

This project is developed by:

* Lin Wuyi (Team Lead)
* Chen Keyu
* Wang Mohan
* Chen Yixuan

ShanghaiTech University, Undergraduate Students

## References

[1] Zongquan Li, Pengxuan Ren, Hao Yang, Jie Zheng, Fang Bai, TEFDTA: a transformer encoder and fingerprint representation combined prediction method for bonded and non-bonded drug–target affinities, Bioinformatics, Volume 40, Issue 1, January 2024, btad778, [https://doi.org/10.1093/bioinformatics/btad778](https://doi.org/10.1093/bioinformatics/btad778)

[2] Lifan Chen, Xiaoqin Tan, Dingyan Wang, Feisheng Zhong, Xiaohong Liu, Tianbiao Yang, Xiaomin Luo, Kaixian Chen, Hualiang Jiang, Mingyue Zheng, TransformerCPI: improving compound–protein interaction prediction by sequence-based deep learning with self-attention mechanism and label reversal experiments, Bioinformatics, Volume 36, Issue 16, 15 August 2020, Pages 4406–4414, [https://doi.org/10.1093/bioinformatics/btaa524](https://doi.org/10.1093/bioinformatics/btaa524)

[3] Lin Wang, Shihang Wang, Hao Yang, Shiwei Li, Xinyu Wang, Yongqi Zhou, Siyuan Tian, Lu Liu, Fang Bai, Conformational Space Profiling Enhances Generic Molecular Representation for AI-Powered Ligand-Based Drug Discovery, Advanced Science , 2024; doi: 10.1002/advs.202403998
