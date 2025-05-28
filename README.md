# Cheminformatics

This repository contains a tutotrial on representation of small molecules representation and its application in case of study to predict toxicity in small molecules.

## Prerequisites
- **Git** installed on your system. [Download Git here](https://git-scm.com/downloads).
- **Anaconda** or **Miniconda** for environment management. [Download here](https://docs.conda.io/en/latest/miniconda.html).
- SSH key configured for your GitHub account. [GitHub SSH setup guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).

---

## Installation Steps

### 1. Clone the Repository (SSH)
Open a terminal and run:
```bash
git clone git@github.com:FabioHerrera97/Cheminformatics.git
cd Cheminformatics
```
### 2. Set Up the Conda Environment

```bash
conda create --name cheminformatics python=3.11

conda activate cheminformatics

pip install -r requirements.txt
```

### 3. Data cleaning and molecular standardization

Run the first part of the [Tutorial_predicting_toxicity_small_molecules](https://github.com/FabioHerrera97/Cheminformatics/blob/main/Tutorial_predicting_toxicity_small_molecules.ipynb). This notebook contains the code and detailed instructions to run the project.

### 4. Data split
The data needs to be splited into a train, validation and test dataset. The similarity splitter from deepmol was used. The data is split in a way that the similarity between the molecules in each set is below a certain threshold. This is useful when we want to make sure that the molecules in the validation and test sets are either not too similar or similar to the molecules in the training set. Due to imcompatibility with other cheminformatic labraries deepmol needs to be run in a different environment. 

```bash
conda create --name deepmol python=3.11

conda activate deepmol

pip install deepmol[all]
```

The clean and standardized data needs to be provided. Check the first part of the tutorial [Tutorial_predicting_toxicity_small_molecules](https://github.com/FabioHerrera97/Cheminformatics/blob/main/Tutorial_predicting_toxicity_small_molecules.ipynb)

```bash
python similarity_splitter.py --input_file data/cleaned_data.csv --smiles_col Canonical_smiles --similarity_threshold 0.7
```
#### Required Arguments

#### Argument	Description

```--input_file:```	  Path to the input CSV file containing your molecular data

```--smiles_col:```	  Name of the column in your CSV that contains the SMILES strings

```--similarity_threshold:```	  Similarity threshold (0.0 to 1.0) for splitting. Molecules with similarity above this threshold won't be placed in different sets.

#### Optional Arguments

#### Argument	Default	Description

```--train_frac:```	  0.7	- Fraction of data to allocate to the training set

```--valid_frac:```	  0.1	- Fraction of data to allocate to the validation set

```--test_frac:```	  0.2	- Fraction of data to allocate to the test set

```--label_col:```	'Toxicity' - Name of the column containing the labels/target values

#### Output

The script will generate three CSV files:


```data/train_data.csv:```	  Contains the training set molecules

```data/validation_data.csv:```	  Contains the validation set molecules

```data/test_data.csv:```	  Contains the test set molecules

Each output file will contain two columns:

```Canonical_smiles:```	  The SMILES representation of the molecule

```Toxicity:```	  The label value (column name will match your input label column)

### 6. Complete the representation and modeling part

Go back to the [Tutorial_predicting_toxicity_small_molecules](https://github.com/FabioHerrera97/Cheminformatics/blob/main/Tutorial_predicting_toxicity_small_molecules.ipynb) and follow the instructions.

### Support

For issues or questions, please open an [Issue](https://github.com/FabioHerrera97/Cheminformatics/issues) on GitHub.
