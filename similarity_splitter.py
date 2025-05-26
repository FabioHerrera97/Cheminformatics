from deepmol.splitters import SimilaritySplitter
from deepmol.loaders import CSVLoader
import pandas as pd 

loader = CSVLoader(dataset_path='cleaned_data.csv',
                   smiles_field='Canonical_smiles',
                   labels_fields=['Toxicity']
                   )


dataset = loader.create_dataset()

similarity_splitter = SimilaritySplitter()

train_dataset, validation_dataset, test_dataset = similarity_splitter.train_valid_test_split(dataset, frac_train=0.7, frac_valid=0.1, frac_test=0.2, homogenous_threshold=0.9)

print("Smiles shape:", len(dataset.smiles))
print("Labels shape:", dataset.y.shape)

def deepmol_to_csv(dataset, name):
    df = pd.DataFrame({
        'Canonical_smiles': dataset.smiles,
        'Toxicity': dataset.y.flatten()
    })
    df.to_csv(f'{name}.csv', index=False)

deepmol_to_csv(train_dataset, 'train_data')
deepmol_to_csv(validation_dataset, 'validation_data')
deepmol_to_csv(test_dataset, 'test_data')

print('The data was splitted properly!')
print('Train set size:', len(train_dataset.smiles))
print('Validation set size:', len(validation_dataset.smiles))
print('Test set size:', len(test_dataset.smiles))
