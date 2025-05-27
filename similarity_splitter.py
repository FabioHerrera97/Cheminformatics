import argparse
from deepmol.splitters import SimilaritySplitter
from deepmol.loaders import CSVLoader
import pandas as pd

def deepmol_to_csv(dataset, name):
    df = pd.DataFrame({
        'Canonical_smiles': dataset.smiles,
        'Toxicity': dataset.y.flatten()
    })
    df.to_csv(f'{name}.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description='Split dataset based on molecular similarity.')
    parser.add_argument('--input_file', help='Path to input CSV file')
    parser.add_argument('--smiles_col', required=True, help='Column name containing SMILES strings')
    parser.add_argument('--similarity_threshold', type=float, required=True, 
                       help='Similarity threshold for splitting (0.0 to 1.0)')
    parser.add_argument('--train_frac', type=float, default=0.7, 
                       help='Fraction of data for training set (default: 0.7)')
    parser.add_argument('--valid_frac', type=float, default=0.1, 
                       help='Fraction of data for validation set (default: 0.1)')
    parser.add_argument('--test_frac', type=float, default=0.2, 
                       help='Fraction of data for test set (default: 0.2)')
    parser.add_argument('--label_col', default='Toxicity', 
                       help='Column name containing labels (default: Toxicity)')
    
    args = parser.parse_args()

    loader = CSVLoader(
        dataset_path=args.input_file,
        smiles_field=args.smiles_col,
        labels_fields=[args.label_col]
    )
    dataset = loader.create_dataset()

    similarity_splitter = SimilaritySplitter()
    train_dataset, validation_dataset, test_dataset = similarity_splitter.train_valid_test_split(
        dataset,
        frac_train=args.train_frac,
        frac_valid=args.valid_frac,
        frac_test=args.test_frac,"
        homogenous_threshold=args.similarity_threshold
    )

    print('Original dataset size:', len(dataset.smiles))
    print('Train set size:', len(train_dataset.smiles))
    print('Validation set size:', len(validation_dataset.smiles))
    print('Test set size:', len(test_dataset.smiles))

    deepmol_to_csv(train_dataset, 'train_data')
    deepmol_to_csv(validation_dataset, 'validation_data')
    deepmol_to_csv(test_dataset, 'test_data')

    print('Splitting completed successfully!')

if __name__ == '__main__':
    main()
