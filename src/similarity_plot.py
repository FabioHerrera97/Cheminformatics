from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class PlotSimilarity:
    def __init__(self, train_dataset, validation_dataset, test_dataset):

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

    def generate_tsne_molecular_similarities(self, smiles_column):
        
        def compute_fingerprint(smiles):
            molecule = Chem.MolFromSmiles(smiles)
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=1024)
            return fingerprint
        fingerprints = [compute_fingerprint(smiles) for smiles in self.train_dataset[smiles_column]]
        fingerprints.extend([compute_fingerprint(smiles) for smiles in self.validation_dataset[smiles_column]])
        fingerprints.extend([compute_fingerprint(smiles) for smiles in self.test_dataset[smiles_column]])

        similarity_matrix = np.zeros((len(fingerprints), len(fingerprints)))

        for i in range (len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

        tsne = TSNE(n_components=2, random_state=42)
        tsne_components = tsne.fit_transform(similarity_matrix)

        train_components = tsne_components[:len(self.train_dataset[smiles_column])]
        validation_components = tsne_components[len(self.train_dataset[smiles_column]):len(self.validation_dataset[smiles_column]) + len(self.train_dataset[smiles_column])]
        test_components = tsne_components[len(self.validation_dataset[smiles_column]) + len(self.train_dataset[smiles_column]):]
        plt.figure(figsize=(10,10))

        plt.scatter(train_components[:,0], train_components[:,1], c='red', label='Train dataset', s=[2]*len(train_components))
        plt.scatter(validation_components[:,0], validation_components[:,1], c='blue', label='Validations dataset', s=[2]*len(validation_components))
        plt.scatter(test_components[:,0], test_components[:,1], c='green', label='Test dataset', s=[2]*len(test_components))

        plt.legend()
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('Molecular Similarity with t-SNE')
        plt.show()