import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

class LLMsEncoding:
    
    def featurize(smiles_list, model_name, max_lenth, padding=True, trust_remote_code=True):
        """
    Featurizes a list of SMILES strings using the pretrained models.

    Parameters:
    -----------
    smiles_list : list of str
        A list of SMILES strings representing molecular structures.
    model_name : str
            The name of the pre-trained model to use for featurization.
    max_length : int, optional (default=512)
            The maximum length of the tokenized input. Inputs longer than this will be truncate
    padding : bool, optional (default=True)
        Whether to pad the tokenized inputs to the same length. If True, the tokenizer will
        pad the inputs to the maximum sequence length in the batch. 
    trust_remote_code : bool, optional (default=False)
            Whether to trust custom code in the model repository (required for MolFormer).

    Returns:
    --------
    numpy.ndarray
        A 2D NumPy array of shape (number_of_smiles, max_lenth), where each row corresponds to
        the embedding of a SMILES string from the input list.

        """

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        chemberta = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=trust_remote_code)

        embeddings_cls = torch.zeros(len(smiles_list), max_lenth)

        with torch.no_grad():
            for i, smiles in enumerate(tqdm(smiles_list)):
                encoded_input = tokenizer(smiles, return_tensors='pt', padding=padding, truncation=True)
                model_output = chemberta(**encoded_input)

                embedding = model_output[0][::, 0, ::]
                embeddings_cls[i] = embedding

        return pd.DataFrame(embeddings_cls.numpy())

def main():
    
    smiles_list = ['CCO', 'C1=CC=CC=C1', 'CC(=O)O']

    print('Performing ChemBERTa embedding \n')
    chemberta_embeddings = LLMsEncoding.featurize(smiles_list, 'DeepChem/ChemBERTa-77M-MTR', 600)
    print(f'{chemberta_embeddings}\n')
    print('Performing MolFormer \n')
    molformer_embeddings = LLMsEncoding.featurize(smiles_list, 'ibm/MolFormer-XL-both-10pct', 2362)
    print(f'{molformer_embeddings}\n')
    print('Performing SELFormer \n')
    selformer_embeddings = LLMsEncoding.featurize(smiles_list, 'HUBioDataLab/SELFormer', 800)
    print(selformer_embeddings)

#main() # uncomment this line tu run the test

if __name__ == 'main':
    main()