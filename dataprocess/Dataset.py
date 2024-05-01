import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
from gensim.models import Word2Vec
from config.Trainingconfig import TrainingConfig

config = TrainingConfig()

class TextDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        
        self.word_vectors, self.word_to_idx = self.load_word2vec_model(config.word2vec_vocab_path)
        self.data = self.load_data(data_path)
        self.voc_size = len(self.word_vectors)

    def load_data(self, directory_path, max_len=512):
        """
        Load data from the directory, applying padding or truncation to each line.
        
        Args:
            directory_path (str): Path to the directory containing text files.
            max_len (int): Maximum length of sequences after padding/truncating.

        Returns:
            list: List of tuples containing processed text data and labels.
        """
        samples = []
        def process_text(text, max_len, pad_value=0):
            """ Truncate or pad the text indices to a fixed length """
            indices = [self.word_to_idx.get(word, self.word_to_idx.get('UNK', 0)) for word in text.split()]
            # Apply truncation
            if len(indices) > max_len:
                indices = indices[:max_len]
            # Apply padding
            elif len(indices) < max_len:
                indices += [pad_value] * (max_len - len(indices))
            return indices

        # Read files from the directory
        for filename in os.listdir(directory_path):
            full_path = os.path.join(directory_path, filename)
            if os.path.isfile(full_path) and full_path.endswith('.txt'):
                with open(full_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        text, label = line.strip().split(' ??? ')
                        processed_indices = process_text(text, max_len)
                        samples.append((processed_indices, int(label)))

        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        try:
            text_indices, label = self.data[idx]
            text_tensor = torch.tensor(text_indices, dtype=torch.long)
            label_tensor = torch.tensor(label, dtype=torch.long)
            return text_tensor, label_tensor
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            # Return a default tensor in case of error
            return torch.zeros(1, dtype=torch.long), torch.tensor(0, dtype=torch.long)


    def load_word2vec_model(self, model_path):
        model = Word2Vec.load(model_path)
        word_vectors = model.wv
        word_to_idx = {word: idx for idx, word in enumerate(word_vectors.index_to_key)}
        word_vectors = {word: word_vectors[word] for word in word_vectors.index_to_key}
        word_vectors["UNK"] = np.zeros(len(word_vectors))  # Adding an 'unknown' vector
        return word_vectors, word_to_idx
    
    def split_dataset(self, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
        total_size = len(self)
        train_size = int(total_size * train_ratio)
        valid_size = int(total_size * valid_ratio)
        test_size = total_size - train_size - valid_size
        train_dataset, valid_dataset, test_dataset = random_split(self, [train_size, valid_size, test_size])
        return train_dataset, valid_dataset, test_dataset

