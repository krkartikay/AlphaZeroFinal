from torch.utils.data import Dataset
import torch

class ChessDataset(Dataset):
    def __init__(self, num_files, file_lengths):
        self.num_files = num_files
        self.file_lengths = file_lengths
        self.cumulative_lengths = [sum(file_lengths[:i + 1]) for i in range(num_files)]
        self.current_file_idx = -1  # Initialize to an invalid index
        self.current_file_data = None  # Placeholder for data from the current file

    def __len__(self):
        return self.cumulative_lengths[-1]

    def load_file(self, file_idx):
        if file_idx != self.current_file_idx:
            with open(f'training_data/data_{file_idx+1}.pt', 'rb') as file:
                self.current_file_data = torch.load(file)
            self.current_file_idx = file_idx

    def __getitem__(self, idx):
        # Find the file this index belongs to
        file_idx = next(i for i, total in enumerate(self.cumulative_lengths) if total > idx)
        
        # Load file data if necessary
        self.load_file(file_idx)
        
        # Adjust index to be relative to the file
        if file_idx > 0:
            idx -= self.cumulative_lengths[file_idx - 1]

        # Extract the specific data point (adjust this based on your data's structure)
        input_output_label = (self.current_file_data['all_inps'][idx],
                              self.current_file_data['all_outs'][idx],
                              self.current_file_data['all_vals'][idx])
        
        return input_output_label
