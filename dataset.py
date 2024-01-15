from torch.utils.data import Dataset
import torch
import threading

class ChessDataset(Dataset):
    def __init__(self, num_files, file_lengths):
        self.num_files = num_files
        self.file_lengths = file_lengths
        self.cumulative_lengths = [sum(file_lengths[:i + 1]) for i in range(num_files)]
        self.current_file_idx = -1
        self.current_file_data = None
        self.next_file_data = None
        self.lock = threading.Lock()

    def __len__(self):
        return self.cumulative_lengths[-1]

    def load_file(self, file_idx):
        # Load the current file if not already loaded
        if file_idx != self.current_file_idx:
            if self.next_file_data is not None and file_idx == self.current_file_idx + 1:
                with self.lock:
                    # Use the prefetched data
                    self.current_file_data = self.next_file_data
                    self.next_file_data = None
            else:
                # Load the current file normally if prefetching is not applicable
                with open(f'training_data/data_{file_idx+1}.pt', 'rb') as file:
                    self.current_file_data = torch.load(file)

            self.current_file_idx = file_idx
            # Start prefetching the next file
            next_file_idx = file_idx + 1
            if next_file_idx + 1 < self.num_files:
                threading.Thread(target=self.prefetch_file, args=(next_file_idx,)).start()

    def prefetch_file(self, file_idx):
        with open(f'training_data/data_{file_idx+1}.pt', 'rb') as file:
            next_file_data = torch.load(file)
        with self.lock:
            self.next_file_data = next_file_data

    def __getitem__(self, idx):
        # Find the file this index belongs to
        file_idx = next(i for i, total in enumerate(self.cumulative_lengths) if total > idx)
        
        # Load file data if necessary
        self.load_file(file_idx)
        
        # Adjust index to be relative to the file
        if file_idx > 0:
            idx -= self.cumulative_lengths[file_idx - 1]

        # Extract the specific data point
        input_output_label = (self.current_file_data['all_inps'][idx],
                              self.current_file_data['all_outs'][idx],
                              self.current_file_data['all_vals'][idx])
        
        return input_output_label
