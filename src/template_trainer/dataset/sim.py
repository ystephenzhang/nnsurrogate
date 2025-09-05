from .base import BaseDataPipe
import glob, os
import pdb

class SimRewardDataPipe(BaseDataPipe):
    '''
    Use data from .pt like 
    torch.save({
        'x_s': torch.tensor(static_array),
        'x_t': torch.tensor(tunable_array),
        'y': torch.tensor(labels_array)
    }, "dataset.pt")
    
    #TODO add random noise
    '''
    def _get_file_list(self):
        """
        Get the list of files in the data directory.

        Returns:
            list: List of file paths.
        """
        return glob.glob(os.path.join(self.data_dir, "*.pt"))
    
    def _read_path(self, file_path):
        """
        Read the simulation data from a .pt file.
        
        Args:
            file_path (str): Path to the .pt file.
            
        Returns:
            tuple: ((x_s, x_t, y), length) where length is the batch dimension size
        """
        import torch
        
        data = torch.load(file_path, map_location='cpu', weights_only=True)
        organized = tuple(torch.stack(tensors) for tensors in zip(*data))
        
        # Get the number of samples from the first dimension of the first tensor
        length = organized[0].shape[0]
        
        #return (torch.cat((x_s, x_t), dim=-1), y), length 
        return organized, length