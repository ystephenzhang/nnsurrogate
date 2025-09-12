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
        target = self.cfg.get("success_target", "score")
        preproc = self.cfg.get("preproc_cost", None)
        
        first_tensor, second_tensor, third_tensor = organized
        if third_tensor.shape[1] == 3:
            if target == 'score':
                # Keep dim 0 and 2 of the third tensor
                processed_third = torch.stack([third_tensor[:, 0], third_tensor[:, 2]], dim=1)
            elif target == 'error':
                # Keep dim 1 and 2 of the third tensor
                processed_third = torch.stack([third_tensor[:, 1], third_tensor[:, 2]], dim=1)
        else:
            processed_third = third_tensor
            
        if preproc:
            if 'heat_1d' in file_path or 'euler_1d' in file_path:
                cfl = second_tensor[:, 0]
                n_space = second_tensor[:, 1]
                processed_third[:, 1] /= (n_space * n_space * n_space / cfl)
            elif "transient" in file_path:
                n_space = second_tensor[:, 1]
                cfl = second_tensor[:, 0]
                processed_third[:, 1] /= (n_space * n_space * n_space / cfl)
        organized = (first_tensor, second_tensor, processed_third)
        
        # Get the number of samples from the first dimension of the first tensor
        length = organized[0].shape[0]
        #return (torch.cat((x_s, x_t), dim=-1), y), length 
        return organized, length