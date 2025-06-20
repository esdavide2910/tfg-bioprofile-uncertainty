
import os
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset

class DataLoaderWrapper:
    
    class MaxillofacialXRayDataset(Dataset):
    
        def __init__(self, metadata_file, images_dir, transform=None):
            """
            metadata_file: Ruta al fichero de metadatos (CSV u otro formato)
            images_dir: Ruta al directorio de imágenes (entrenamiento o prueba)
            transform: Transformaciones a aplicar a las imágenes (normalización, etc.)
            """
            self.metadata = pd.read_csv(metadata_file)  # Cargar los metadatos
            self.images_dir = images_dir
            self.transform = transform
        
        def __len__(self):
            return len(self.metadata)
            
        def __getitem__(self, idx):
            # Obteniene el nombre de la imagen y su valor desde los metadatos
            img_name = os.path.join(self.images_dir, self.metadata.iloc[idx]['ID'])  # Ajusta según la estructura
            target = self.metadata.iloc[idx]['Age'].astype(np.float32)  # Ajusta según el formato de tus metadatos
            
            # Abre la imagen
            image = Image.open(img_name)
            
            # Aplica transformaciones si es necesario
            if self.transform:
                image = self.transform(image)
            
            return image, target
    
    
    def create_loader(dataset, indices, batch_size, seed_worker, generator):
        subset = Subset(dataset, indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2, 
                        pin_memory=True, worker_init_fn=seed_worker, generator=generator)
    
    
    def __init__(self, data_path, train_transform, default_transform, include_calib=False, batch_size=32):
        
        self.batch_size = batch_size
        
        trainset = self.MaxillofacialXRayDataset(
            metadata_file = data_path + 'metadata_train.csv',
            images_dir = data_path + 'train/',
            transform = train_transform
        )
        
        validset = self.MaxillofacialXRayDataset(
            metadata_file = data_path+ 'metadata_train.csv',  
            images_dir = data_path + 'train/',               
            transform = default_transform                   
        )
        
        testset = self.MaxillofacialXRayDataset(
            metadata_file = data_path+ 'metadata_train.csv',  
            images_dir = data_path + 'train/',               
            transform = default_transform                   
        )
        
    
    def x():
        
        return
        
        
        
        
    
    
    