import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import download_images
from tqdm import tqdm
from PIL import Image
from constants import entity_unit_map
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASET_FOLDER = '../dataset/'
train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
sample_test = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test.csv'))
sample_test_out = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test_out.csv'))



class Image_dataset(Dataset):
    DATASET_FOLDER = '/kaggle/input/codesssss/student_resource_3/dataset/'
    
    def __init__(self,entity_data_type,data_folder = 'train'):
        super().__init__()
        
        self.data_folder = data_folder
        self.group_id = []
        self.images = []
        self.entity_data_type = entity_data_type
        self.transforms = transforms.Compose([
            transforms.Resize([300,300]),
        ])


        if(self.data_folder == 'train'):
            self.data = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
            self.data = self.data.loc[self.data['entity_name'] == entity_data_type].reset_index(drop=True)[0:20]
            
            self.value = []
            self.unit = []

        elif(self.data_folder == 'test'):
            self.index = []
            self.data = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
            self.data = self.data.loc[self.data['entity_name'] == entity_data_type].reset_index(drop=True)
            
        elif(self.data_folder == 'sample_test'):
            self.index = []
            self.data = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test.csv'))
            self.data = self.data.loc[self.data['entity_name'] == entity_data_type].reset_index(drop=True)
            
        download_images(self.data['image_link'],f'../{self.data_folder}_images')

        

        for _ in range(len(self.data)):
        
            length = len('https://m.media-amazon.com/images/I/')
            image_path = self.data['image_link'][_][length:]

            if(os.path.exists(f'../{self.data_folder}_images/{image_path}')):
                if(self.data_folder == 'train'):
                    entity_value = self.data['entity_value'][_]
                    print(entity_value)
                    if('[' and ']' and ' to ' and ',' not in entity_value and len(entity_value.split(' ')) == 2 and entity_value.split(' ')[-1] in list(entity_unit_map[entity_data_type])):
                        image = Image.open(f'../{self.data_folder}_images/{image_path}')
                        self.images.append(self.transforms(image))
                        self.group_id.append(self.data['group_id'][_])
                        entity_value = entity_value.split(' ')
                        
                        self.unit.append(entity_value[-1])
                        self.value.append(entity_value[0])
                        
                        
                elif(self.data_folder == 'test' or self.data_folder == 'sample_test'):
                    self.index.append(self.data['index'][_])
                    image = Image.open(f'../{self.data_folder}_images/{image_path}')
                    self.images.append(self.transforms(image))
                    self.group_id.append(self.data['group_id'][_])

    def __len__(self):
        return len(self.images)
 
    def __getitem__(self, index):
        if(self.data_folder == 'train'):
            return self.images[index],self.value[index],self.unit[index]
        elif(self.data_folder == 'test' or self.data_folder == 'sample_test'):
            return self.index[index],self.images[index]
        

# train_dataset = Image_dataset(data_folder='train',entity_data_type='maximum_weight_recommendation')