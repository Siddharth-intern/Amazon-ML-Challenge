import os
import random
import pandas as pd
from transformers import AutoProcessor,AutoModel
from Image_Encoder import Image_encoder
from dataset import Image_dataset
from constants import entity_unit_map
from Value_regressor import regressor
from Classifier import classification
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from Value_lstm import Inference_lstm

device = "cuda" if torch.cuda.is_available() else "cpu"

Image_model = "openai/clip-vit-base-patch32"
entity_types = list(entity_unit_map.keys())
entity_types.sort()

processor = AutoProcessor.from_pretrained(Image_model)
model = AutoModel.from_pretrained("openai/clip-vit-base-patch32",output_hidden_states = True).to(device)
Image_model = Image_encoder(processor=processor,model = model)

BATCH_SIZE = 10

def Inference(dataset,Image_Encoder_model,mode = 'Train'):

    Total_predicted_units = []
    Total_predicted_values = []
    Possible_Entity_units = list(entity_unit_map[dataset.entity_data_type])
    Possible_Entity_units.sort()
    ground_truth_values = []
        
    le = LabelEncoder()
    le.fit(Possible_Entity_units)
    
    for _ in range(0,len(dataset),BATCH_SIZE):
        Image_representation = Image_Encoder_model.features(dataset.images[_:_+BATCH_SIZE])

        if(mode == 'train'):
            Entity_value = dataset.value[_:_+BATCH_SIZE]
            Entity_units = le.transform(dataset.unit[_:_+BATCH_SIZE])
        else:
            Entity_value = [0 for i in range(BATCH_SIZE)]
            Entity_units = le.transform([Possible_Entity_units[0] for i in range(BATCH_SIZE)])
    

        Batch_Predicted_unit = classification(mode,Image_representation,entity_sub_types=Entity_units,entity_data_type = dataset.entity_data_type,possible_entity_units=Possible_Entity_units)
        Batch_Predicted_unit = le.inverse_transform(Batch_Predicted_unit.to('cpu'))
        Batch_predicted_value = Inference_lstm(mode,Entity_value,Image_representation,dataset.entity_data_type,Batch_Predicted_unit)
        

        Total_predicted_units.extend(Batch_Predicted_unit)
        Total_predicted_values.extend(Batch_predicted_value)
        
    Predicted_Entity = [_[0]+' '+_[1] for _ in list(zip(Total_predicted_values,Total_predicted_units))]


    return Predicted_Entity


if __name__ == "__main__":

    predictions = []
    indices = []

    # for x in tqdm(range(10)):
    #     for _ in entity_unit_map.keys():
    #         dataset = Image_dataset(data_folder='train',entity_data_type=_)
    #         predictions += Inference(dataset,Image_model,mode = 'train')

    for _ in tqdm(entity_unit_map.keys()):
        dataset = Image_dataset(data_folder='test',entity_data_type=_)
        indices+= dataset.index
        predictions += Inference(dataset,Image_model,mode = 'test')

    for a in predictions:
        if('_' in a):
            a = a.replace('_',' ')
            a = a.replace('[','')
            a = a.replace(']','')
            a = a.replace(',','')

    print(predictions)
    df = pd.DataFrame(columns = ['index','prediction'])
    df['index'] = indices
    df['prediction'] = predictions
    df.index.name = 'MyIdx'
    df = df.sort_values(by = ['index', 'MyIdx'], ascending = [True, True])
    
    df.reset_index(drop=True, inplace=True)
    df.set_index('index',drop=True,inplace=True)
    df.to_csv('test_out.csv')