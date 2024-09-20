import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import os
from constants import entity_unit_map
import numpy as np

# Hyperparameters
seq_length = 5  # Length of input sequence
num_classes = 13  # Digits from 0 to 9,.,s,e
num_layers = 2
embed_dim = 256
learning_rate = 0.001
vocab = [str(x) for x in range(10)]+['.','e','s']



# Step 3: Define the LSTM Model
class DigitLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,entity_data_type):
        super(DigitLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.possible_entity_subtypes = list(entity_unit_map[entity_data_type])

    
    def forward(self,x,hidden,cell):
        # Initialize hidden and cell states
        
        out = self.embedding(x).unsqueeze(1)
        # Forward propagate LSTM
        # print(out.shape)
        out, (hidden,cell) = self.lstm(out, (hidden, cell))
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell

    def init_hidden(self, image_representation, entity_unit):
    
        
        onehot_entity_unit = [[1 if(y == self.possible_entity_subtypes.index(x)) else 0 for y in range(len(self.possible_entity_subtypes))] for x in entity_unit]
        onehot_entity_unit = torch.tensor(onehot_entity_unit)

        batch_size = image_representation.shape[0]
        hidden = torch.zeros(self.num_layers,batch_size,self.hidden_size)
        cell = torch.cat((image_representation,onehot_entity_unit),dim = 1 ).repeat(self.num_layers,1,1)
        return hidden, cell


def string_to_list(entity_value):
    a = list(entity_value)
    a.insert(0,'s')
    a.append('e')
    return a

def Inference_lstm(mode,entity_value,image_representation,entity_data_type,predicted_entity_units):

    batch_size = image_representation.shape[0]
    loss_fn = nn.CrossEntropyLoss()
    hidden_size = image_representation.shape[1]+len(entity_unit_map[entity_data_type])
    input_size = embed_dim
    max_sequence_length = 10
    input_sequence = [] 
    target_sequence = []


    model = DigitLSTM(input_size = input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes,entity_data_type = entity_data_type)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    if os.path.exists(f'./Model_checkpoints/{entity_data_type}_Pytorch_lstm.pth'):
        model.load_state_dict(torch.load(f'./Model_checkpoints/{entity_data_type}_Pytorch_lstm.pth', weights_only=True))
        
    
    if(mode == 'train'):
        for _ in range(len(entity_value)):
            entity_value[_] = string_to_list(entity_value[_])
            input_sequence.append(entity_value[_][0:-1])
            target_sequence.append(entity_value[_][1:])

        optimizer.zero_grad()   
        loss = 0

        for _ in range(len(input_sequence)):
            hidden, cell = model.init_hidden(image_representation[_:_+1],predicted_entity_units[_:_+1])
            # hidden = hidden[:,_,:].unsqueeze(1)
            # cell = cell[:,_,:].unsqueeze(1)

            for c in range(len(input_sequence[_])):
                # print(torch.tensor(vocab.index(input_sequence[_][c])).unsqueeze(dim = 0).unsqueeze(dim = 0).unsqueeze(dim=0))
                pred, hidden, cell = model(torch.tensor(vocab.index(input_sequence[_][c])).unsqueeze(dim = 0), hidden, cell)
                
                print(target_sequence[_])
                loss += loss_fn(pred,torch.tensor(vocab.index(target_sequence[_][c])).unsqueeze(0))
                

                if(target_sequence[_][c] == 'e'):
                    break

        loss.backward()
        optimizer.step()
        loss = loss.item()/max_sequence_length
        torch.save(model.state_dict(),f'./Model_checkpoints/{entity_data_type}_Pytorch_lstm.pth')
    
    predicted_values = ['' for i in range(batch_size)]


    for _ in range(batch_size):
        input_character = 's'
        hidden, cell = model.init_hidden(image_representation[_:_+1],predicted_entity_units[_:_+1])
        for c in range(max_sequence_length):
            pred, hidden, cell = model(torch.tensor(vocab.index(input_character)).unsqueeze(0), hidden, cell)
            input_character = vocab[torch.argmax(pred,dim = 1).item()]
            if(input_character == 'e'):
                break
            if(input_character != 's'):
                predicted_values[_]+=input_character
            
    return predicted_values