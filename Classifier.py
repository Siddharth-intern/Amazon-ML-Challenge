# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.inspection import DecisionBoundaryDisplay
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import os
import numpy as np
from PIL import Image
import joblib
from constants import entity_unit_map
import torch
import torch.nn as nn
device = "cuda" #if torch.cuda.is_available() else "cpu"



class Classifier_network(nn.Module):
    def __init__(self,input_nodes,output_nodes):
        super().__init__()
        self.fc1 = nn.Linear(input_nodes,(input_nodes)//2)
        # self.fc2 = nn.Linear((input_nodes)//2,(input_nodes)//2)
        self.fc2 = nn.Linear((input_nodes)//2,output_nodes)
        self.Relu = nn.ELU()

    def forward(self,x):
        x = self.fc1(x)
        x = self.Relu(x)
        x = self.fc2(x)
    
        return x
    

LEARNING_RATE = 0.01
lossfn1 = nn.CrossEntropyLoss()


# num_epochs = 10000
# torch.manual_seed(1)
#     for epoch in range(num_epochs):
#     hidden, cell = model.init_hidden(batch_size)
#     seq_batch, target_batch = next(iter(seq_dl))
#     optimizer.zero_grad()
#     loss = 0
#     for c in range(seq_length):
#     pred, hidden, cell = model(seq_batch[:, c], hidden, cell)
#     loss += loss_fn(pred, target_batch[:, c])
#     loss.backward()
#     optimizer.step()
#     loss = loss.item()/seq_length
#     if epoch % 500 == 0:
#     print(f'Epoch {epoch} loss: {loss:.4f}')

def classification(mode,image_representation,entity_sub_types,entity_data_type,possible_entity_units):
    
    model = Classifier_network(image_representation.shape[1],len(possible_entity_units)).to(device)
    if os.path.exists(f'./Model_checkpoints/{entity_data_type}_Pytorch_neuralnet.pth'):
        model.load_state_dict(torch.load(f'./Model_checkpoints/{entity_data_type}_Pytorch_neuralnet.pth', weights_only=True))
    optimizer = torch.optim.Adam(model.parameters(),lr = LEARNING_RATE) 

    if(mode == 'train'):
        predicted = model(image_representation.to(device))
        loss = lossfn1(predicted,torch.tensor(entity_sub_types,dtype=torch.long).to(device))
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        torch.save(model.state_dict(),f'./Model_checkpoints/{entity_data_type}_Pytorch_neuralnet.pth')
        predicted = model(image_representation.to(device))
        predicted_units = torch.argmax(predicted,dim = 1)

        return predicted_units
    
    elif(mode == 'test'):
        predicted = model(image_representation.to(device))
        predicted_units = torch.argmax(predicted,dim = 1)
    
        return predicted_units


# def predict_or_train_with_model(mode, model_name, model, train_encoded_images, train_entity_subtypes,entity_type):
    
#     if mode == "train":
#         clf = make_pipeline(StandardScaler(), model)
#         clf.fit(train_encoded_images, train_entity_subtypes)
#         joblib.dump(clf, '../classifier_models/'+entity_type+model_name+".joblib")
#         predicted_units = clf.predict(train_encoded_images)
#         possible_units = list(entity_unit_map[entity_type])
#         possible_units.sort()
#         predicted_units = [entity_unit_map[entity_type][x] for x in predicted_units]
#         return predicted_units

        
#     if mode == "test":
#         clf = joblib.load('../classifier_models/'+entity_type+model_name+".joblib")
#         predicted_units = clf.predict(train_encoded_images)
#         possible_units = list(entity_unit_map[entity_type])
#         possible_units.sort()
#         predicted_units = [entity_unit_map[entity_type][x] for x in predicted_units]
#         return predicted_units
    

# models_list = [
#     # "Linear SVM",
#     # "RBF SVM",
#     # "Gaussian Process",
#     # "Decision Tree",
#     # "Random Forest",
#     # "Neural Net",
#     # "AdaBoost",
#     # "Naive Bayes",
#     # "QDA",
#     # "XGBClassifier",
#     ]

# corresponding_classifiers = [
#     SVC(kernel="linear", C=0.025, random_state=42),
#     SVC(gamma=2, C=1, random_state=42),
#     GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
#     DecisionTreeClassifier(max_depth=5, random_state=42),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42),
#     MLPClassifier(alpha=1, max_iter=1000, random_state=42),
#     AdaBoostClassifier(algorithm="SAMME", random_state=42),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis(),
#     XGBClassifier(n_estimators=100,max_depth=5,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,random_state=42),
#     ]

# def classification(mode,train_encoded_images,train_entity_subtypes,entity_type,models_list = models_list,corresponding_classifiers = corresponding_classifiers):
#     if(mode == "train"):
#         for clf in corresponding_classifiers:
#             model_name = models_list[corresponding_classifiers.index(clf)]
#             predict_or_train_with_model(mode, model_name, 
#                                                     clf, train_encoded_images, train_entity_subtypes,entity_type)
    
#     if(mode == "test"):
#         for clf in corresponding_classifiers:
#             model_name = models_list[corresponding_classifiers.index(clf)]
#             predicted_val = predict_or_train_with_model(mode, model_name, 
#                                                     clf, train_encoded_images, train_entity_subtypes,entity_type)
#             print(f"Predicted units by {model_name} : ",predicted_val)