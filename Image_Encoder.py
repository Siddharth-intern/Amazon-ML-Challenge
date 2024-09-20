from transformers import AutoProcessor,AutoModel
from transformers import  DonutProcessor, VisionEncoderDecoderModel
from torchvision import transforms
from PIL import Image
import os
from utils import download_images
import pandas as pd
import torch
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"


# processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
# model = AutoModel.from_pretrained("openai/clip-vit-base-patch32",output_hidden_states = True).to(device)


class Image_encoder():
    def __init__(self,processor,model):
        self.processor = processor
        self.model = model
         
        
    def features(self,image):
        inputs = self.processor(text = ['sometext'],images = image, return_tensors="pt").to(device)
        outputs = self.model(**inputs).vision_model_output.last_hidden_state[:,0,:].to('cpu').detach()
        
        return outputs