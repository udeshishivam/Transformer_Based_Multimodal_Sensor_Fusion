"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18, resnet34

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from .tools import gen_dx_bx, cumsum_trick, QuickCumsum
from time import time


import os
import pickle
from numpy.core.fromnumeric import argmax
import pandas as pd
from PIL import Image
import tensorflow as tf

import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

#model = load_model('/home/kaushek/TFGrid/runs/weather_image_recognition_2.keras')
#class_labels = ['fog', 'night', 'rain', 'sandstorm', 'snow', 'sunny']

WEATHER_SIZES = 6

# Load models
night_model_path='/home/kaushek/DeepRob/TFGrid/models/NightForest.sav'
precipitation_model_path='/home/kaushek/DeepRob/TFGrid/models/PrecipitationCNN.h5'
fog_model_path='/home/kaushek/DeepRob/TFGrid/models/FogCNN.h5'
night_model = pickle.load(open(night_model_path, 'rb'))
precipitation_model = tf.keras.models.load_model(precipitation_model_path)
fog_model = tf.keras.models.load_model(fog_model_path)

def convert_to_feature(image):
    """
    Extract average value of HSV and RGB layers
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    h = np.average(h)
    s = np.average(s)
    v = np.average(v)

    rgb = image
    r, g, b = cv2.split(rgb)
    r = np.average(r)
    g = np.average(g)
    b = np.average(b)

    return [h, s, v, r, g, b]

def convert_to_feature(image):
    """
    Extract average value of HSV and RGB layers
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    h = np.average(h)
    s = np.average(s)
    v = np.average(v)

    rgb = image
    r, g, b = cv2.split(rgb)
    r = np.average(r)
    g = np.average(g)
    b = np.average(b)

    return [h, s, v, r, g, b]

def run_inference_single_image(image):
    """
    Run inference on a single image and print the results.
    
    Args:
    - image_path (str): Path to the image.
    - night_model_path (str): Path to the pre-trained NightForest model.
    - precipitation_model_path (str): Path to the pre-trained PrecipitationCNN model.
    - fog_model_path (str): Path to the pre-trained FogCNN model.
    
    Returns:
    - None
    """

    # Process the single image
    # image = Image.open(image_path)
    # image = image.resize((224, 224), Image.ANTIALIAS)
    # image = np.array(image, dtype=np.uint8)

    # Generate features for the image
    # Resize the input image (assuming it might not be 224x224)
    image_resized = cv2.resize(image, (224, 224))
    features = convert_to_feature(image_resized)

    # Prediction
    night_prediction = night_model.predict([features])
    precipitation_prediction = precipitation_model.predict(np.expand_dims(image_resized, axis=0), verbose=0)
    fog_prediction = fog_model.predict(np.expand_dims(image_resized, axis=0), verbose=0)

    # Convert predictions to integer
    night_label = night_prediction
    precipitation_label = argmax(precipitation_prediction, axis=1)
    fog_label = fog_prediction > 0.5

    # Determine the weather and fog conditions
    weather_result = "night" if night_label[0] == 1 else "clear" if precipitation_label[0] == 0 else "rain" if precipitation_label[0] == 1 else "snow"
    fog_result = "fog" if fog_label[0] == 1 else "no_fog"

    # Print the results
    return (weather_result, fog_result)

def weather_predictor(img_np):
    img_resized = cv2.resize(img_np, (228, 228))

    if img_resized.max() > 1.0:
        img_resized = img_resized / 255.0

    input_img = np.expand_dims(img_resized, axis=0)  # shape: (1, 228, 228, 3)

    # Predict
    prediction = model.predict(input_img, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Optional:
    # print(f"Predicted: {class_labels[predicted_class]}")

    return predicted_class


###############################################
################## Transfuser #################
###############################################

class ImageCNN(nn.Module):
    """ 
    Encoder network for image input list.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    """

    def __init__(self, c_dim, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.features = resnet34()
        self.features.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features.fc = nn.Sequential()

    def forward(self, inputs):
        c = 0
        for x in inputs:
            if self.normalize:
                x = normalize_imagenet(x)
            c += self.features(x)
        return c

def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x


class LidarEncoder(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        num_classes: output feature dimension
        in_channels: input channels
    """

    def __init__(self, num_classes=512, in_channels=64):
        super().__init__()

        self._model = resnet18()
        
        self._model.fc = nn.Sequential()
        _tmp = self._model.conv1
        self._model.conv1 = nn.Conv2d(64, out_channels=_tmp.out_channels, 
            kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding, bias=_tmp.bias)

    def forward(self, inputs):
        features = 0
        for lidar_data in inputs:
            lidar_feature = self._model(lidar_data)
            features += lidar_feature

        return features


# Dynamic Weather based Attention model: Weather Embeddings on the images
class WeatherAwareSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


# Learnable Attention: A multi-head self-attention layer with a learnable attention mask.
class LearnableAttention(nn.Module):
    """
    A vanilla multi-head self-attention layer with a learnable attention mask.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head

        # Key, Query, Value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # Learnable attention mask (initialized to zero for no masking initially)
        self.mask = nn.Parameter(torch.zeros(1, 1, 1, 1), requires_grad=True)  # Mask is learnable
        
        # Regularization (dropout)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # Output projection
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.size()

        # Calculate the key, query, and value for all heads
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Compute attention scores with the learnable mask
        attn = self.compute_attention(q, k, B, T)

        # Apply the attention dropout
        attn = self.attn_drop(attn)

        # Apply the attention to the value vectors
        y = attn @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Reassemble all head outputs side by side

        # Apply the output projection and residual dropout
        y = self.resid_drop(self.proj(y))

        return y

    def compute_attention(self, q, k, B, T):
        """
        Computes the attention scores with a learnable mask.
        The mask is added to the attention scores before applying softmax.
        """
        # Compute raw attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)

        # Apply the learnable attention mask
        attn = attn + self.mask  # Add learnable mask to the attention scores

        # Apply softmax to the attention scores
        attn = F.softmax(attn, dim=-1)

        return attn


# Dynamic Window Attention
#    A multi-head self-attention layer with dynamic windowing for attention computation.
#    The attention is computed based on a sliding window around each query token.
class DynamicWindowAttention(nn.Module):
    """
    A vanilla multi-head dynamic window attention layer with a projection at the end.
    This mechanism performs self-attention with a local window of fixed size around each token.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, window_size=5):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.window_size = window_size  # Define the local attention window size

        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # Dropout for regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # Output projection
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.size()

        # Calculate query, key, values for all heads
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Compute attention scores with dynamic windowing
        attn = self.compute_attention(q, k, B, T)

        # Apply attention dropout
        attn = self.attn_drop(attn)

        # Compute the output by attending to the value vectors
        y = attn @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs side by side

        # Apply the output projection and residual dropout
        y = self.resid_drop(self.proj(y))

        return y

    def compute_attention(self, q, k, B, T):
        """
        Computes the attention scores with a dynamic window, i.e., attending only to nearby tokens in a local window.
        """
        # Initialize the attention matrix
        attn = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)

        # Apply local window attention (mask out tokens outside the window)
        for i in range(T):
            # Define the range of the window (centered around token i)
            start = max(0, i - self.window_size // 2)
            end = min(T, i + self.window_size // 2 + 1)
            
            # Create a mask for attention scores
            mask = torch.zeros(T).to(attn.device)
            mask[start:end] = 1

            # Apply the mask: zero out attention scores outside the local window
            attn[:, :, i, :] = attn[:, :, i, :] * mask

        # Apply softmax to the attention scores
        attn = F.softmax(attn, dim=-1)

        return attn

class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        #self.attn = WeatherAwareSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        #self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        #self.attn = DynamicWindowAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.attn = LearnableAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True), # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        B, T, C = x.size()

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, n_embd, n_head, n_weather, block_exp, n_layer, 
                    vert_anchors, horz_anchors, seq_len, 
                    embd_pdrop, attn_pdrop, resid_pdrop, config):
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = seq_len
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        self.config = config

        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(torch.zeros(1, (self.config.n_views + 1) * seq_len * vert_anchors * horz_anchors, n_embd))
        
        # Weather embedding (sunny, foggy, rainy, snowy, cloudy or other)
        self.weather_emb = nn.Embedding(num_embeddings=n_weather, embedding_dim=n_embd)
        
        # velocity embedding
        self.vel_emb = nn.Linear(1, n_embd)
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, 
                        block_exp, attn_pdrop, resid_pdrop)
                        for layer in range(n_layer)])
        
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = seq_len
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # create the pytorch optimizer object
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        return optim_groups

    def forward(self, image_tensor, lidar_tensor, weather_condition, fog_condition):
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            lidar_tensor (tensor): B*seq_len, C, H, W
            velocity (tensor): ego-velocity
        """
        #print('0-lidar_tensor.shape: ', lidar_tensor.shape)#torch.Size([4, 64, 8, 8])        
        #print('0-image_tensor.shape: ', image_tensor.shape)#torch.Size([4, 64, 8, 8])
        #print('lidar_tensor.shape[2:4]: ', lidar_tensor.shape[2:4]) #torch.Size([8, 8])

        bz = lidar_tensor.shape[0] // self.seq_len
        h, w = lidar_tensor.shape[2:4]
        
        # forward the image model for token embeddings
        image_tensor = image_tensor.view(bz, self.seq_len, -1, h, w)
        lidar_tensor = lidar_tensor.view(bz, self.seq_len, -1, h, w)

        #print('1-lidar_tensor.shape: ', lidar_tensor.shape)#torch.Size([4, 1, 64, 8, 8])
        #print('1-image_tensor.shape: ', image_tensor.shape)#torch.Size([4, 4, 16, 8, 8])    
        
        # pad token embeddings along number of tokens dimension

        token_embeddings = torch.cat([image_tensor, lidar_tensor], dim=1).permute(0,1,3,4,2).contiguous()
        #print('token_embeddings.shape: ',token_embeddings.shape) torch.Size([4, 2, 8, 8, 64])
        token_embeddings = token_embeddings.view(bz, -1, self.n_embd) # (B, an * T, C)
        #print('token_embeddings.shape: ',token_embeddings.shape) torch.Size([4, 128, 64])
        
        # project velocity to n_embed
        #velocity_embeddings = self.vel_emb(velocity.unsqueeze(1)) # (B, C)

        # Processing weather embedding
        device = torch.device('cpu') if not torch.cuda.is_available() else torch.device(f'cuda:{0}')
        
        # fog_condition - fog (0)
        # weather_condition - night (1) or clear (2) or snow (3) or rain (4)
        
        if (fog_condition == "fog"):
            weather_fin = 0
        else:
            weather_fin = 1 if (weather_condition == "night") else 2 if (weather_condition == "clear")  else 3 if (weather_condition == "night") else 4
        
        #weather_embedding = self.weather_emb(torch.tensor([weather_fin], dtype=torch.long).to(device)).unsqueeze(1)
        weather_condition_tensor = torch.tensor([weather_fin], dtype=torch.long).to(device)
        weather_embedding = self.weather_emb(weather_condition_tensor).unsqueeze(1)  # Shape: [1, 1, 64]

        # add (learnable) positional embedding and velocity embedding for all tokens
        
        
        x = self.drop(self.pos_emb + token_embeddings + weather_embedding) #+ velocity_embeddings.unsqueeze(1)) # (B, an * T, C)
        #x = self.drop(self.pos_emb + token_embeddings) #+ velocity_embeddings.unsqueeze(1)) # (B, an * T, C)
        
        
        #print('drop x.shape: ',x.shape) torch.Size([4, 128, 64])

        # x = self.drop(token_embeddings + velocity_embeddings.unsqueeze(1)) # (B, an * T, C)
        x = self.blocks(x) # (B, an * T, C)
        #print('blocks x.shape: ',x.shape) torch.Size([4, 128, 64])

        x = self.ln_f(x) # (B, an * T, C)
        
        #print('ln_f x.shape: ',x.shape) torch.Size([4, 128, 64])
        #x = x.view(bz, (self.config.n_views + 1) * self.seq_len, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.view(bz, (self.seq_len + 1) * self.seq_len, self.vert_anchors, self.horz_anchors, self.n_embd)
        #print('x.shape: ',x.shape) torch.Size([4, 2, 8, 8, 64])
        
        x = x.permute(0,1,4,2,3).contiguous() # same as token_embeddings
        #torch.Size([4, 2, 64, 8, 8])

        image_tensor_out = x[:, :self.seq_len, :, :, :].contiguous().view(bz * self.seq_len, -1, h, w)
        lidar_tensor_out = x[:, self.seq_len:, :, :, :].contiguous().view(bz * self.seq_len, -1, h, w)

        return image_tensor_out, lidar_tensor_out


class Transfuser(nn.Module):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.avgpool = nn.AdaptiveAvgPool2d((self.config.vert_anchors, self.config.horz_anchors))
        
        self.image_encoder = ImageCNN(512, normalize=True)
        self.lidar_encoder = LidarEncoder(num_classes=512, in_channels=2)

        self.transformer1 = GPT(n_embd=64,
                            n_head=config.n_head, 
                            block_exp=config.block_exp, 
                            n_weather=WEATHER_SIZES,
                            n_layer=config.n_layer, 
                            vert_anchors=config.vert_anchors, 
                            horz_anchors=config.horz_anchors, 
                            seq_len=config.seq_len, 
                            embd_pdrop=config.embd_pdrop, 
                            attn_pdrop=config.attn_pdrop, 
                            resid_pdrop=config.resid_pdrop,
                            config=config)
        self.transformer2 = GPT(n_embd=128,
                            n_head=config.n_head, 
                            block_exp=config.block_exp, 
                            n_weather=WEATHER_SIZES,
                            n_layer=config.n_layer, 
                            vert_anchors=config.vert_anchors, 
                            horz_anchors=config.horz_anchors, 
                            seq_len=config.seq_len, 
                            embd_pdrop=config.embd_pdrop, 
                            attn_pdrop=config.attn_pdrop, 
                            resid_pdrop=config.resid_pdrop,
                            config=config)
        self.up1 = nn.ConvTranspose2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size =3,
                                    stride=4
                                    )

        self.up2 = nn.ConvTranspose2d(in_channels=128,
                                    out_channels=128,
                                    kernel_size =3,
                                    stride=8
                                    )         
  

        self.inChannels = 384
        self.concat_conv = nn.Sequential(
            nn.Conv2d(self.inChannels, self.inChannels, kernel_size=3, padding=(1,1)),
            nn.BatchNorm2d(self.inChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inChannels, self.inChannels, kernel_size=3, padding=(1,1)),
            nn.BatchNorm2d(self.inChannels),
            nn.ReLU(inplace=True)
        )                                                                                             

        
    def forward(self, image_list, lidar_list, weather_class, fog_class):
        '''
        Image + LiDAR feature fusion using transformers
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            velocity (tensor): input velocity from speedometer
        '''
        if self.image_encoder.normalize:
            image_list = [normalize_imagenet(image_input) for image_input in image_list]
        #len(image_list) #4
        #image_list[0].shape) #torch.Size([64, 256, 256])

        bz, lidar_channel, h, w = lidar_list.shape
        img_channel = image_list[0].shape[0]           
        self.config.n_views = len(image_list) // self.config.seq_len

        image_tensor = torch.stack(image_list, dim=1).view(bz * self.config.seq_len, img_channel, h, w)
        lidar_tensor = torch.stack(list(lidar_list), dim=1).view(bz * self.config.seq_len, lidar_channel, h, w)

        #image_tensor.shape)#torch.Size([4, 64, 256, 256])
        #lidar_tensor.shape)#torch.Size([4, 64, 256, 256])

        image_features = self.image_encoder.features.conv1(image_tensor)
        image_features = self.image_encoder.features.bn1(image_features)
        image_features = self.image_encoder.features.relu(image_features)
        image_features = self.image_encoder.features.maxpool(image_features)
        lidar_features = self.lidar_encoder._model.conv1(lidar_tensor)
        lidar_features = self.lidar_encoder._model.bn1(lidar_features)
        lidar_features = self.lidar_encoder._model.relu(lidar_features)
        lidar_features = self.lidar_encoder._model.maxpool(lidar_features)

        image_features = self.image_encoder.features.layer1(image_features)
        lidar_features = self.lidar_encoder._model.layer1(lidar_features)

        #print('l-image_features: ',image_features.shape) torch.Size([4, 64, 64, 64])
        #print('l-lidar_features: ',lidar_features.shape) torch.Size([4, 64, 64, 64])
        # fusion at (B, 64, 64, 64)
        image_embd_layer1 = self.avgpool(image_features)
        lidar_embd_layer1 = self.avgpool(lidar_features)
        image_features_layer1, lidar_features_layer1 = self.transformer1(image_embd_layer1, lidar_embd_layer1, weather_class, fog_class)
        image_features_layer1 = F.interpolate(image_features_layer1, scale_factor=8, mode='bilinear')
        lidar_features_layer1 = F.interpolate(lidar_features_layer1, scale_factor=8, mode='bilinear')
        image_features = image_features + image_features_layer1
        lidar_features = lidar_features + lidar_features_layer1

        #print('1-image_features: ',image_features.shape)
        #print('1-lidar_features: ',lidar_features.shape) torch.Size([4, 64, 64, 64])
        ## add deconv

        deConv1Img = self.up1(image_features, output_size = torch.Size([bz,img_channel, 256, 256]))
        deConv1Points = self.up1(lidar_features, output_size = torch.Size([bz,lidar_channel, 256, 256]))

        image_features = self.image_encoder.features.layer2(image_features)
        lidar_features = self.lidar_encoder._model.layer2(lidar_features)
        # fusion at (B, 128, 32, 32)
        image_embd_layer2 = self.avgpool(image_features)
        lidar_embd_layer2 = self.avgpool(lidar_features)
        image_features_layer2, lidar_features_layer2 = self.transformer2(image_embd_layer2, lidar_embd_layer2, weather_class, fog_class)
        image_features_layer2 = F.interpolate(image_features_layer2, scale_factor=4, mode='bilinear')
        lidar_features_layer2 = F.interpolate(lidar_features_layer2, scale_factor=4, mode='bilinear')
        image_features = image_features + image_features_layer2
        lidar_features = lidar_features + lidar_features_layer2

        #print('2-image_features: ',image_features.shape)
        #print('2-lidar_features: ',lidar_features.shape)torch.Size([4, 128, 32, 32])

        deConv2Img = self.up2(image_features, output_size = torch.Size([bz,img_channel, 256, 256]))
        deConv2Points = self.up2(lidar_features, output_size = torch.Size([bz,lidar_channel, 256, 256]))

        Img_fused_features = torch.cat([deConv1Img, deConv2Img], dim=1)
        Points_fused_features = torch.cat([deConv1Points, deConv2Points], dim=1)

        fused_features  = torch.cat([Img_fused_features, Points_fused_features], dim=1)
        
        fused_features = self.concat_conv(fused_features)

        return fused_features



###############################################
################ point pillars ################
###############################################
class Empty(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Empty, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        return args


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        use_norm = True

        if use_norm:
            self.norm = nn.BatchNorm1d(self.units,eps=1e-3, momentum=0.01)
            self.linear = nn.Linear(in_channels, self.units,bias=False)
        else:
            self.norm = Empty
            self.linear = nn.Linear(in_channels, self.units,bias=True)

    def forward(self, inputs):

        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = F.relu(x)

        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarFeatureNet(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64,),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = 'PillarFeatureNet'
        assert len(num_filters) > 0
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters) # [3,64]
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors): #features [num_voxels, max_points, ndim] 
        #voxels from two batches are concatenated and coord have information corrd [num_voxels, (batch, x,y)]
        # pdb.set_trace()

        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].float().unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].float().unsqueeze(1) * self.vy + self.y_offset)

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features) #here they are considering num of voxels as batch size for linear layer

        return features.squeeze()


class PointPillarsScatter(nn.Module):
    def __init__(self,
                 output_shape,
                 num_input_features=64):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = 'PointPillarsScatter'
        self.output_shape = output_shape
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.nchannels = num_input_features

    def forward(self, voxel_features, coords, batch_size):

        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(self.nchannels, self.nx * self.ny, dtype=voxel_features.dtype,
                                 device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny, self.nx)

        return batch_canvas


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]

    Returns:
        [type]: [description]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class PillarFeatures(nn.Module):

    def __init__(self, cfg):
        super(PillarFeatures, self).__init__()
        # voxel feature extractor
        self.cfg = cfg
        self.voxel_feature_extractor = PillarFeatureNet( num_input_features = cfg['input_features'],
                use_norm = cfg['use_norm'],
                num_filters=cfg['vfe_filters'],
                with_distance=cfg['with_distance'],
                voxel_size=cfg['voxel_size'],
                pc_range=cfg['pc_range'])

        grid_size = (np.asarray(cfg['pc_range'][3:]) - np.asarray(cfg['pc_range'][:3])) / np.asarray(cfg['voxel_size'])
        grid_size = np.round(grid_size).astype(np.int64)
        dense_shape = [1] + grid_size[::-1].tolist() + [cfg['vfe_filters'][-1]] #grid_size[::-1] reverses the index from xyz to zyx

        # Middle feature extractor
        self.middle_feature_extractor = PointPillarsScatter(output_shape = dense_shape,
                                        num_input_features = cfg['vfe_filters'][-1])        
    

    def forward(self, voxels, coors, num_points,bsz):

                  
        voxel_features = self.voxel_feature_extractor(voxels, num_points, coors)
        ### voxel_feature_extractor deliver pillar features
        
        spatial_features = self.middle_feature_extractor(voxel_features, coors, bsz)#self.cfg['batch_size']
        ### spatial_features (pseudo image based from Pillar features)

        return spatial_features

###############################################
############### Lift-splat ###################
###############################################

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320+112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)
        # Depth
        x = self.depthnet(x) ### generates features C(64) and alphas D, out D+C

        depth = self.get_depth_dist(x[:, :self.D]) ### applies softmax over alphas D(based on the number of bins for depths)
        #depth.unsqueeze(1) : , x[:, self.D:(self.D + self.C)].unsqueeze(2)  Column
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2) #alpha * C


        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0), ##number of output channels outC
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class LPT(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC,pp_config,tf_config):
        super(LPT, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf
        self.pp_config = pp_config

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample) ## camC features from lift, D number of discrete bins for depths
        
        # sum lift-splat features and PointPillars features
        self.inFeatures = self.camC + self.pp_config['vfe_filters'][0]        

        self.pointpillars = PillarFeatures(pp_config)

        self.transfuser = Transfuser(tf_config)

        self.bevencode = BevEncode(inC=384, outC=outC) ### outC number of output channels

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
    
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)        
        D, _, _ = ds.shape #
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        #print('xs: ', xs.shape)
        #print('ys: ', ys.shape)
        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape ### B batch size, N number of cameras

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape
        
        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts] ### sort all points accoirdn to bin id (ranks are pillars?)

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks) ### "Cumulative sum pooling trick"
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y) ### nx:  tensor([200, 200,   1])  -- ( ,64,1,200,200)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1) ### unbind removes tensor 2nd dimension (z) then dimension B*C*X*Y
                                                                                                               ##H*W?
        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans): ### get pillars
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)

        x = self.voxel_pooling(geom, x) #### deliver tensor B*C*X*Y

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans, voxels, coors, num_points):
        # B,N,_,_,_ = x.shape
        # for b in range(B):
        #     for n in range(N):
        #         img = x[b, n]  # shape: (3, H, W)
        #         img_np = img.permute(1, 2, 0).cpu().numpy()

        #         # Create a Figure object
        #         fig = Figure(figsize=(4, 4))
        #         canvas = FigureCanvas(fig)
        #         ax = fig.add_subplot(111)
        #         ax.imshow(img_np)
        #         ax.set_title(f"Batch {b}, Image {n}")
        #         ax.axis('off')

        #         # Draw the figure on canvas and show as image using OpenCV
        #         canvas.draw()
        #         width, height = fig.get_size_inches() * fig.get_dpi()
        #         image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        #         # Use OpenCV to show the image and wait for key
        #         import cv2
        #         cv2.imshow("Image", image)
        #         print("Press any key in the OpenCV window to continue...")
        #         cv2.waitKey(0)
        #         cv2.destroyAllWindows()
        
        img = x[0, 0]  # shape: (3, H, W)
        img_np = img.permute(1, 2, 0).cpu().numpy()
        weather_class, fog_class = run_inference_single_image(img_np)
        
        # For Plotting Purose
        # fig = Figure(figsize=(4, 4))
        # canvas = FigureCanvas(fig)
        # ax = fig.add_subplot(111)
        # ax.imshow(img_np)
        # ax.set_title(f"{class_labels[weather_class]}")
        # ax.axis('off')

        # # Draw the figure on canvas and show as image using OpenCV
        # canvas.draw()
        # width, height = fig.get_size_inches() * fig.get_dpi()
        # image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        # # Use OpenCV to show the image and wait for key
        # cv2.imshow(f"{class_labels[weather_class]}", image)
        # print("Press any key in the OpenCV window to continue...")
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans) 
        bsz = x.shape[0]
        pointpillars_features = self.pointpillars(voxels, coors, num_points,bsz)

        x = self.transfuser(x,pointpillars_features, weather_class, fog_class)#transfuser        
        
        x = self.bevencode(x) ### encoder to create BEV

        return x


def compile_model(grid_conf, data_aug_conf, outC,cfg_pp,tf_config):
    return LPT(grid_conf, data_aug_conf, outC,cfg_pp,tf_config)
