import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from datetime import datetime
import json
import pandas as pd

# Deep Learning Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF

# TensorFlow for additional components
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0

# Scientific Libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import skimage
from skimage import feature, measure, filters, segmentation
from skimage.feature import local_binary_pattern
from scipy import ndimage
from scipy.spatial.distance import cdist

# Visualization and Interpretation
import matplotlib.patches as mpatches
# from matplotlib.backends.backend_tkagg import FigureCanvasTkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class StainNormalizer:
    """Macenko Stain Normalization for H&E histology images"""
    
    def __init__(self):
        # Reference H&E matrix from literature
        self.target_stains = np.array([[0.5626, 0.2159],
                                    [0.7201, 0.8012],
                                    [0.4062, 0.5581]])
        self.target_concentrations = np.array([[1.9705, 1.0308]])
    
    def normalize(self, image):
        """Apply Macenko normalization"""
        try:
            # Convert to OD space
            image_od = self.rgb_to_od(image)
            
            # Get stain matrix
            stain_matrix = self.get_stain_matrix(image_od)
            
            # Get concentrations
            concentrations = self.get_concentrations(image_od, stain_matrix)
            
            # Normalize concentrations
            normalized_conc = self.normalize_concentrations(concentrations)
            
            # Reconstruct image
            normalized_od = np.dot(normalized_conc, self.target_stains.T)
            normalized_image = self.od_to_rgb(normalized_od)
            
            return np.clip(normalized_image, 0, 255).astype(np.uint8)
        except:
            return image
    
    def rgb_to_od(self, image):
        """Convert RGB to Optical Density"""
        image = image.astype(np.float64)
        image = np.maximum(image, 1)  # Avoid log(0)
        return -np.log(image / 255.0)
    
    def od_to_rgb(self, od):
        """Convert Optical Density to RGB"""
        return 255 * np.exp(-od)
    
    def get_stain_matrix(self, od_image):
        """Extract stain matrix using SVD"""
        od_flat = od_image.reshape(-1, 3)
        # Remove background pixels
        od_flat = od_flat[np.sum(od_flat, axis=1) > 0.3]
        
        if len(od_flat) == 0:
            return self.target_stains
        
        # SVD
        U, s, Vt = np.linalg.svd(od_flat, full_matrices=False)
        
        # Get top 2 stain directions
        stain_matrix = Vt[:2, :].T
        
        # Ensure proper orientation
        if stain_matrix[0, 0] < 0:
            stain_matrix[:, 0] *= -1
        if stain_matrix[0, 1] < 0:
            stain_matrix[:, 1] *= -1
            
        return stain_matrix
    
    def get_concentrations(self, od_image, stain_matrix):
        """Get stain concentrations"""
        od_flat = od_image.reshape(-1, 3)
        concentrations = np.linalg.lstsq(stain_matrix, od_flat.T, rcond=None)[0]
        return concentrations.T.reshape(od_image.shape[:2] + (2,))
    
    def normalize_concentrations(self, concentrations):
        """Normalize concentration values"""
        conc_flat = concentrations.reshape(-1, 2)
        
        # Get percentiles for normalization
        p99 = np.percentile(conc_flat, 99, axis=0)
        
        # Normalize
        normalized = conc_flat * (self.target_concentrations / (p99 + 1e-6))
        
        return normalized.reshape(concentrations.shape)

class AttentionBlock(nn.Module):
    """Attention mechanism for U-Net"""
    
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

class AttentionUNet(nn.Module):
    """Attention U-Net for tumor segmentation"""
    
    def __init__(self, n_channels=3, n_classes=1):
        super(AttentionUNet, self).__init__()
        
        # Encoder with EfficientNet backbone
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.backbone_features = nn.ModuleList(list(self.backbone.features.children()))
        
        # Capture layers at these points: [1, 2, 4, 6]
        self.skip_idxs = [1, 2, 4, 6]
        self.skip_channels = [16, 24, 40, 112]  # known from EfficientNet-B0
        self.bottleneck_channels = 1280  # output of last feature
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(1280, 512, 2, stride=2)
        self.att1 = AttentionBlock(512, 112, 256)
        self.dec1 = self.conv_block(512 + 112, 512)
        # self.att1 = AttentionBlock(512, 192, 256)
        # self.dec1 = self.conv_block(512 + 192, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att2 = AttentionBlock(256, 40, 128)
        self.dec2 = self.conv_block(256 + 40, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att3 = AttentionBlock(128, 24, 64)
        self.dec3 = self.conv_block(128 + 24, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att4 = AttentionBlock(64, 16, 32)
        self.dec4 = self.conv_block(64 + 16, 64)
        
        self.up5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec5 = self.conv_block(32, 32)
        
        self.final = nn.Conv2d(32, n_classes, 1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        features = []
        for i, layer in enumerate(self.backbone_features):
            x = layer(x)
            if i in [1, 2, 4, 6]:  # Skip connections at different scales
                features.append(x)
        # Ensure features order is deepest first
        features = features[::-1]
        
        
        # Decoder with attention
        d1 = self.up1(x)
        # Resize features[3] to match d1
        features[3] = F.interpolate(features[3], size=d1.shape[2:], mode='bilinear', align_corners=False)

        d1 = self.att1(d1, features[3])
        d1 = torch.cat([d1, features[3]], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = self.att2(d2, features[2])
        d2 = torch.cat([d2, features[2]], dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        d3 = self.att3(d3, features[1])
        d3 = torch.cat([d3, features[1]], dim=1)
        d3 = self.dec3(d3)
        
        d4 = self.up4(d3)
        d4 = self.att4(d4, features[0])
        d4 = torch.cat([d4, features[0]], dim=1)
        d4 = self.dec4(d4)
        
        d5 = self.up5(d4)
        d5 = self.dec5(d5)
        
        output = torch.sigmoid(self.final(d5))
        
        return output