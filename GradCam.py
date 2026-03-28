import numpy as np
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
import glob
from skimage import filters, morphology, segmentation, measure
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')

class MacenkoStainNormalizer:
    def __init__(self):
        self.target_stains = np.array([[0.5626, 0.2159],
                                       [0.7201, 0.8012], 
                                       [0.4062, 0.5581]])
        self.target_concentrations = np.array([[1.9705, 1.0308]])
        
    def rgb_to_od(self, img):
        img = np.maximum(img, 1e-6)
        return -np.log(img / 255.0)
    
    def od_to_rgb(self, od):
        rgb = 255 * np.exp(-od)
        return np.clip(rgb, 0, 255).astype(np.uint8)
    
    def get_stain_matrix(self, od, beta=0.15, alpha=1):
        od_flat = od.reshape(-1, 3)
        od_hat = od_flat[(od_flat > beta).any(axis=1)]
        
        if len(od_hat) == 0:
            return self.target_stains
            
        cov = np.cov(od_hat.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        idx = np.argsort(eigenvals)[::-1]
        eigenvecs = eigenvecs[:, idx]
        
        proj = np.dot(od_hat, eigenvecs[:, :2])
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        
        min_phi = np.percentile(phi, alpha)
        max_phi = np.percentile(phi, 100 - alpha)
        
        v1 = np.dot(eigenvecs[:, :2], [np.cos(min_phi), np.sin(min_phi)])
        v2 = np.dot(eigenvecs[:, :2], [np.cos(max_phi), np.sin(max_phi)])
        
        if v1[0] > 0: v1 = -v1
        if v2[0] > 0: v2 = -v2
        
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        return np.column_stack([v1, v2])
    
    def separate_stains(self, od, stain_matrix):
        od_flat = od.reshape(-1, 3)
        concentrations = np.linalg.lstsq(stain_matrix, od_flat.T, rcond=None)[0]
        concentrations = np.maximum(concentrations, 0)
        return concentrations.T.reshape(od.shape[:2] + (2,))
    
    def normalize_he(self, image):
        od = self.rgb_to_od(image)
        source_stains = self.get_stain_matrix(od)
        concentrations = self.separate_stains(od, source_stains)
        normalized_od = np.dot(concentrations, self.target_stains.T)
        normalized_rgb = self.od_to_rgb(normalized_od)
        return normalized_rgb

class ImageProcessor:
    @staticmethod
    def create_binary_mask(image, method='otsu'):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        if method == 'otsu':
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        else:
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    @staticmethod
    def extract_features(image, mask):
        props = measure.regionprops(mask.astype(int), intensity_image=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        
        if len(props) == 0:
            return {}
        
        largest_region = max(props, key=lambda x: x.area)
        
        features = {
            'area': largest_region.area,
            'perimeter': largest_region.perimeter,
            'eccentricity': largest_region.eccentricity,
            'solidity': largest_region.solidity,
            'extent': largest_region.extent,
            'mean_intensity': largest_region.mean_intensity,
            'compactness': (largest_region.perimeter ** 2) / (4 * np.pi * largest_region.area) if largest_region.area > 0 else 0
        }
        
        return features

class GradCAMPlusPlus:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        target_layer = dict(self.model.named_modules())[self.target_layer_name]
        self.hooks.append(target_layer.register_backward_hook(backward_hook))
        self.hooks.append(target_layer.register_forward_hook(forward_hook))
    
    def generate_cam(self, input_tensor, class_idx):
        self.model.eval()
        
        output = self.model(input_tensor)
        self.model.zero_grad()
        
        class_score = output[:, class_idx]
        class_score.backward(retain_graph=True)
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        alpha = F.relu(gradients).sum(dim=(1, 2)) / (2.0 * gradients.sum(dim=(1, 2)) + activations.sum(dim=(1, 2)) * gradients.pow(2).sum(dim=(1, 2)))
        alpha = alpha.view(-1, 1, 1)
        
        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2)).view(-1, 1, 1)
        
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def cleanup(self):
        for hook in self.hooks:
            hook.remove()

class ScoreCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.activations = None
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        target_layer = dict(self.model.named_modules())[self.target_layer_name]
        self.hooks.append(target_layer.register_forward_hook(forward_hook))
    
    def generate_cam(self, input_tensor, class_idx):
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(input_tensor)
            base_score = F.softmax(output, dim=1)[0, class_idx].item()
        
        activations = self.activations[0]
        n_channels = activations.shape[0]
        
        scores = []
        for i in range(n_channels):
            mask = activations[i:i+1]
            mask = F.interpolate(mask.unsqueeze(0), size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            
            masked_input = input_tensor * mask
            
            with torch.no_grad():
                masked_output = self.model(masked_input)
                score = F.softmax(masked_output, dim=1)[0, class_idx].item()
            
            scores.append(score)
        
        scores = torch.tensor(scores).view(-1, 1, 1)
        cam = (scores * activations).sum(dim=0)
        
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def cleanup(self):
        for hook in self.hooks:
            hook.remove()

class BreakHisDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, normalizer=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.normalizer = normalizer
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.normalizer:
            image = self.normalizer.normalize_he(image)
        
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        return image, label

class AttentionBlock(nn.Module):
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

class BreakHisClassifier(nn.Module):
    def __init__(self, num_classes=8, efficientnet_version='b0'):
        super(BreakHisClassifier, self).__init__()
        
        self.backbone = EfficientNet.from_pretrained(f'efficientnet-{efficientnet_version}')
        
        feature_dims = {
            'b0': [32, 24, 40, 112, 1280],
            'b1': [32, 24, 40, 112, 1280],
            'b2': [32, 24, 48, 120, 1408],
            'b3': [40, 32, 48, 136, 1536],
            'b4': [48, 32, 56, 160, 1792]
        }
        
        dims = feature_dims.get(efficientnet_version, feature_dims['b0'])
        
        self.up_conv4 = nn.ConvTranspose2d(dims[4], dims[3], kernel_size=2, stride=2)
        self.att4 = AttentionBlock(dims[3], dims[3], dims[3]//2)
        self.conv4 = self._make_conv_block(dims[3] + dims[3], dims[3])
        
        self.up_conv3 = nn.ConvTranspose2d(dims[3], dims[2], kernel_size=2, stride=2)
        self.att3 = AttentionBlock(dims[2], dims[2], dims[2]//2)
        self.conv3 = self._make_conv_block(dims[2] + dims[2], dims[2])
        
        self.up_conv2 = nn.ConvTranspose2d(dims[2], dims[1], kernel_size=2, stride=2)
        self.att2 = AttentionBlock(dims[1], dims[1], dims[1]//2)
        self.conv2 = self._make_conv_block(dims[1] + dims[1], dims[1])
        
        self.up_conv1 = nn.ConvTranspose2d(dims[1], dims[0], kernel_size=2, stride=2)
        self.att1 = AttentionBlock(dims[0], dims[0], dims[0]//2)
        self.conv1 = self._make_conv_block(dims[0] + dims[0], dims[0])
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dims[4], 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def extract_encoder_features(self, x):
        features = []
        
        x = self.backbone._conv_stem(x)
        x = self.backbone._bn0(x)
        x = self.backbone._swish(x)
        features.append(x)
        
        for idx, block in enumerate(self.backbone._blocks):
            x = block(x)
            if idx in [2, 4, 10, 16]:
                features.append(x)
        
        x = self.backbone._conv_head(x)
        x = self.backbone._bn1(x)
        x = self.backbone._swish(x)
        features.append(x)
        
        return features
    
    def forward(self, x):
        encoder_features = self.extract_encoder_features(x)
        classification_output = self.classifier(encoder_features[-1])
        return classification_output

class BreakHisAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.normalizer = MacenkoStainNormalizer()
        self.processor = ImageProcessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_encoder = LabelEncoder()
        self.image_paths = []
        self.labels = []
        self.magnifications = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.mag_train = None
        self.mag_test = None
        
        self.class_mapping = {
            'adenosis': 'Benign - Adenosis',
            'fibroadenoma': 'Benign - Fibroadenoma', 
            'phyllodes_tumor': 'Benign - Phyllodes Tumor',
            'tubular_adenoma': 'Benign - Tubular Adenoma',
            'ductal_carcinoma': 'Malignant - Ductal Carcinoma',
            'lobular_carcinoma': 'Malignant - Lobular Carcinoma',
            'mucinous_carcinoma': 'Malignant - Mucinous Carcinoma',
            'papillary_carcinoma': 'Malignant - Papillary Carcinoma'
        }
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
        
        self.gradcam = None
        self.scorecam = None
    
    def detect_dataset_structure(self):
        structure_info = {
            'magnifications': set(),
            'cancer_types': set(),
            'total_images': 0,
            'structure_type': 'unknown'
        }
        
        print(f"Analyzing dataset structure in: {self.dataset_path}")
        
        for root, dirs, files in os.walk(self.dataset_path):
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
            structure_info['total_images'] += len(image_files)
            
            for dirname in dirs:
                if 'X' in dirname and any(mag in dirname for mag in ['40', '100', '200', '400']):
                    structure_info['magnifications'].add(dirname)
            
            path_parts = root.lower().split(os.sep)
            for part in path_parts:
                if any(cancer in part for cancer in ['adenosis', 'fibroadenoma', 'phyllodes', 'tubular', 
                                                   'ductal', 'lobular', 'mucinous', 'papillary']):
                    structure_info['cancer_types'].add(part)
        
        print(f"Found magnifications: {structure_info['magnifications']}")
        print(f"Found cancer types: {structure_info['cancer_types']}")
        print(f"Total images found: {structure_info['total_images']}")
        
        return structure_info
        
    def load_dataset(self, selected_magnification='400X'):
        class_counts = {}
        self.image_paths = []
        self.labels = []
        self.magnifications = []
        
        print(f"Loading dataset for magnification: {selected_magnification}")
        
        structure_info = self.detect_dataset_structure()
        
        if structure_info['total_images'] == 0:
            raise ValueError("No image files found in the dataset directory")
        
        success = False
        
        success = self.load_standard_breakhis(selected_magnification, class_counts)
        
        if not success:
            success = self.load_flexible_structure(selected_magnification, class_counts)
        
        if not success:
            success = self.load_flat_structure(selected_magnification, class_counts)
        
        if not success or len(self.image_paths) == 0:
            raise ValueError(f"No images found for magnification {selected_magnification}. "
                           f"Available magnifications: {structure_info['magnifications']}")
        
        self.labels = self.label_encoder.fit_transform(self.labels)
        
        print(f"Successfully loaded {len(self.image_paths)} images")
        return class_counts
    
    def load_standard_breakhis(self, selected_magnification, class_counts):
        try:
            possible_structures = [
                os.path.join(self.dataset_path, "BreaKHis_v1", "histology_slides", "breast"),
                os.path.join(self.dataset_path, "histology_slides", "breast"),
                os.path.join(self.dataset_path, "breast"),
                self.dataset_path
            ]
            
            breast_path = None
            for structure in possible_structures:
                if os.path.exists(structure):
                    benign_path = os.path.join(structure, "benign")
                    malignant_path = os.path.join(structure, "malignant")
                    if os.path.exists(benign_path) and os.path.exists(malignant_path):
                        breast_path = structure
                        break
            
            if not breast_path:
                return False
            
            benign_path = os.path.join(breast_path, "benign")
            malignant_path = os.path.join(breast_path, "malignant")
            
            for category_path, category_type in [(benign_path, 'benign'), (malignant_path, 'malignant')]:
                for cancer_type_folder in os.listdir(category_path):
                    cancer_type_path = os.path.join(category_path, cancer_type_folder)
                    
                    if not os.path.isdir(cancer_type_path):
                        continue
                    
                    found_images = False
                    for sub_folder in os.listdir(cancer_type_path):
                        if sub_folder.startswith("SOB_"):
                            subfolder_path = os.path.join(cancer_type_path, sub_folder)
                            if os.path.isdir(subfolder_path):
                                magnification_path = os.path.join(subfolder_path, selected_magnification)
                                if os.path.exists(magnification_path):
                                    images = self.get_images_from_folder(magnification_path)
                                    if images:
                                        self.add_images_to_dataset(images, cancer_type_folder, selected_magnification, class_counts)
                                        found_images = True
                    
                    if not found_images:
                        magnification_path = os.path.join(cancer_type_path, selected_magnification)
                        if os.path.exists(magnification_path):
                            images = self.get_images_from_folder(magnification_path)
                            if images:
                                self.add_images_to_dataset(images, cancer_type_folder, selected_magnification, class_counts)
                                found_images = True
            
            return len(self.image_paths) > 0
            
        except Exception as e:
            print(f"Standard BreakHis loading failed: {e}")
            return False
    
    def load_flexible_structure(self, selected_magnification, class_counts):
        try:
            print("Trying flexible structure loading...")
            
            for root, dirs, files in os.walk(self.dataset_path):
                if selected_magnification in os.path.basename(root):
                    images = self.get_images_from_folder(root)
                    if images:
                        cancer_type = self.infer_cancer_type_from_path(root)
                        if cancer_type:
                            self.add_images_to_dataset(images, cancer_type, selected_magnification, class_counts)
                
                for dirname in dirs:
                    if dirname == selected_magnification:
                        mag_path = os.path.join(root, dirname)
                        images = self.get_images_from_folder(mag_path)
                        if images:
                            cancer_type = self.infer_cancer_type_from_path(root)
                            if cancer_type:
                                self.add_images_to_dataset(images, cancer_type, selected_magnification, class_counts)
            
            return len(self.image_paths) > 0
            
        except Exception as e:
            print(f"Flexible structure loading failed: {e}")
            return False
    
    def load_flat_structure(self, selected_magnification, class_counts):
        try:
            print("Trying flat structure loading...")
            
            all_images = []
            for root, dirs, files in os.walk(self.dataset_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                        all_images.append(os.path.join(root, file))
            
            filtered_images = []
            for img_path in all_images:
                if selected_magnification in img_path or selected_magnification.replace('X', '') in img_path:
                    filtered_images.append(img_path)
            
            if not filtered_images:
                print(f"No images found with magnification {selected_magnification}, using all available images")
                filtered_images = all_images
            
            for img_path in filtered_images:
                cancer_type = self.infer_cancer_type_from_path(img_path)
                if cancer_type:
                    self.add_images_to_dataset([img_path], cancer_type, selected_magnification, class_counts)
                else:
                    default_type = 'adenosis' if 'benign' in img_path.lower() else 'ductal_carcinoma'
                    self.add_images_to_dataset([img_path], default_type, selected_magnification, class_counts)
            
            return len(self.image_paths) > 0
            
        except Exception as e:
            print(f"Flat structure loading failed: {e}")
            return False
    
    def get_images_from_folder(self, folder_path):
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        images = []
        
        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
            images.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
        
        return images
    
    def infer_cancer_type_from_path(self, path):
        path_lower = path.lower()
        
        type_keywords = {
            'adenosis': 'adenosis',
            'fibroadenoma': 'fibroadenoma',
            'phyllodes': 'phyllodes_tumor',
            'tubular': 'tubular_adenoma',
            'ductal': 'ductal_carcinoma',
            'lobular': 'lobular_carcinoma',
            'mucinous': 'mucinous_carcinoma',
            'papillary': 'papillary_carcinoma'
        }
        
        for keyword, cancer_type in type_keywords.items():
            if keyword in path_lower:
                return cancer_type
        
        return None
    
    def add_images_to_dataset(self, images, cancer_type, magnification, class_counts):
        normalized_type = cancer_type.lower().replace(' ', '_')
        
        self.image_paths.extend(images)
        self.labels.extend([normalized_type] * len(images))
        self.magnifications.extend([magnification] * len(images))
        
        if normalized_type not in class_counts:
            class_counts[normalized_type] = 0
        class_counts[normalized_type] += len(images)
        
        print(f"Added {len(images)} images for {normalized_type}")
        
    def split_dataset(self, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test, self.mag_train, self.mag_test = train_test_split(
            self.image_paths, self.labels, self.magnifications, 
            test_size=test_size, random_state=42, stratify=self.labels
        )
        
    def create_dataloaders(self, batch_size=16):
        train_dataset = BreakHisDataset(
            self.X_train, self.y_train, 
            transform=self.transform, 
            normalizer=self.normalizer
        )
        
        test_dataset = BreakHisDataset(
            self.X_test, self.y_test, 
            transform=self.transform, 
            normalizer=self.normalizer
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, 
            shuffle=True, num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, 
            shuffle=False, num_workers=0
        )
        
        return train_loader, test_loader
    
    def setup_explainability_tools(self):
        if self.model is not None:
            if self.gradcam is not None:
                self.gradcam.cleanup()
            if self.scorecam is not None:
                self.scorecam.cleanup()
            
            target_layer = 'backbone._blocks.15._depthwise_conv'
            self.gradcam = GradCAMPlusPlus(self.model, target_layer)
            self.scorecam = ScoreCAM(self.model, target_layer)
    
    def train_model(self, epochs=50, learning_rate=0.001, progress_callback=None):
        num_classes = len(self.label_encoder.classes_)
        self.model = BreakHisClassifier(num_classes=num_classes).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        
        train_loader, test_loader = self.create_dataloaders()
        
        best_accuracy = 0.0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            scheduler.step()
            
            train_accuracy = 100. * train_correct / train_total
            
            self.model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()
            
            test_accuracy = 100. * test_correct / test_total
            
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(self.model.state_dict(), 'best_breakhis_multiclass_model.pth')
            
            if progress_callback:
                progress_callback(epoch + 1, epochs, train_accuracy, test_accuracy)
        
        self.setup_explainability_tools()
        return best_accuracy
        
    def evaluate_model(self):
        if self.model is None:
            return None
            
        _, test_loader = self.create_dataloaders()
        
        self.model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        accuracy = accuracy_score(y_true, y_pred)
        class_names = self.label_encoder.classes_
        report = classification_report(y_true, y_pred, target_names=class_names)
        cm = confusion_matrix(y_true, y_pred)
        
        return accuracy, report, cm, class_names
    
    def generate_explainability_maps(self, image_tensor, predicted_class):
        explainability_maps = {}
        
        if self.gradcam is not None:
            try:
                gradcam_map = self.gradcam.generate_cam(image_tensor, predicted_class)
                gradcam_map = cv2.resize(gradcam_map, (224, 224))
                explainability_maps['gradcam++'] = gradcam_map
            except Exception as e:
                print(f"Grad-CAM++ generation failed: {e}")
                explainability_maps['gradcam++'] = None
        
        if self.scorecam is not None:
            try:
                scorecam_map = self.scorecam.generate_cam(image_tensor, predicted_class)
                scorecam_map = cv2.resize(scorecam_map, (224, 224))
                explainability_maps['scorecam'] = scorecam_map
            except Exception as e:
                print(f"Score-CAM generation failed: {e}")
                explainability_maps['scorecam'] = None
        
        return explainability_maps
    
    def process_and_predict_image(self, image_path):
        if self.model is None:
            return None
            
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        normalized_image = self.normalizer.normalize_he(original_image)
        
        binary_mask = self.processor.create_binary_mask(normalized_image)
        
        features = self.processor.extract_features(normalized_image, binary_mask)
        
        image_tensor = self.transform(normalized_image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = output.argmax(1).item()
            confidence = probabilities[0][predicted_class].item()
        
        explainability_maps = self.generate_explainability_maps(image_tensor, predicted_class)
        
        class_name = self.label_encoder.inverse_transform([predicted_class])[0]
        full_class_name = self.class_mapping.get(class_name, class_name)
        
        probs_dict = {}
        for i, class_label in enumerate(self.label_encoder.classes_):
            probs_dict[self.class_mapping.get(class_label, class_label)] = probabilities[0][i].item()
        
        return {
            'original_image': original_image,
            'normalized_image': normalized_image,
            'binary_mask': binary_mask,
            'features': features,
            'explainability_maps': explainability_maps,
            'prediction': {
                'class': full_class_name,
                'confidence': confidence,
                'probabilities': probs_dict
            }
        }
    
    def save_model(self, filepath):
        if self.model is None:
            return False
            
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_encoder': self.label_encoder,
            'class_mapping': self.class_mapping,
            'num_classes': len(self.label_encoder.classes_)
        }, filepath)
        return True
    
    def load_model(self, filepath, num_classes=None):
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if num_classes is None:
            num_classes = checkpoint.get('num_classes', 8)
        
        self.model = BreakHisClassifier(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.label_encoder = checkpoint['label_encoder']
        self.class_mapping = checkpoint.get('class_mapping', self.class_mapping)
        
        self.setup_explainability_tools()
        return True

class BreakHisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BreakHis Multi-Class Cancer Classification with Explainable AI")
        self.root.geometry("1800x1200")
        self.root.configure(bg='#f0f0f0')
        
        self.analyzer = None
        self.dataset_path = None
        
        self.setup_styles()
        self.create_widgets()
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), background='#f0f0f0')
        style.configure('Info.TLabel', font=('Arial', 10), background='#f0f0f0')
        style.configure('Success.TLabel', font=('Arial', 10, 'bold'), foreground='green', background='#f0f0f0')
        style.configure('Error.TLabel', font=('Arial', 10, 'bold'), foreground='red', background='#f0f0f0')
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(title_frame, text="BreakHis Multi-Class Breast Cancer Classification with Explainable AI", 
                 style='Title.TLabel').pack()
        
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_control_panel(left_frame)
        self.create_visualization_panel(right_frame)
        
    def create_control_panel(self, parent):
        control_frame = ttk.LabelFrame(parent, text="Control Panel", padding=10)
        control_frame.pack(fill=tk.BOTH, expand=True)
        
        dataset_frame = ttk.LabelFrame(control_frame, text="Dataset Setup", padding=10)
        dataset_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(dataset_frame, text="Select BreakHis Dataset Folder", 
                  command=self.select_dataset, width=30).pack(pady=2)
        
        self.dataset_label = ttk.Label(dataset_frame, text="No dataset selected", style='Info.TLabel')
        self.dataset_label.pack(pady=2)
        
        mag_frame = ttk.Frame(dataset_frame)
        mag_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(mag_frame, text="Magnification:", style='Info.TLabel').pack(side=tk.LEFT)
        self.mag_var = tk.StringVar(value="400X")
        mag_combo = ttk.Combobox(mag_frame, textvariable=self.mag_var, 
                                values=["40X", "100X", "200X", "400X"], 
                                state="readonly", width=10)
        mag_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        self.load_btn = ttk.Button(dataset_frame, text="Load Dataset", 
                                  command=self.load_dataset, state="disabled", width=30)
        self.load_btn.pack(pady=2)
        
        self.dataset_info = tk.Text(dataset_frame, height=6, width=35, font=('Courier', 8))
        self.dataset_info.pack(pady=2, fill=tk.BOTH, expand=True)
        
        training_frame = ttk.LabelFrame(control_frame, text="Model Training", padding=10)
        training_frame.pack(fill=tk.X, pady=(0, 10))
        
        params_frame = ttk.Frame(training_frame)
        params_frame.pack(fill=tk.X)
        
        ttk.Label(params_frame, text="Epochs:", style='Info.TLabel').grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.epochs_var = tk.StringVar(value="30")
        ttk.Entry(params_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=(0, 10))
        
        ttk.Label(params_frame, text="LR:", style='Info.TLabel').grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.lr_var = tk.StringVar(value="0.0001")
        ttk.Entry(params_frame, textvariable=self.lr_var, width=10).grid(row=0, column=3)
        
        self.train_btn = ttk.Button(training_frame, text="Train Multi-Class Model", 
                                   command=self.train_model, state="disabled", width=30)
        self.train_btn.pack(pady=(10, 0))
        
        self.training_status = ttk.Label(training_frame, text="", style='Info.TLabel')
        self.training_status.pack(pady=2)
        
        model_frame = ttk.LabelFrame(control_frame, text="Model Management", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(model_frame, text="Save Model", command=self.save_model, width=30).pack(pady=2)
        ttk.Button(model_frame, text="Load Model", command=self.load_model, width=30).pack(pady=2)
        ttk.Button(model_frame, text="Evaluate Model", command=self.evaluate_model, width=30).pack(pady=2)
        
        self.model_status = ttk.Label(model_frame, text="No model loaded", style='Info.TLabel')
        self.model_status.pack(pady=2)
        
        explainability_frame = ttk.LabelFrame(control_frame, text="Explainability Options", padding=10)
        explainability_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.gradcam_var = tk.BooleanVar(value=True)
        self.scorecam_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(explainability_frame, text="Enable Grad-CAM++", 
                       variable=self.gradcam_var).pack(anchor='w')
        ttk.Checkbutton(explainability_frame, text="Enable Score-CAM", 
                       variable=self.scorecam_var).pack(anchor='w')
        
        prediction_frame = ttk.LabelFrame(control_frame, text="Image Analysis", padding=10)
        prediction_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        ttk.Button(prediction_frame, text="Analyze Cancer Image", 
                  command=self.analyze_image, width=30).pack(pady=2)
        
        self.prediction_text = tk.Text(prediction_frame, height=15, width=35, font=('Courier', 8))
        self.prediction_text.pack(pady=(5, 0), fill=tk.BOTH, expand=True)
        
        scrollbar_pred = ttk.Scrollbar(prediction_frame, orient="vertical", command=self.prediction_text.yview)
        scrollbar_pred.pack(side=tk.RIGHT, fill=tk.Y)
        self.prediction_text.configure(yscrollcommand=scrollbar_pred.set)
        
        self.progress = ttk.Progressbar(control_frame, mode='determinate')
        self.progress.pack(fill=tk.X, pady=(10, 0))
        
    def create_visualization_panel(self, parent):
        viz_frame = ttk.LabelFrame(parent, text="Explainable AI Visualization Pipeline", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fig, self.axes = plt.subplots(3, 3, figsize=(18, 15))
        self.fig.suptitle('BreakHis Multi-Class Cancer Analysis with Explainable AI', fontsize=16, fontweight='bold')
        
        titles = [
            ['Original Image', 'Stain Normalized', 'Binary Mask'],
            ['Grad-CAM++ Heatmap', 'Score-CAM Heatmap', 'Combined Visualization'],
            ['Class Probabilities', 'Feature Analysis', 'Confusion Matrix']
        ]
        
        for i in range(3):
            for j in range(3):
                self.axes[i, j].set_title(titles[i][j], fontweight='bold')
                self.axes[i, j].axis('off')
                if not (i == 2 and j in [0, 2]):
                    self.axes[i, j].text(0.5, 0.5, 'No Image\nLoaded', ha='center', va='center', 
                                       fontsize=12, transform=self.axes[i, j].transAxes, color='gray')
        
        self.axes[2, 0].text(0.5, 0.5, 'No Prediction\nMade', ha='center', va='center', 
                            fontsize=12, transform=self.axes[2, 0].transAxes, color='gray')
        
        self.axes[2, 2].text(0.5, 0.5, 'No Evaluation\nPerformed', ha='center', va='center', 
                            fontsize=12, transform=self.axes[2, 2].transAxes, color='gray')
        
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def select_dataset(self):
        self.dataset_path = filedialog.askdirectory(title="Select BreakHis Dataset Folder")
        
        if self.dataset_path:
            self.dataset_label.configure(text=f"Selected: {os.path.basename(self.dataset_path)}")
            self.load_btn.configure(state="normal")
            
    def load_dataset(self):
        if not self.dataset_path:
            messagebox.showwarning("Warning", "Please select dataset path first")
            return
            
        try:
            self.progress.configure(mode='indeterminate')
            self.progress.start()
            
            self.analyzer = BreakHisAnalyzer(self.dataset_path)
            class_counts = self.analyzer.load_dataset(self.mag_var.get())
            self.analyzer.split_dataset()
            
            self.train_btn.configure(state="normal")
            self.progress.stop()
            self.progress.configure(mode='determinate')
            
            info_text = f"Dataset loaded - {self.mag_var.get()}\n"
            info_text += f"{'='*35}\n"
            info_text += f"CLASS DISTRIBUTION:\n"
            
            for class_name, count in class_counts.items():
                display_name = self.analyzer.class_mapping.get(class_name, class_name)
                info_text += f"{display_name}: {count}\n"
            
            total_images = sum(class_counts.values())
            info_text += f"\nTotal Images: {total_images}\n"
            info_text += f"Training: {len(self.analyzer.X_train)}\n"
            info_text += f"Testing: {len(self.analyzer.X_test)}"
            
            self.dataset_info.delete(1.0, tk.END)
            self.dataset_info.insert(tk.END, info_text)
            
        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
            
    def train_model(self):
        if not self.analyzer:
            messagebox.showwarning("Warning", "Please load dataset first")
            return
            
        try:
            epochs = int(self.epochs_var.get())
            learning_rate = float(self.lr_var.get())
            
            self.progress.configure(mode='determinate', maximum=epochs)
            self.progress['value'] = 0
            
            def progress_callback(current_epoch, total_epochs, train_acc, test_acc):
                self.progress['value'] = current_epoch
                status_text = f"Epoch {current_epoch}/{total_epochs}\n"
                status_text += f"Train Acc: {train_acc:.2f}%\n"
                status_text += f"Test Acc: {test_acc:.2f}%"
                self.training_status.configure(text=status_text)
                self.root.update()
            
            self.training_status.configure(text="Multi-class training started...")
            best_accuracy = self.analyzer.train_model(
                epochs=epochs, 
                learning_rate=learning_rate, 
                progress_callback=progress_callback
            )
            
            self.training_status.configure(text=f"Training completed!\nBest Accuracy: {best_accuracy:.2f}%")
            self.model_status.configure(text="Multi-class model trained with XAI", style='Success.TLabel')
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.training_status.configure(text="Training failed!")
            
    def evaluate_model(self):
        if not self.analyzer or self.analyzer.model is None:
            messagebox.showwarning("Warning", "Please train or load a model first")
            return
            
        try:
            self.progress.configure(mode='indeterminate')
            self.progress.start()
            
            accuracy, report, cm, class_names = self.analyzer.evaluate_model()
            
            self.progress.stop()
            
            eval_text = f"Multi-Class Evaluation Results\n"
            eval_text += f"{'='*40}\n"
            eval_text += f"Overall Accuracy: {accuracy:.4f}\n\n"
            
            lines = report.split('\n')
            eval_text += f"Per-Class Results:\n"
            for line in lines:
                if any(cls in line for cls in class_names):
                    eval_text += f"{line}\n"
            
            eval_text += f"\nMacro Avg:\n"
            for line in lines:
                if 'macro avg' in line:
                    eval_text += f"{line}\n"
            
            self.prediction_text.delete(1.0, tk.END)
            self.prediction_text.insert(tk.END, eval_text)
            
            self.plot_confusion_matrix(cm, class_names)
            
        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Error", f"Evaluation failed: {str(e)}")
            
    def plot_confusion_matrix(self, cm, class_names):
        self.axes[2, 2].clear()
        
        short_names = [name.split(' - ')[-1] if ' - ' in name else name for name in class_names]
        
        im = self.axes[2, 2].imshow(cm, interpolation='nearest', cmap='Blues')
        self.axes[2, 2].set_title('Confusion Matrix', fontweight='bold')
        
        tick_marks = np.arange(len(class_names))
        self.axes[2, 2].set_xticks(tick_marks)
        self.axes[2, 2].set_yticks(tick_marks)
        self.axes[2, 2].set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
        self.axes[2, 2].set_yticklabels(short_names, fontsize=8)
        
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            self.axes[2, 2].text(j, i, format(cm[i, j], 'd'),
                               ha="center", va="center", fontsize=8,
                               color="white" if cm[i, j] > thresh else "black")
        
        self.axes[2, 2].set_ylabel('True Label', fontweight='bold')
        self.axes[2, 2].set_xlabel('Predicted Label', fontweight='bold')
        
        self.canvas.draw()
            
    def analyze_image(self):
        if not self.analyzer or self.analyzer.model is None:
            messagebox.showwarning("Warning", "Please train or load a model first")
            return
            
        image_path = filedialog.askopenfilename(
            title="Select Cancer Image for Analysis",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        )
        
        if image_path:
            try:
                self.progress.configure(mode='indeterminate')
                self.progress.start()
                
                result = self.analyzer.process_and_predict_image(image_path)
                
                if result:
                    self.display_analysis_results(result)
                    self.update_prediction_text(result, os.path.basename(image_path))
                
                self.progress.stop()
                
            except Exception as e:
                self.progress.stop()
                messagebox.showerror("Error", f"Analysis failed: {str(e)}")
                
    def display_analysis_results(self, result):
        for i in range(3):
            for j in range(3):
                if not (i == 2 and j in [0, 2]):
                    self.axes[i, j].clear()
                    self.axes[i, j].axis('off')
        
        self.axes[0, 0].imshow(result['original_image'])
        self.axes[0, 0].set_title('Original Image', fontweight='bold')
        
        self.axes[0, 1].imshow(result['normalized_image'])
        self.axes[0, 1].set_title('Stain Normalized', fontweight='bold')
        
        self.axes[0, 2].imshow(result['binary_mask'], cmap='gray')
        self.axes[0, 2].set_title('Binary Mask', fontweight='bold')
        
        explainability_maps = result['explainability_maps']
        
        if explainability_maps.get('gradcam++') is not None:
            heatmap = explainability_maps['gradcam++']
            self.axes[1, 0].imshow(result['normalized_image'])
            self.axes[1, 0].imshow(heatmap, alpha=0.4, cmap='jet')
            self.axes[1, 0].set_title('Grad-CAM++ Heatmap', fontweight='bold')
        else:
            self.axes[1, 0].text(0.5, 0.5, 'Grad-CAM++\nUnavailable', ha='center', va='center', 
                               fontsize=12, transform=self.axes[1, 0].transAxes, color='red')
        
        if explainability_maps.get('scorecam') is not None:
            heatmap = explainability_maps['scorecam']
            self.axes[1, 1].imshow(result['normalized_image'])
            self.axes[1, 1].imshow(heatmap, alpha=0.4, cmap='plasma')
            self.axes[1, 1].set_title('Score-CAM Heatmap', fontweight='bold')
        else:
            self.axes[1, 1].text(0.5, 0.5, 'Score-CAM\nUnavailable', ha='center', va='center', 
                               fontsize=12, transform=self.axes[1, 1].transAxes, color='red')
        
        if (explainability_maps.get('gradcam++') is not None and 
            explainability_maps.get('scorecam') is not None):
            combined_heatmap = (explainability_maps['gradcam++'] + explainability_maps['scorecam']) / 2
            self.axes[1, 2].imshow(result['normalized_image'])
            self.axes[1, 2].imshow(combined_heatmap, alpha=0.5, cmap='viridis')
            self.axes[1, 2].set_title('Combined XAI Visualization', fontweight='bold')
        else:
            self.axes[1, 2].text(0.5, 0.5, 'Combined XAI\nUnavailable', ha='center', va='center', 
                               fontsize=12, transform=self.axes[1, 2].transAxes, color='orange')
        
        pred = result['prediction']
        self.plot_class_probabilities(pred['probabilities'])
        self.plot_feature_analysis(result['features'])
        
        self.fig.suptitle(f'XAI Cancer Classification: {pred["class"]} (Confidence: {pred["confidence"]:.3f})', 
                         fontsize=16, fontweight='bold')
        
        for i in range(3):
            for j in range(3):
                if not (i == 2 and j in [0, 2]):
                    self.axes[i, j].axis('off')
        
        self.canvas.draw()
        
    def plot_class_probabilities(self, probabilities):
        self.axes[2, 0].clear()
        
        classes = list(probabilities.keys())
        probs = list(probabilities.values())
        
        colors = ['green' if 'Benign' in cls else 'red' for cls in classes]
        
        bars = self.axes[2, 0].bar(range(len(classes)), probs, color=colors, alpha=0.7)
        
        self.axes[2, 0].set_xlabel('Cancer Types', fontweight='bold')
        self.axes[2, 0].set_ylabel('Probability', fontweight='bold')
        self.axes[2, 0].set_title('Class Probabilities', fontweight='bold')
        
        short_labels = []
        for cls in classes:
            if ' - ' in cls:
                short_labels.append(cls.split(' - ')[-1])
            else:
                short_labels.append(cls)
        
        self.axes[2, 0].set_xticks(range(len(classes)))
        self.axes[2, 0].set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
        
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            self.axes[2, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
        
        self.axes[2, 0].set_ylim(0, 1.0)
        self.axes[2, 0].grid(True, alpha=0.3)
        
    def plot_feature_analysis(self, features):
        self.axes[2, 1].clear()
        
        if not features:
            self.axes[2, 1].text(0.5, 0.5, 'No Features\nExtracted', ha='center', va='center', 
                               fontsize=12, transform=self.axes[2, 1].transAxes, color='gray')
            return
        
        feature_names = list(features.keys())
        feature_values = list(features.values())
        
        normalized_values = []
        for val in feature_values:
            if isinstance(val, (int, float)) and not np.isnan(val):
                normalized_values.append(val)
            else:
                normalized_values.append(0)
        
        if len(normalized_values) > 0:
            max_val = max(normalized_values) if max(normalized_values) > 0 else 1
            normalized_values = [val / max_val for val in normalized_values]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
        bars = self.axes[2, 1].barh(range(len(feature_names)), normalized_values, color=colors)
        
        self.axes[2, 1].set_yticks(range(len(feature_names)))
        self.axes[2, 1].set_yticklabels(feature_names, fontsize=8)
        self.axes[2, 1].set_xlabel('Normalized Values', fontweight='bold')
        self.axes[2, 1].set_title('Morphological Features', fontweight='bold')
        
        for bar, val in zip(bars, normalized_values):
            width = bar.get_width()
            self.axes[2, 1].text(width + 0.01, bar.get_y() + bar.get_height()/2,
                               f'{val:.3f}', ha='left', va='center', fontsize=7)
        
        self.axes[2, 1].set_xlim(0, 1.2)
        self.axes[2, 1].grid(True, alpha=0.3, axis='x')
        
    def update_prediction_text(self, result, filename):
        pred = result['prediction']
        features = result['features']
        explainability_maps = result['explainability_maps']
        
        analysis_text = f"XAI Cancer Analysis: {filename}\n"
        analysis_text += f"{'='*50}\n\n"
        
        analysis_text += f"CLASSIFICATION RESULT:\n"
        analysis_text += f"Diagnosed Type: {pred['class']}\n"
        analysis_text += f"Confidence: {pred['confidence']:.4f}\n\n"
        
        analysis_text += f"CLASS PROBABILITIES:\n"
        sorted_probs = sorted(pred['probabilities'].items(), key=lambda x: x[1], reverse=True)
        for class_name, prob in sorted_probs:
            analysis_text += f"{class_name}: {prob:.4f}\n"
        
        if features:
            analysis_text += f"\nMORPHOLOGICAL FEATURES:\n"
            analysis_text += f"Area: {features.get('area', 'N/A')}\n"
            analysis_text += f"Perimeter: {features.get('perimeter', 'N/A'):.2f}\n"
            analysis_text += f"Eccentricity: {features.get('eccentricity', 'N/A'):.4f}\n"
            analysis_text += f"Solidity: {features.get('solidity', 'N/A'):.4f}\n"
            analysis_text += f"Extent: {features.get('extent', 'N/A'):.4f}\n"
            analysis_text += f"Mean Intensity: {features.get('mean_intensity', 'N/A'):.2f}\n"
            analysis_text += f"Compactness: {features.get('compactness', 'N/A'):.4f}\n"
        
        analysis_text += f"\nEXPLAINABILITY ANALYSIS:\n"
        if explainability_maps.get('gradcam++') is not None:
            analysis_text += f"✓ Grad-CAM++ heatmap generated\n"
            analysis_text += f"  - Shows gradient-weighted activations\n"
            analysis_text += f"  - Highlights discriminative regions\n"
        else:
            analysis_text += f"✗ Grad-CAM++ generation failed\n"
        
        if explainability_maps.get('scorecam') is not None:
            analysis_text += f"✓ Score-CAM heatmap generated\n"
            analysis_text += f"  - Shows activation-based importance\n"
            analysis_text += f"  - Model-agnostic visualization\n"
        else:
            analysis_text += f"✗ Score-CAM generation failed\n"
        
        analysis_text += f"\nPROCESSING PIPELINE:\n"
        analysis_text += f"1. ✓ Original histology image loaded\n"
        analysis_text += f"2. ✓ Macenko stain normalization\n"
        analysis_text += f"3. ✓ Binary tissue mask generated\n"
        analysis_text += f"4. ✓ Morphological features extracted\n"
        analysis_text += f"5. ✓ Multi-class CNN classification\n"
        analysis_text += f"6. ✓ Explainable AI heatmaps generated\n"
        
        tumor_type = "MALIGNANT" if "Malignant" in pred['class'] else "BENIGN"
        confidence_level = "HIGH" if pred['confidence'] > 0.8 else "MEDIUM" if pred['confidence'] > 0.6 else "LOW"
        
        analysis_text += f"\nCLINICAL ASSESSMENT:\n"
        analysis_text += f"Tumor Category: {tumor_type}\n"
        analysis_text += f"Specific Type: {pred['class'].split(' - ')[-1] if ' - ' in pred['class'] else pred['class']}\n"
        analysis_text += f"Diagnostic Confidence: {confidence_level}\n"
        analysis_text += f"XAI Support: {'Available' if any(explainability_maps.values()) else 'Limited'}\n"
        
        if tumor_type == "MALIGNANT":
            analysis_text += f"\n⚠️  MALIGNANT TISSUE DETECTED\n"
            analysis_text += f"   Requires immediate clinical attention\n"
            analysis_text += f"   XAI heatmaps show critical regions\n"
        else:
            analysis_text += f"\n✓  BENIGN TISSUE DETECTED\n"
            analysis_text += f"   Continue routine monitoring\n"
            analysis_text += f"   XAI confirms benign characteristics\n"
        
        analysis_text += f"\nXAI INTERPRETATION GUIDE:\n"
        analysis_text += f"• Red/Hot regions: High importance areas\n"
        analysis_text += f"• Blue/Cool regions: Low importance areas\n"
        analysis_text += f"• Grad-CAM++: Gradient-based attribution\n"
        analysis_text += f"• Score-CAM: Activation-based attribution\n"
        analysis_text += f"• Combined view: Consensus importance\n"
        
        self.prediction_text.delete(1.0, tk.END)
        self.prediction_text.insert(tk.END, analysis_text)
        
    def save_model(self):
        if not self.analyzer or self.analyzer.model is None:
            messagebox.showwarning("Warning", "No model to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Multi-Class XAI Model",
            defaultextension=".pth",
            filetypes=[("PyTorch files", "*.pth")]
        )
        
        if file_path:
            try:
                success = self.analyzer.save_model(file_path)
                if success:
                    messagebox.showinfo("Success", "Multi-class XAI model saved successfully!")
                    self.model_status.configure(text=f"XAI Model saved: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")
                
    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="Load Multi-Class XAI Model",
            filetypes=[("PyTorch files", "*.pth")]
        )
        
        if file_path:
            try:
                if not self.analyzer:
                    self.analyzer = BreakHisAnalyzer("")
                
                success = self.analyzer.load_model(file_path)
                if success:
                    messagebox.showinfo("Success", "Multi-class XAI model loaded successfully!")
                    self.model_status.configure(text=f"XAI Model loaded: {os.path.basename(file_path)}", 
                                              style='Success.TLabel')
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                self.model_status.configure(text="Model load failed", style='Error.TLabel')

def main():
    root = tk.Tk()
    app = BreakHisGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Application Error", f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()