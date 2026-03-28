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
            'Adenosis': 'Benign - Adenosis',
            'Fibroadenoma': 'Benign - Fibroadenoma', 
            'Phyllodes_Tumor': 'Benign - Phyllodes Tumor',
            'Tubular_Adenoma': 'Benign - Tubular Adenoma',
            'Ductal_Carcinoma': 'Malignant - Ductal Carcinoma',
            'Lobular_Carcinoma': 'Malignant - Lobular Carcinoma',
            'Mucinous_Carcinoma': 'Malignant - Mucinous Carcinoma',
            'Papillary_Carcinoma': 'Malignant - Papillary Carcinoma'
        }
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
        
    def load_dataset(self, selected_magnification='400X'):
        class_counts = {}
        
        sob_path = os.path.join(self.dataset_path, "SOB")
        if not os.path.exists(sob_path):
            sob_path = self.dataset_path
        
        benign_path = os.path.join(sob_path, "Benign")
        malignant_path = os.path.join(sob_path, "Malignant")
        
        if not os.path.exists(benign_path) or not os.path.exists(malignant_path):
            raise ValueError("Dataset structure not found. Expected BreakHis/SOB/Benign and BreakHis/SOB/Malignant folders")
        
        benign_types = ['Adenosis', 'Fibroadenoma', 'Phyllodes_Tumor', 'Tubular_Adenoma']
        malignant_types = ['Ductal_Carcinoma', 'Lobular_Carcinoma', 'Mucinous_Carcinoma', 'Papillary_Carcinoma']
        
        for cancer_type in benign_types:
            type_path = os.path.join(benign_path, cancer_type, selected_magnification)
            if os.path.exists(type_path):
                images = glob.glob(os.path.join(type_path, "*.png"))
                self.image_paths.extend(images)
                self.labels.extend([cancer_type] * len(images))
                self.magnifications.extend([selected_magnification] * len(images))
                class_counts[cancer_type] = len(images)
        
        for cancer_type in malignant_types:
            type_path = os.path.join(malignant_path, cancer_type, selected_magnification)
            if os.path.exists(type_path):
                images = glob.glob(os.path.join(type_path, "*.png"))
                self.image_paths.extend(images)
                self.labels.extend([cancer_type] * len(images))
                self.magnifications.extend([selected_magnification] * len(images))
                class_counts[cancer_type] = len(images)
        
        if not self.image_paths:
            raise ValueError(f"No images found for magnification {selected_magnification}")
        
        self.labels = self.label_encoder.fit_transform(self.labels)
        
        return class_counts
        
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
        
        return True

class BreakHisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BreakHis Multi-Class Cancer Classification System")
        self.root.geometry("1600x1000")
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
        
        ttk.Label(title_frame, text="BreakHis Multi-Class Breast Cancer Classification System", 
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
        viz_frame = ttk.LabelFrame(parent, text="Multi-Class Image Processing Pipeline", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.suptitle('BreakHis Multi-Class Cancer Analysis Pipeline', fontsize=14, fontweight='bold')
        
        self.axes[0, 0].set_title('Original Image')
        self.axes[0, 1].set_title('Stain Normalized')
        self.axes[0, 2].set_title('Binary Mask')
        self.axes[1, 0].set_title('Features Overlay')
        self.axes[1, 1].set_title('Class Probabilities')
        self.axes[1, 2].set_title('Confusion Matrix')
        
        for ax in self.axes.flat:
            ax.axis('off')
            if ax != self.axes[1, 1] and ax != self.axes[1, 2]:
                ax.text(0.5, 0.5, 'No Image\nLoaded', ha='center', va='center', 
                       fontsize=12, transform=ax.transAxes, color='gray')
        
        self.axes[1, 1].text(0.5, 0.5, 'No Prediction\nMade', ha='center', va='center', 
                            fontsize=12, transform=self.axes[1, 1].transAxes, color='gray')
        
        self.axes[1, 2].text(0.5, 0.5, 'No Evaluation\nPerformed', ha='center', va='center', 
                            fontsize=12, transform=self.axes[1, 2].transAxes, color='gray')
        
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
            info_text += f"BENIGN CLASSES:\n"
            info_text += f"Adenosis: {class_counts.get('Adenosis', 0)}\n"
            info_text += f"Fibroadenoma: {class_counts.get('Fibroadenoma', 0)}\n"
            info_text += f"Phyllodes Tumor: {class_counts.get('Phyllodes_Tumor', 0)}\n"
            info_text += f"Tubular Adenoma: {class_counts.get('Tubular_Adenoma', 0)}\n\n"
            info_text += f"MALIGNANT CLASSES:\n"
            info_text += f"Ductal Carcinoma: {class_counts.get('Ductal_Carcinoma', 0)}\n"
            info_text += f"Lobular Carcinoma: {class_counts.get('Lobular_Carcinoma', 0)}\n"
            info_text += f"Mucinous Carcinoma: {class_counts.get('Mucinous_Carcinoma', 0)}\n"
            info_text += f"Papillary Carcinoma: {class_counts.get('Papillary_Carcinoma', 0)}\n\n"
            
            total_images = sum(class_counts.values())
            info_text += f"Total Images: {total_images}\n"
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
            self.model_status.configure(text="Multi-class model trained", style='Success.TLabel')
            
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
        self.axes[1, 2].clear()
        
        short_names = [name.split(' - ')[-1] if ' - ' in name else name for name in class_names]
        
        im = self.axes[1, 2].imshow(cm, interpolation='nearest', cmap='Blues')
        self.axes[1, 2].set_title('Confusion Matrix', fontweight='bold')
        
        tick_marks = np.arange(len(class_names))
        self.axes[1, 2].set_xticks(tick_marks)
        self.axes[1, 2].set_yticks(tick_marks)
        self.axes[1, 2].set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
        self.axes[1, 2].set_yticklabels(short_names, fontsize=8)
        
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            self.axes[1, 2].text(j, i, format(cm[i, j], 'd'),
                               ha="center", va="center", fontsize=8,
                               color="white" if cm[i, j] > thresh else "black")
        
        self.axes[1, 2].set_ylabel('True Label', fontweight='bold')
        self.axes[1, 2].set_xlabel('Predicted Label', fontweight='bold')
        
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
        for i in range(2):
            for j in range(2):
                self.axes[i, j].clear()
                self.axes[i, j].axis('off')
        
        self.axes[0, 0].imshow(result['original_image'])
        self.axes[0, 0].set_title('Original Image', fontweight='bold')
        
        self.axes[0, 1].imshow(result['normalized_image'])
        self.axes[0, 1].set_title('Stain Normalized', fontweight='bold')
        
        self.axes[0, 2].imshow(result['binary_mask'], cmap='gray')
        self.axes[0, 2].set_title('Binary Mask', fontweight='bold')
        
        overlay = result['normalized_image'].copy()
        mask_colored = np.zeros_like(overlay)
        mask_colored[result['binary_mask'] == 0] = [255, 0, 0]
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        self.axes[1, 0].imshow(overlay)
        self.axes[1, 0].set_title('Features Overlay', fontweight='bold')
        
        pred = result['prediction']
        self.plot_class_probabilities(pred['probabilities'])
        
        self.fig.suptitle(f'Cancer Classification: {pred["class"]} (Confidence: {pred["confidence"]:.3f})', 
                         fontsize=14, fontweight='bold')
        
        for i in range(2):
            for j in range(3):
                if i < 2 and j < 3:
                    if not (i == 1 and j in [1, 2]):
                        self.axes[i, j].axis('off')
        
        self.canvas.draw()
        
    def plot_class_probabilities(self, probabilities):
        self.axes[1, 1].clear()
        
        classes = list(probabilities.keys())
        probs = list(probabilities.values())
        
        colors = ['green' if 'Benign' in cls else 'red' for cls in classes]
        
        bars = self.axes[1, 1].bar(range(len(classes)), probs, color=colors, alpha=0.7)
        
        self.axes[1, 1].set_xlabel('Cancer Types', fontweight='bold')
        self.axes[1, 1].set_ylabel('Probability', fontweight='bold')
        self.axes[1, 1].set_title('Class Probabilities', fontweight='bold')
        
        short_labels = []
        for cls in classes:
            if ' - ' in cls:
                short_labels.append(cls.split(' - ')[-1])
            else:
                short_labels.append(cls)
        
        self.axes[1, 1].set_xticks(range(len(classes)))
        self.axes[1, 1].set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
        
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            self.axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
        
        self.axes[1, 1].set_ylim(0, 1.0)
        self.axes[1, 1].grid(True, alpha=0.3)
        
    def update_prediction_text(self, result, filename):
        pred = result['prediction']
        features = result['features']
        
        analysis_text = f"Cancer Analysis: {filename}\n"
        analysis_text += f"{'='*45}\n\n"
        
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
        
        analysis_text += f"\nPROCESSING PIPELINE:\n"
        analysis_text += f"1. ✓ Original histology image loaded\n"
        analysis_text += f"2. ✓ Macenko stain normalization\n"
        analysis_text += f"3. ✓ Binary tissue mask generated\n"
        analysis_text += f"4. ✓ Morphological features extracted\n"
        analysis_text += f"5. ✓ Multi-class CNN classification\n"
        
        tumor_type = "MALIGNANT" if "Malignant" in pred['class'] else "BENIGN"
        confidence_level = "HIGH" if pred['confidence'] > 0.8 else "MEDIUM" if pred['confidence'] > 0.6 else "LOW"
        
        analysis_text += f"\nCLINICAL ASSESSMENT:\n"
        analysis_text += f"Tumor Category: {tumor_type}\n"
        analysis_text += f"Specific Type: {pred['class'].split(' - ')[-1] if ' - ' in pred['class'] else pred['class']}\n"
        analysis_text += f"Diagnostic Confidence: {confidence_level}\n"
        
        if tumor_type == "MALIGNANT":
            analysis_text += f"⚠️  MALIGNANT TISSUE DETECTED\n"
            analysis_text += f"   Requires immediate clinical attention\n"
        else:
            analysis_text += f"✓  BENIGN TISSUE DETECTED\n"
            analysis_text += f"   Continue routine monitoring\n"
        
        self.prediction_text.delete(1.0, tk.END)
        self.prediction_text.insert(tk.END, analysis_text)
        
    def save_model(self):
        if not self.analyzer or self.analyzer.model is None:
            messagebox.showwarning("Warning", "No model to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Multi-Class Model",
            defaultextension=".pth",
            filetypes=[("PyTorch files", "*.pth")]
        )
        
        if file_path:
            try:
                success = self.analyzer.save_model(file_path)
                if success:
                    messagebox.showinfo("Success", "Multi-class model saved successfully!")
                    self.model_status.configure(text=f"Model saved: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")
                
    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="Load Multi-Class Model",
            filetypes=[("PyTorch files", "*.pth")]
        )
        
        if file_path:
            try:
                if not self.analyzer:
                    self.analyzer = BreakHisAnalyzer("")
                
                success = self.analyzer.load_model(file_path)
                if success:
                    messagebox.showinfo("Success", "Multi-class model loaded successfully!")
                    self.model_status.configure(text=f"Model loaded: {os.path.basename(file_path)}", 
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