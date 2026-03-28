import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
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
from skimage import filters, morphology, segmentation
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
    def __init__(self, num_classes=2, efficientnet_version='b0'):
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_encoder = LabelEncoder()
        self.image_paths = []
        self.labels = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
        
    def load_dataset(self):
        print("Loading BreakHis dataset...")
        
        benign_path = os.path.join(self.dataset_path, "**", "*benign*", "*.png")
        malignant_path = os.path.join(self.dataset_path, "**", "*malignant*", "*.png")
        
        benign_files = glob.glob(benign_path, recursive=True)
        malignant_files = glob.glob(malignant_path, recursive=True)
        
        if not benign_files and not malignant_files:
            benign_path = os.path.join(self.dataset_path, "**", "*B*", "*.png")
            malignant_path = os.path.join(self.dataset_path, "**", "*M*", "*.png")
            benign_files = glob.glob(benign_path, recursive=True)
            malignant_files = glob.glob(malignant_path, recursive=True)
        
        if not benign_files and not malignant_files:
            for root, dirs, files in os.walk(self.dataset_path):
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(root, file)
                        if 'benign' in file.lower() or 'B' in os.path.basename(root):
                            benign_files.append(file_path)
                        elif 'malignant' in file.lower() or 'M' in os.path.basename(root):
                            malignant_files.append(file_path)
        
        self.image_paths = benign_files + malignant_files
        self.labels = ['benign'] * len(benign_files) + ['malignant'] * len(malignant_files)
        
        print(f"Found {len(benign_files)} benign images")
        print(f"Found {len(malignant_files)} malignant images")
        print(f"Total images: {len(self.image_paths)}")
        
        self.labels = self.label_encoder.fit_transform(self.labels)
        
    def split_dataset(self, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.image_paths, self.labels, test_size=test_size, 
            random_state=42, stratify=self.labels
        )
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Testing samples: {len(self.X_test)}")
        
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
            shuffle=True, num_workers=4
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, 
            shuffle=False, num_workers=4
        )
        
        return train_loader, test_loader
    
    def train_model(self, epochs=50, learning_rate=0.001):
        print("Initializing model...")
        self.model = BreakHisClassifier(num_classes=2).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        train_loader, test_loader = self.create_dataloaders()
        
        print("Starting training...")
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
                
                if batch_idx % 50 == 0:
                    print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
            
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
            
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%')
            print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Acc: {test_accuracy:.2f}%')
            print('-' * 60)
            
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(self.model.state_dict(), 'best_breakhis_model.pth')
                print(f'New best model saved with accuracy: {best_accuracy:.2f}%')
        
        print(f'Training completed. Best accuracy: {best_accuracy:.2f}%')
        
    def evaluate_model(self):
        if self.model is None:
            print("Model not trained yet!")
            return
            
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
        print(f"\nFinal Test Accuracy: {accuracy:.4f}")
        
        class_names = self.label_encoder.classes_
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        
        return accuracy, y_true, y_pred
    
    def predict_image(self, image_path):
        if self.model is None:
            print("Model not trained yet!")
            return None
            
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        normalized_image = self.normalizer.normalize_he(image)
        
        image_tensor = self.transform(normalized_image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = output.argmax(1).item()
            confidence = probabilities[0][predicted_class].item()
        
        class_name = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return {
            'class': class_name,
            'confidence': confidence,
            'probabilities': {
                'benign': probabilities[0][0].item(),
                'malignant': probabilities[0][1].item()
            }
        }
    
    def save_model(self, filepath):
        if self.model is None:
            print("No model to save!")
            return
            
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_encoder': self.label_encoder,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model = BreakHisClassifier(num_classes=2).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.label_encoder = checkpoint['label_encoder']
        
        print(f"Model loaded from {filepath}")

class BreakHisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BreakHis Cancer Classification")
        self.root.geometry("900x700")
        
        self.analyzer = None
        self.dataset_path = None
        
        self.create_widgets()
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        title_label = ttk.Label(main_frame, text="BreakHis Breast Cancer Classification", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        dataset_frame = ttk.LabelFrame(main_frame, text="Dataset Setup", padding="10")
        dataset_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(dataset_frame, text="Select BreakHis Dataset Folder", 
                  command=self.select_dataset).grid(row=0, column=0, padx=(0, 5))
        
        self.dataset_label = ttk.Label(dataset_frame, text="No dataset selected")
        self.dataset_label.grid(row=0, column=1, padx=(5, 0))
        
        ttk.Button(dataset_frame, text="Load Dataset", 
                  command=self.load_dataset, state="disabled").grid(row=1, column=0, pady=(10, 0))
        
        self.load_btn = ttk.Button(dataset_frame, text="Load Dataset", 
                                  command=self.load_dataset, state="disabled")
        self.load_btn.grid(row=1, column=0, pady=(10, 0))
        
        training_frame = ttk.LabelFrame(main_frame, text="Model Training", padding="10")
        training_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(training_frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W)
        self.epochs_var = tk.StringVar(value="20")
        ttk.Entry(training_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=(5, 10))
        
        ttk.Label(training_frame, text="Learning Rate:").grid(row=0, column=2, sticky=tk.W)
        self.lr_var = tk.StringVar(value="0.001")
        ttk.Entry(training_frame, textvariable=self.lr_var, width=10).grid(row=0, column=3, padx=(5, 0))
        
        self.train_btn = ttk.Button(training_frame, text="Train Model", 
                                   command=self.train_model, state="disabled")
        self.train_btn.grid(row=1, column=0, columnspan=4, pady=(10, 0))
        
        model_frame = ttk.LabelFrame(main_frame, text="Model Management", padding="10")
        model_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(model_frame, text="Save Model", 
                  command=self.save_model).grid(row=0, column=0, padx=(0, 5))
        
        ttk.Button(model_frame, text="Load Model", 
                  command=self.load_model).grid(row=0, column=1, padx=(5, 5))
        
        ttk.Button(model_frame, text="Evaluate Model", 
                  command=self.evaluate_model).grid(row=0, column=2, padx=(5, 0))
        
        prediction_frame = ttk.LabelFrame(main_frame, text="Single Image Prediction", padding="10")
        prediction_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(prediction_frame, text="Select Image for Prediction", 
                  command=self.predict_single_image).grid(row=0, column=0, columnspan=3)
        
        self.prediction_result = ttk.Label(prediction_frame, text="No prediction made")
        self.prediction_result.grid(row=1, column=0, columnspan=3, pady=(10, 0))
        
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.log_text = tk.Text(main_frame, height=15, width=80)
        self.log_text.grid(row=6, column=0, columnspan=3, pady=(10, 0))
        
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=6, column=3, sticky=(tk.N, tk.S), pady=(10, 0))
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
    def log_message(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()
        
    def select_dataset(self):
        self.dataset_path = filedialog.askdirectory(title="Select BreakHis Dataset Folder")
        
        if self.dataset_path:
            self.dataset_label.configure(text=f"Selected: {os.path.basename(self.dataset_path)}")
            self.load_btn.configure(state="normal")
            self.log_message(f"Dataset path selected: {self.dataset_path}")
            
    def load_dataset(self):
        if not self.dataset_path:
            messagebox.showwarning("Warning", "Please select dataset path first")
            return
            
        try:
            self.progress.start()
            self.log_message("Loading BreakHis dataset...")
            
            self.analyzer = BreakHisAnalyzer(self.dataset_path)
            self.analyzer.load_dataset()
            self.analyzer.split_dataset()
            
            self.train_btn.configure(state="normal")
            self.progress.stop()
            self.log_message("Dataset loaded successfully!")
            self.log_message(f"Total images: {len(self.analyzer.image_paths)}")
            self.log_message(f"Training samples: {len(self.analyzer.X_train)}")
            self.log_message(f"Testing samples: {len(self.analyzer.X_test)}")
            
        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
            self.log_message(f"Error loading dataset: {str(e)}")
            
    def train_model(self):
        if not self.analyzer:
            messagebox.showwarning("Warning", "Please load dataset first")
            return
            
        try:
            epochs = int(self.epochs_var.get())
            learning_rate = float(self.lr_var.get())
            
            self.progress.start()
            self.log_message(f"Starting training with {epochs} epochs and LR {learning_rate}")
            
            self.analyzer.train_model(epochs=epochs, learning_rate=learning_rate)
            
            self.progress.stop()
            self.log_message("Training completed!")
            
        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.log_message(f"Training error: {str(e)}")
            
    def evaluate_model(self):
        if not self.analyzer or self.analyzer.model is None:
            messagebox.showwarning("Warning", "Please train or load a model first")
            return
            
        try:
            self.progress.start()
            self.log_message("Evaluating model...")
            
            accuracy, y_true, y_pred = self.analyzer.evaluate_model()
            
            self.progress.stop()
            self.log_message(f"Evaluation completed! Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Error", f"Evaluation failed: {str(e)}")
            self.log_message(f"Evaluation error: {str(e)}")
            
    def predict_single_image(self):
        if not self.analyzer or self.analyzer.model is None:
            messagebox.showwarning("Warning", "Please train or load a model first")
            return
            
        image_path = filedialog.askopenfilename(
            title="Select Image for Prediction",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        )
        
        if image_path:
            try:
                self.progress.start()
                self.log_message(f"Predicting image: {os.path.basename(image_path)}")
                
                result = self.analyzer.predict_image(image_path)
                
                if result:
                    prediction_text = f"Prediction: {result['class'].upper()}\n"
                    prediction_text += f"Confidence: {result['confidence']:.4f}\n"
                    prediction_text += f"Benign: {result['probabilities']['benign']:.4f}\n"
                    prediction_text += f"Malignant: {result['probabilities']['malignant']:.4f}"
                    
                    self.prediction_result.configure(text=prediction_text)
                    self.log_message(f"Prediction: {result['class']} (confidence: {result['confidence']:.4f})")
                
                self.progress.stop()
                
            except Exception as e:
                self.progress.stop()
                messagebox.showerror("Error", f"Prediction failed: {str(e)}")
                self.log_message(f"Prediction error: {str(e)}")
                
    def save_model(self):
        if not self.analyzer or self.analyzer.model is None:
            messagebox.showwarning("Warning", "No model to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".pth",
            filetypes=[("PyTorch files", "*.pth")]
        )
        
        if file_path:
            try:
                self.analyzer.save_model(file_path)
                self.log_message(f"Model saved to: {file_path}")
                messagebox.showinfo("Success", "Model saved successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")
                self.log_message(f"Save error: {str(e)}")
                
    def load_model(self):
        if not self.analyzer:
            messagebox.showwarning("Warning", "Please set up dataset first")
            return
            
        file_path = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("PyTorch files", "*.pth")]
        )
        
        if file_path:
            try:
                self.analyzer.load_model(file_path)
                self.log_message(f"Model loaded from: {file_path}")
                messagebox.showinfo("Success", "Model loaded successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                self.log_message(f"Load error: {str(e)}")

def main():
    root = tk.Tk()
    app = BreakHisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()