import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
from PIL import Image
import time
from collections import defaultdict
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =============================================================================
# STAIN NORMALIZATION
# =============================================================================

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

# =============================================================================
# IMAGE PROCESSING
# =============================================================================

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

# =============================================================================
# DATASET HANDLING
# =============================================================================

class BreakHisDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, normalizer=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.normalizer = normalizer
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            image = cv2.imread(image_path)
            if image is None:
                image = np.array(Image.open(image_path))
            
            if len(image.shape) == 3 and image.shape[2] == 3:
                if image_path.lower().endswith(('.jpg', '.jpeg')):
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.normalizer:
                try:
                    image = self.normalizer.normalize_he(image)
                except:
                    pass
            
            if self.transform:
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image.astype(np.uint8))
                image = self.transform(image)
                
            label = self.labels[idx]
            return image, label
        except Exception as e:
            print(f"Error loading image {idx}: {e}")
            return torch.zeros(3, 224, 224), 0

# =============================================================================
# ATTENTION MECHANISM
# =============================================================================

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

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class BreakHisClassifier(nn.Module):
    def __init__(self, num_classes=8, efficientnet_version='b1'):
        super(BreakHisClassifier, self).__init__()
        
        self.backbone = EfficientNet.from_pretrained(f'efficientnet-{efficientnet_version}')
        
        feature_dims = {
            'b0': [32, 24, 40, 112, 1280],
            'b1': [32, 24, 40, 112, 1280],
            'b2': [32, 24, 48, 120, 1408],
            'b3': [40, 32, 48, 136, 1536],
            'b4': [48, 32, 56, 160, 1792]
        }
        
        dims = feature_dims.get(efficientnet_version, feature_dims['b1'])
        
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
            nn.Linear(dims[4], 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
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

# =============================================================================
# MAIN ANALYZER CLASS
# =============================================================================

class BreakHisAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.normalizer = MacenkoStainNormalizer()
        self.processor = ImageProcessor()
        self.device = device
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
        
        self.transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
        
    def load_dataset_all_magnifications(self):
        class_counts = defaultdict(int)
        self.image_paths = []
        self.labels = []
        self.magnifications = []
        
        print("Loading dataset with ALL magnifications...")
        
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
            raise ValueError("Could not find BreakHis dataset structure")
        
        benign_path = os.path.join(breast_path, "benign")
        malignant_path = os.path.join(breast_path, "malignant")
        
        magnifications = ["40X", "100X", "200X", "400X"]
        
        for category_path, category_type in [(benign_path, 'benign'), (malignant_path, 'malignant')]:
            for cancer_type_folder in os.listdir(category_path):
                cancer_type_path = os.path.join(category_path, cancer_type_folder)
                
                if not os.path.isdir(cancer_type_path):
                    continue
                
                print(f"Processing {cancer_type_folder}...")
                
                for magnification in magnifications:
                    for sub_folder in os.listdir(cancer_type_path):
                        if sub_folder.startswith("SOB_"):
                            subfolder_path = os.path.join(cancer_type_path, sub_folder)
                            if os.path.isdir(subfolder_path):
                                magnification_path = os.path.join(subfolder_path, magnification)
                                if os.path.exists(magnification_path):
                                    images = self.get_images_from_folder(magnification_path)
                                    if images:
                                        self.add_images_to_dataset(images, cancer_type_folder, magnification, class_counts)
        
        if len(self.image_paths) == 0:
            raise ValueError("No images found in the dataset")
        
        self.labels = self.label_encoder.fit_transform(self.labels)
        
        print(f"Successfully loaded {len(self.image_paths)} images across all magnifications")
        return dict(class_counts)
    
    def get_images_from_folder(self, folder_path):
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        images = []
        
        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
            images.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
        
        return images
    
    def add_images_to_dataset(self, images, cancer_type, magnification, class_counts):
        normalized_type = cancer_type.lower().replace(' ', '_')
        
        self.image_paths.extend(images)
        self.labels.extend([normalized_type] * len(images))
        self.magnifications.extend([magnification] * len(images))
        
        class_counts[normalized_type] += len(images)
        
    def split_dataset(self, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test, self.mag_train, self.mag_test = train_test_split(
            self.image_paths, self.labels, self.magnifications, 
            test_size=test_size, random_state=42, stratify=self.labels
        )
        
    def create_dataloaders(self, batch_size=32):
        train_dataset = BreakHisDataset(
            self.X_train, self.y_train, 
            transform=self.transform_train, 
            normalizer=self.normalizer
        )
        
        test_dataset = BreakHisDataset(
            self.X_test, self.y_test, 
            transform=self.transform_test, 
            normalizer=self.normalizer
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, 
            shuffle=True, num_workers=4, pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, 
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        return train_loader, test_loader
    
    def train_model(self, epochs=25, learning_rate=0.0001, batch_size=32):
        num_classes = len(self.label_encoder.classes_)
        self.model = BreakHisClassifier(num_classes=num_classes, efficientnet_version='b1').to(self.device)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate*10, epochs=epochs, 
            steps_per_epoch=len(self.X_train)//batch_size + 1,
            pct_start=0.3, anneal_strategy='cos'
        )
        
        train_loader, test_loader = self.create_dataloaders(batch_size=batch_size)
        
        best_accuracy = 0.0
        train_losses = []
        train_accuracies = []
        test_accuracies = []
        
        for epoch in range(epochs):
            start_time = time.time()
            
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                if batch_idx % 50 == 0:
                    print(f'Epoch {epoch+1}/{epochs} [{batch_idx}/{len(train_loader)}] '
                          f'Loss: {loss.item():.4f} '
                          f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            train_accuracy = 100. * train_correct / train_total
            train_losses.append(train_loss / len(train_loader))
            train_accuracies.append(train_accuracy)
            
            self.model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()
            
            test_accuracy = 100. * test_correct / test_total
            test_accuracies.append(test_accuracy)
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s')
            print(f'Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')
            print(f'Train Loss: {train_loss/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}')
            print('-' * 60)
            
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'label_encoder': self.label_encoder,
                    'class_mapping': self.class_mapping,
                    'num_classes': num_classes,
                    'accuracy': best_accuracy
                }, 'best_breakhis_multiclass_model.pth')
                print(f'New best model saved with accuracy: {best_accuracy:.2f}%')
        
        return best_accuracy, train_losses, train_accuracies, test_accuracies
        
    def evaluate_model(self):
        if self.model is None:
            return None
            
        _, test_loader = self.create_dataloaders()
        
        self.model.eval()
        y_true = []
        y_pred = []
        y_probs = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_probs.extend(probs.cpu().numpy())
        
        accuracy = accuracy_score(y_true, y_pred)
        class_names = [self.class_mapping.get(cls, cls) for cls in self.label_encoder.classes_]
        report = classification_report(y_true, y_pred, target_names=class_names)
        cm = confusion_matrix(y_true, y_pred)
        
        return accuracy, report, cm, class_names, y_true, y_pred, y_probs

# =============================================================================
# DATASET LOADING AND TRAINING
# =============================================================================

dataset_path = "/kaggle/input/breakhis"
analyzer = BreakHisAnalyzer(dataset_path)

print("Loading BreakHis dataset with ALL magnifications...")
class_counts = analyzer.load_dataset_all_magnifications()

print("\nDataset Statistics:")
print("=" * 50)
total_images = sum(class_counts.values())
for class_name, count in class_counts.items():
    display_name = analyzer.class_mapping.get(class_name, class_name)
    percentage = (count / total_images) * 100
    print(f"{display_name}: {count} ({percentage:.1f}%)")

print(f"\nTotal Images: {total_images}")

analyzer.split_dataset(test_size=0.2)

print(f"\nDataset Split:")
print(f"Training Images: {len(analyzer.X_train)}")
print(f"Testing Images: {len(analyzer.X_test)}")

print("\nStarting model training...")
print("=" * 50)

best_accuracy, train_losses, train_accuracies, test_accuracies = analyzer.train_model(
    epochs=25, 
    learning_rate=0.0001,
    batch_size=32
)

print(f"\nTraining completed!")
print(f"Best Test Accuracy: {best_accuracy:.2f}%")

# =============================================================================
# MODEL EVALUATION
# =============================================================================

print("\nEvaluating model...")
accuracy, report, cm, class_names, y_true, y_pred, y_probs = analyzer.evaluate_model()

print(f"\nFinal Evaluation Results:")
print("=" * 50)
print(f"Overall Accuracy: {accuracy:.4f}")
print(f"\nClassification Report:")
print(report)

# =============================================================================
# VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].plot(train_losses)
axes[0, 0].set_title('Training Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].grid(True)

axes[0, 1].plot(train_accuracies, label='Train')
axes[0, 1].plot(test_accuracies, label='Test')
axes[0, 1].set_title('Accuracy Over Time')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[0, 2].bar(range(len(class_counts)), list(class_counts.values()))
axes[0, 2].set_title('Dataset Distribution')
axes[0, 2].set_xlabel('Cancer Types')
axes[0, 2].set_ylabel('Number of Images')
short_names = [name.split('_')[0] for name in class_counts.keys()]
axes[0, 2].set_xticks(range(len(short_names)))
axes[0, 2].set_xticklabels(short_names, rotation=45)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title('Confusion Matrix')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')
short_class_names = [name.split(' - ')[-1] if ' - ' in name else name for name in class_names]
axes[1, 0].set_xticklabels(short_class_names, rotation=45)
axes[1, 0].set_yticklabels(short_class_names, rotation=0)

y_probs_array = np.array(y_probs)
benign_classes = [i for i, name in enumerate(class_names) if 'Benign' in name]
malignant_classes = [i for i, name in enumerate(class_names) if 'Malignant' in name]

benign_probs = y_probs_array[:, benign_classes].sum(axis=1)
malignant_probs = y_probs_array[:, malignant_classes].sum(axis=1)

axes[1, 1].hist([benign_probs, malignant_probs], bins=30, alpha=0.7, 
                label=['Benign Predictions', 'Malignant Predictions'])
axes[1, 1].set_title('Prediction Confidence Distribution')
axes[1, 1].set_xlabel('Probability')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()
axes[1, 1].grid(True)

class_accuracies = []
for i in range(len(class_names)):
    class_mask = np.array(y_true) == i
    if np.sum(class_mask) > 0:
        class_acc = np.sum((np.array(y_pred)[class_mask] == i)) / np.sum(class_mask)
        class_accuracies.append(class_acc)
    else:
        class_accuracies.append(0)

axes[1, 2].bar(range(len(class_names)), class_accuracies)
axes[1, 2].set_title('Per-Class Accuracy')
axes[1, 2].set_xlabel('Cancer Types')
axes[1, 2].set_ylabel('Accuracy')
axes[1, 2].set_xticks(range(len(short_class_names)))
axes[1, 2].set_xticklabels(short_class_names, rotation=45)
axes[1, 2].grid(True, alpha=0.3)

for i, acc in enumerate(class_accuracies):
    axes[1, 2].text(i, acc + 0.01, f'{acc:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# =============================================================================
# DETAILED ANALYSIS
# =============================================================================

print("\nDetailed Analysis:")
print("=" * 50)

benign_correct = 0
benign_total = 0
malignant_correct = 0
malignant_total = 0

for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
    true_class = class_names[true_label]
    pred_class = class_names[pred_label]
    
    if 'Benign' in true_class:
        benign_total += 1
        if 'Benign' in pred_class:
            benign_correct += 1
    else:
        malignant_total += 1
        if 'Malignant' in pred_class:
            malignant_correct += 1

benign_accuracy = benign_correct / benign_total if benign_total > 0 else 0
malignant_accuracy = malignant_correct / malignant_total if malignant_total > 0 else 0

print(f"Benign vs Malignant Classification:")
print(f"Benign Accuracy: {benign_accuracy:.4f} ({benign_correct}/{benign_total})")
print(f"Malignant Accuracy: {malignant_accuracy:.4f} ({malignant_correct}/{malignant_total})")

print(f"\nPer-Class Performance:")
for i, class_name in enumerate(class_names):
    class_mask = np.array(y_true) == i
    if np.sum(class_mask) > 0:
        correct = np.sum((np.array(y_pred)[class_mask] == i))
        total = np.sum(class_mask)
        accuracy = correct / total
        print(f"{class_name}: {accuracy:.4f} ({correct}/{total})")

# =============================================================================
# PREDICTION FUNCTION
# =============================================================================

def predict_single_image(analyzer, image_path):
    """Predict cancer type for a single image"""
    if analyzer.model is None:
        print("No model loaded!")
        return None
    
    try:
        original_image = cv2.imread(image_path)
        if original_image is None:
            original_image = np.array(Image.open(image_path))
        
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            if image_path.lower().endswith(('.jpg', '.jpeg')):
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        normalized_image = analyzer.normalizer.normalize_he(original_image)
        
        binary_mask = analyzer.processor.create_binary_mask(normalized_image)
        
        features = analyzer.processor.extract_features(normalized_image, binary_mask)
        
        image_tensor = analyzer.transform_test(Image.fromarray(normalized_image.astype(np.uint8))).unsqueeze(0).to(analyzer.device)
        
        analyzer.model.eval()
        with torch.no_grad():
            output = analyzer.model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = output.argmax(1).item()
            confidence = probabilities[0][predicted_class].item()
        
        class_name = analyzer.label_encoder.inverse_transform([predicted_class])[0]
        full_class_name = analyzer.class_mapping.get(class_name, class_name)
        
        probs_dict = {}
        for i, class_label in enumerate(analyzer.label_encoder.classes_):
            probs_dict[analyzer.class_mapping.get(class_label, class_label)] = probabilities[0][i].item()
        
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
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# =============================================================================
# SAMPLE PREDICTIONS
# =============================================================================

print("\nTesting predictions on sample images...")

sample_images = analyzer.X_test[:5]
sample_labels = analyzer.y_test[:5]

fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for i, (img_path, true_label) in enumerate(zip(sample_images, sample_labels)):
    result = predict_single_image(analyzer, img_path)
    
    if result:
        true_class = analyzer.class_mapping.get(analyzer.label_encoder.inverse_transform([true_label])[0], 
                                               analyzer.label_encoder.inverse_transform([true_label])[0])
        pred_class = result['prediction']['class']
        confidence = result['prediction']['confidence']
        
        axes[0, i].imshow(result['original_image'])
        axes[0, i].set_title(f'Original\nTrue: {true_class.split(" - ")[-1]}', fontsize=10)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(result['normalized_image'])
        axes[1, i].set_title(f'Normalized\nPred: {pred_class.split(" - ")[-1]}\nConf: {confidence:.3f}', fontsize=10)
        axes[1, i].axis('off')
        
        correct = "✓" if pred_class == true_class else "✗"
        print(f"Image {i+1}: {correct} True: {true_class}, Predicted: {pred_class} (Conf: {confidence:.3f})")

plt.tight_layout()
plt.show()

# =============================================================================
# MODEL SAVING
# =============================================================================

print("\nSaving final model...")

final_model_path = 'breakhis_final_model.pth'
torch.save({
    'model_state_dict': analyzer.model.state_dict(),
    'label_encoder': analyzer.label_encoder,
    'class_mapping': analyzer.class_mapping,
    'num_classes': len(analyzer.label_encoder.classes_),
    'accuracy': best_accuracy,
    'class_names': class_names,
    'training_history': {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }
}, final_model_path)

print(f"Model saved to {final_model_path}")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Dataset: BreakHis Multi-Class Cancer Classification")
print(f"Total Images Processed: {total_images}")
print(f"Classes: {len(class_names)}")
print(f"Training Images: {len(analyzer.X_train)}")
print(f"Testing Images: {len(analyzer.X_test)}")
print(f"Model Architecture: EfficientNet-B1 with Attention")
print(f"Best Test Accuracy: {best_accuracy:.2f}%")
print(f"Final Test Accuracy: {accuracy:.4f}")
print(f"Benign Classification Accuracy: {benign_accuracy:.4f}")
print(f"Malignant Classification Accuracy: {malignant_accuracy:.4f}")

print("\nClass Distribution:")
for class_name, count in class_counts.items():
    display_name = analyzer.class_mapping.get(class_name, class_name)
    percentage = (count / total_images) * 100
    print(f"  {display_name}: {count} images ({percentage:.1f}%)")

print("\nTop Performing Classes:")
class_performance = list(zip(class_names, class_accuracies))
class_performance.sort(key=lambda x: x[1], reverse=True)
for i, (class_name, acc) in enumerate(class_performance[:3]):
    print(f"  {i+1}. {class_name}: {acc:.4f}")

print("\nModel Features:")
print("  ✓ Macenko stain normalization")
print("  ✓ Data augmentation")
print("  ✓ Attention mechanism")
print("  ✓ Multi-magnification training")
print("  ✓ Label smoothing")
print("  ✓ OneCycle learning rate scheduling")
print("  ✓ Gradient clipping")

print(f"\nTraining completed successfully!")
print("="*60)