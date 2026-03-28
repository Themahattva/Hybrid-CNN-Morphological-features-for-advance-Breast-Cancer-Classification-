import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import os
import glob
from skimage import filters, morphology, segmentation, measure, feature
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy import ndimage
from scipy.stats import skew, kurtosis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import seaborn as sns
from tqdm import tqdm
import time
import torch.backends.cudnn as cudnn
import timm
from torchvision.transforms import functional as TF
import random
import math
from collections import OrderedDict
import logging
warnings.filterwarnings('ignore')

def setup_gpu_optimizations():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        device = torch.cuda.get_device_properties(0)
        print(f"GPU Memory: {device.total_memory / 1e9:.1f} GB")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Compute Capability: {device.major}.{device.minor}")
        return True
    return False

print("Environment Setup Complete")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

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

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.num_classes = num_classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

class ElasticTransform:
    def __init__(self, alpha=34, sigma=4, alpha_affine=0.1):
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine

    def __call__(self, img):
        if random.random() < 0.5:
            return img
        
        img_array = np.array(img)
        shape = img_array.shape
        shape_size = shape[:2]

        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        
        pts1 = np.float32([center_square + square_size,
                          [center_square[0] + square_size, center_square[1] - square_size],
                          center_square - square_size])
        pts2 = pts1 + np.random.uniform(-self.alpha_affine, self.alpha_affine, size=pts1.shape).astype(np.float32)
        
        M = cv2.getAffineTransform(pts1, pts2)
        img_array = cv2.warpAffine(img_array, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        dx = cv2.GaussianBlur((np.random.rand(*shape_size) * 2 - 1), (0, 0), self.sigma) * self.alpha
        dy = cv2.GaussianBlur((np.random.rand(*shape_size) * 2 - 1), (0, 0), self.sigma) * self.alpha

        x, y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        if len(shape) == 3:
            for i in range(shape[2]):
                img_array[:, :, i] = ndimage.map_coordinates(img_array[:, :, i], indices, order=1).reshape(shape_size)
        else:
            img_array = ndimage.map_coordinates(img_array, indices, order=1).reshape(shape_size)

        return Image.fromarray(img_array.astype(np.uint8))

class ComprehensiveFeatureExtractor:
    def __init__(self):
        self.feature_names = []
        self._setup_feature_names()
    
    def _setup_feature_names(self):
        self.feature_names = [
            'area', 'perimeter', 'eccentricity', 'solidity', 'extent', 'mean_intensity', 'compactness',
            'lbp_mean', 'lbp_std', 'lbp_skew', 'lbp_kurt',
            'glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation',
            'haralick_1', 'haralick_2', 'haralick_3', 'haralick_4', 'haralick_5', 'haralick_6', 'haralick_7',
            'gabor_1', 'gabor_2', 'gabor_3', 'gabor_4',
            'rgb_mean_r', 'rgb_mean_g', 'rgb_mean_b', 'rgb_std_r', 'rgb_std_g', 'rgb_std_b',
            'hsv_mean_h', 'hsv_mean_s', 'hsv_mean_v', 'hsv_std_h', 'hsv_std_s', 'hsv_std_v',
            'lab_mean_l', 'lab_mean_a', 'lab_mean_b', 'lab_std_l', 'lab_std_a', 'lab_std_b',
            'rgb_skew_r', 'rgb_skew_g', 'rgb_skew_b', 'rgb_kurt_r', 'rgb_kurt_g', 'rgb_kurt_b',
            'nuclear_size_mean', 'nuclear_size_std', 'nuclear_density', 'nuclear_circularity',
            'fractal_dimension', 'convex_hull_ratio', 'fourier_1', 'fourier_2', 'fourier_3',
            'zernike_1', 'zernike_2', 'zernike_3'
        ]
    
    def extract_texture_features(self, image, mask):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        lbp_features = [
            np.mean(lbp), np.std(lbp), skew(lbp.flatten()), kurtosis(lbp.flatten())
        ]
        
        glcm = graycomatrix(gray, distances=[1], angles=[0, 45, 90, 135], symmetric=True, normed=True)
        glcm_features = [
            np.mean(graycoprops(glcm, 'contrast')),
            np.mean(graycoprops(glcm, 'dissimilarity')),
            np.mean(graycoprops(glcm, 'homogeneity')),
            np.mean(graycoprops(glcm, 'energy')),
            np.mean(graycoprops(glcm, 'correlation'))
        ]
        
        haralick_features = []
        try:
            from mahotas.features import haralick
            haralick_feat = haralick(gray)
            haralick_features = np.mean(haralick_feat, axis=0)[:7].tolist()
        except:
            haralick_features = [0] * 7
        
        gabor_features = []
        for angle in [0, 45, 90, 135]:
            filtered = filters.gabor(gray, frequency=0.1, theta=np.radians(angle))
            gabor_features.append(np.std(filtered[0]))
        
        return lbp_features + glcm_features + haralick_features + gabor_features
    
    def extract_color_features(self, image, mask):
        rgb_features = []
        for i in range(3):
            channel = image[:, :, i]
            rgb_features.extend([
                np.mean(channel), np.std(channel)
            ])
        
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv_features = []
        for i in range(3):
            channel = hsv[:, :, i]
            hsv_features.extend([
                np.mean(channel), np.std(channel)
            ])
        
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab_features = []
        for i in range(3):
            channel = lab[:, :, i]
            lab_features.extend([
                np.mean(channel), np.std(channel)
            ])
        
        rgb_higher_moments = []
        for i in range(3):
            channel = image[:, :, i]
            rgb_higher_moments.extend([
                skew(channel.flatten()), kurtosis(channel.flatten())
            ])
        
        return rgb_features + hsv_features + lab_features + rgb_higher_moments
    
    def extract_nuclear_features(self, image, mask):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        labeled, num_objects = ndimage.label(mask)
        
        if num_objects == 0:
            return [0, 0, 0, 0]
        
        sizes = ndimage.sum_labels(mask, labeled, range(1, num_objects + 1))
        nuclear_size_mean = np.mean(sizes) if len(sizes) > 0 else 0
        nuclear_size_std = np.std(sizes) if len(sizes) > 0 else 0
        
        nuclear_density = num_objects / (image.shape[0] * image.shape[1])
        
        props = measure.regionprops(labeled)
        circularities = []
        for prop in props:
            if prop.perimeter > 0:
                circularity = 4 * np.pi * prop.area / (prop.perimeter ** 2)
                circularities.append(circularity)
        
        nuclear_circularity = np.mean(circularities) if circularities else 0
        
        return [nuclear_size_mean, nuclear_size_std, nuclear_density, nuclear_circularity]
    
    def extract_advanced_geometric_features(self, image, mask):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        fractal_dim = self.calculate_fractal_dimension(gray)
        
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        convex_hull_ratio = 0
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_contour)
            contour_area = cv2.contourArea(largest_contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                convex_hull_ratio = contour_area / hull_area
        
        fourier_features = self.calculate_fourier_descriptors(mask)
        zernike_features = self.calculate_zernike_moments(gray)
        
        return [fractal_dim, convex_hull_ratio] + fourier_features + zernike_features
    
    def calculate_fractal_dimension(self, image):
        try:
            sizes = np.arange(2, 100, 2)
            counts = []
            for size in sizes:
                box_count = np.sum(image > 127)
                counts.append(box_count)
            
            counts = np.array(counts)
            sizes = sizes[:len(counts)]
            
            if len(counts) > 3:
                coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
                return -coeffs[0]
            else:
                return 1.5
        except:
            return 1.5
    
    def calculate_fourier_descriptors(self, mask, num_descriptors=3):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return [0] * num_descriptors
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        if len(largest_contour) < 4:
            return [0] * num_descriptors
        
        try:
            contour_complex = largest_contour[:, 0, 0] + 1j * largest_contour[:, 0, 1]
            fourier_result = np.fft.fft(contour_complex)
            fourier_descriptors = np.abs(fourier_result)
            
            normalized_descriptors = fourier_descriptors[1:num_descriptors+1] / fourier_descriptors[1] if fourier_descriptors[1] > 0 else fourier_descriptors[1:num_descriptors+1]
            
            return normalized_descriptors.real.tolist()
        except:
            return [0] * num_descriptors
    
    def calculate_zernike_moments(self, image, num_moments=3):
        try:
            from mahotas.features import zernike_moments
            moments = zernike_moments(image, radius=min(image.shape)//2)
            return moments[:num_moments].tolist()
        except:
            return [0] * num_moments
    
    def extract_basic_features(self, image, mask):
        props = measure.regionprops(mask.astype(int), intensity_image=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        
        if len(props) == 0:
            return np.zeros(7)
        
        largest_region = max(props, key=lambda x: x.area)
        
        features = [
            largest_region.area,
            largest_region.perimeter,
            largest_region.eccentricity,
            largest_region.solidity,
            largest_region.extent,
            largest_region.mean_intensity,
            (largest_region.perimeter ** 2) / (4 * np.pi * largest_region.area) if largest_region.area > 0 else 0
        ]
        
        return np.array(features, dtype=np.float32)
    
    def extract_all_features(self, image, mask):
        try:
            basic_features = self.extract_basic_features(image, mask)
            texture_features = self.extract_texture_features(image, mask)
            color_features = self.extract_color_features(image, mask)
            nuclear_features = self.extract_nuclear_features(image, mask)
            geometric_features = self.extract_advanced_geometric_features(image, mask)
            
            all_features = np.concatenate([
                basic_features,
                texture_features,
                color_features,
                nuclear_features,
                geometric_features
            ])
            
            all_features = np.nan_to_num(all_features, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return all_features
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.zeros(len(self.feature_names))

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

class MixUp:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).cuda()
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam

class BreakHisDataset(Dataset):
    def __init__(self, image_paths, labels, morphological_features=None, transform=None, normalizer=None, is_training=False):
        self.image_paths = image_paths
        self.labels = labels
        self.morphological_features = morphological_features
        self.transform = transform
        self.normalizer = normalizer
        self.is_training = is_training
        
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
        
        if self.morphological_features is not None:
            morph_features = torch.tensor(self.morphological_features[idx], dtype=torch.float32)
            return image, morph_features, label
        
        return image, label

class MultiScaleFeaturePyramid(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, 256, 1) for _ in range(3)
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(256, 256, 3, padding=1) for _ in range(3)
        ])
        
    def forward(self, features):
        laterals = [lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, features)]
        
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] += F.interpolate(laterals[i + 1], scale_factor=2, mode='nearest')
        
        fpn_outs = [fpn_conv(lateral) for fpn_conv, lateral in zip(self.fpn_convs, laterals)]
        
        return fpn_outs

class EfficientNetBackbone(nn.Module):
    def __init__(self, model_name='efficientnet_b3'):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True, out_indices=[2, 3, 4])
        
        feature_dims = self.backbone.feature_info.channels()
        self.fpn = MultiScaleFeaturePyramid(feature_dims[-1])
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        features = self.backbone(x)
        
        fpn_features = self.fpn(features)
        
        pooled_features = []
        for feat in fpn_features:
            pooled = self.global_pool(feat)
            pooled_features.append(pooled.flatten(1))
        
        combined = torch.cat(pooled_features, dim=1)
        
        return combined

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

class AdvancedBreakHisClassifier(nn.Module):
    def __init__(self, num_classes=8, num_morphological_features=52, fusion_strategy='attention_weighted'):
        super().__init__()
        
        self.num_morphological_features = num_morphological_features
        self.fusion_strategy = fusion_strategy
        
        self.backbone = EfficientNetBackbone('efficientnet_b3')
        self.cnn_feature_dim = 768
        
        self.cnn_feature_extractor = nn.Sequential(
            nn.Linear(self.cnn_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.morph_feature_processor = nn.Sequential(
            nn.Linear(num_morphological_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        if fusion_strategy == 'attention_weighted':
            self.feature_attention = nn.Sequential(
                nn.Linear(320, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 320),
                nn.Sigmoid()
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(320, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes)
            )
    
    def forward(self, x, morphological_features):
        cnn_features = self.backbone(x)
        cnn_features = self.cnn_feature_extractor(cnn_features)
        
        morph_features = self.morph_feature_processor(morphological_features)
        
        combined_features = torch.cat([cnn_features, morph_features], dim=1)
        attention_weights = self.feature_attention(combined_features)
        weighted_features = combined_features * attention_weights
        output = self.classifier(weighted_features)
        
        return output

class ImprovedTrainingPipeline:
    def __init__(self, model, device, use_mixed_precision=True):
        self.model = model
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        
    def train_with_cross_validation(self, X_train, y_train, morph_features, n_splits=5, epochs=30):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            print(f"\nFold {fold + 1}/{n_splits}")
            print("-" * 40)
            
            X_fold_train = [X_train[i] for i in train_idx]
            X_fold_val = [X_train[i] for i in val_idx]
            y_fold_train = [y_train[i] for i in train_idx]
            y_fold_val = [y_train[i] for i in val_idx]
            
            morph_fold_train = morph_features[train_idx] if morph_features is not None else None
            morph_fold_val = morph_features[val_idx] if morph_features is not None else None
            
            fold_score = self._train_single_fold(
                X_fold_train, X_fold_val, y_fold_train, y_fold_val,
                morph_fold_train, morph_fold_val, epochs
            )
            
            cv_scores.append(fold_score)
            print(f"Fold {fold + 1} Score: {fold_score:.4f}")
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        print(f"\nCross-Validation Results:")
        print(f"Mean Score: {mean_score:.4f} ± {std_score:.4f}")
        print(f"95% Confidence Interval: [{mean_score - 1.96*std_score:.4f}, {mean_score + 1.96*std_score:.4f}]")
        
        return cv_scores, mean_score, std_score

    def _train_single_fold(self, X_train, X_val, y_train, y_val, morph_train, morph_val, epochs):
        num_classes = len(set(y_train))
        
        focal_loss = FocalLoss(alpha=1, gamma=2)
        label_smoothing = LabelSmoothingLoss(num_classes, smoothing=0.1)
        
        def combined_loss(outputs, targets):
            fl = focal_loss(outputs, targets)
            ls = label_smoothing(outputs, targets)
            return 0.7 * fl + 0.3 * ls
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=3e-4,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=3e-3,
            epochs=epochs,
            steps_per_epoch=len(X_train) // 64 + 1,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            ElasticTransform(alpha=34, sigma=4),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = BreakHisDataset(
            X_train, y_train, morph_train, transform=transform, 
            normalizer=MacenkoStainNormalizer(), is_training=True
        )
        val_dataset = BreakHisDataset(
            X_val, y_val, morph_val, transform=val_transform, 
            normalizer=MacenkoStainNormalizer(), is_training=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        
        best_val_acc = 0.0
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch_idx, batch_data in enumerate(train_pbar):
                if len(batch_data) == 3:
                    images, morph_features, labels = batch_data
                    morph_features = morph_features.to(self.device, non_blocking=True)
                else:
                    images, labels = batch_data
                    morph_features = torch.zeros((images.size(0), 52), device=self.device)
                
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images, morph_features)
                        loss = combined_loss(outputs, labels)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images, morph_features)
                    loss = combined_loss(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                if batch_idx % 20 == 0:
                    current_acc = 100. * train_correct / train_total
                    train_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{current_acc:.2f}%',
                        'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
                    })
            
            val_acc = self._validate_fold(val_loader)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return best_val_acc
    
    def _validate_fold(self, val_loader):
        self.model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 3:
                    images, morph_features, labels = batch_data
                    morph_features = morph_features.to(self.device, non_blocking=True)
                else:
                    images, labels = batch_data
                    morph_features = torch.zeros((images.size(0), 52), device=self.device)
                
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images, morph_features)
                else:
                    outputs = self.model(images, morph_features)
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        return 100. * val_correct / val_total

class BreakHisAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.normalizer = MacenkoStainNormalizer()
        self.processor = ImageProcessor()
        self.feature_extractor = ComprehensiveFeatureExtractor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_gpu = setup_gpu_optimizations()
        self.use_mixed_precision = self.use_gpu
        
        print(f"Using device: {self.device}")
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.morph_scaler = StandardScaler()
        self.image_paths = []
        self.labels = []
        self.magnifications = []
        self.morphological_features = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.morph_train = None
        self.morph_test = None
        
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
            ElasticTransform(alpha=34, sigma=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_all_morphological_features(self):
        self.morphological_features = []
        total_images = len(self.image_paths)
        
        print(f"Extracting comprehensive morphological features for {total_images} images...")
        
        for i, image_path in enumerate(tqdm(self.image_paths, desc="Processing images")):
            try:
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                normalized_image = self.normalizer.normalize_he(image)
                binary_mask = self.processor.create_binary_mask(normalized_image)
                features = self.feature_extractor.extract_all_features(normalized_image, binary_mask)
                self.morphological_features.append(features)
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                self.morphological_features.append(np.zeros(len(self.feature_extractor.feature_names)))
        
        self.morphological_features = np.array(self.morphological_features)
        self.morphological_features = self.morph_scaler.fit_transform(self.morphological_features)
        
        print(f"Extracted and normalized comprehensive features: {self.morphological_features.shape}")
        print(f"Feature count: {len(self.feature_extractor.feature_names)}")
    
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
        
    def load_dataset(self, selected_magnification='400X', extract_morphological=True):
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
            raise ValueError(f"No images found for magnification {selected_magnification}")
        
        self.labels = self.label_encoder.fit_transform(self.labels)
        
        if extract_morphological:
            self.extract_all_morphological_features()
        
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
        if len(self.morphological_features) > 0:
            self.X_train, self.X_test, self.y_train, self.y_test, self.morph_train, self.morph_test = train_test_split(
                self.image_paths, self.labels, self.morphological_features,
                test_size=test_size, random_state=42, stratify=self.labels
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.image_paths, self.labels, 
                test_size=test_size, random_state=42, stratify=self.labels
            )
            self.morph_train = None
            self.morph_test = None
    
    def train_model_with_cv(self, epochs=30, learning_rate=3e-4):
        num_classes = len(self.label_encoder.classes_)
        num_features = len(self.feature_extractor.feature_names)
        
        print(f"Training model with {num_classes} classes and {num_features} morphological features")
        
        self.model = AdvancedBreakHisClassifier(
            num_classes=num_classes,
            num_morphological_features=num_features,
            fusion_strategy='attention_weighted'
        ).to(self.device)
        
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        
        trainer = ImprovedTrainingPipeline(self.model, self.device, self.use_mixed_precision)
        
        cv_scores, mean_score, std_score = trainer.train_with_cross_validation(
            self.X_train, self.y_train, self.morph_train, n_splits=5, epochs=epochs
        )
        
        print(f"\nFinal Cross-Validation Results:")
        print(f"Mean Accuracy: {mean_score:.4f} ± {std_score:.4f}")
        print(f"Individual fold scores: {[f'{score:.4f}' for score in cv_scores]}")
        
        return cv_scores, mean_score, std_score
    
    def train_final_model(self, epochs=30):
        num_classes = len(self.label_encoder.classes_)
        num_features = len(self.feature_extractor.feature_names)
        
        self.model = AdvancedBreakHisClassifier(
            num_classes=num_classes,
            num_morphological_features=num_features,
            fusion_strategy='attention_weighted'
        ).to(self.device)
        
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        
        focal_loss = FocalLoss(alpha=1, gamma=2)
        label_smoothing = LabelSmoothingLoss(num_classes, smoothing=0.1)
        
        def combined_loss(outputs, targets):
            fl = focal_loss(outputs, targets)
            ls = label_smoothing(outputs, targets)
            return 0.7 * fl + 0.3 * ls
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=3e-4,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        steps_per_epoch = len(self.X_train) // 64 + 1
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=3e-3,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        train_dataset = BreakHisDataset(
            self.X_train, self.y_train, self.morph_train,
            transform=self.transform, normalizer=self.normalizer, is_training=True
        )
        
        test_dataset = BreakHisDataset(
            self.X_test, self.y_test, self.morph_test,
            transform=self.test_transform, normalizer=self.normalizer, is_training=False
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=True, 
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=64, shuffle=False, 
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        
        best_accuracy = 0.0
        train_losses = []
        train_accuracies = []
        val_accuracies = []
        patience_counter = 0
        patience = 10
        
        scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
        
        print(f"Starting final training for {epochs} epochs...")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            
            for batch_idx, batch_data in enumerate(train_pbar):
                if len(batch_data) == 3:
                    images, morph_features, labels = batch_data
                    morph_features = morph_features.to(self.device, non_blocking=True)
                else:
                    images, labels = batch_data
                    morph_features = torch.zeros((images.size(0), num_features), device=self.device)
                
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                if self.use_mixed_precision and scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images, morph_features)
                        loss = combined_loss(outputs, labels)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(images, morph_features)
                    loss = combined_loss(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                if batch_idx % 20 == 0:
                    current_acc = 100. * train_correct / train_total
                    train_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{current_acc:.2f}%',
                        'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
                    })
            
            train_accuracy = 100. * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            self.model.eval()
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
                for batch_data in val_pbar:
                    if len(batch_data) == 3:
                        images, morph_features, labels = batch_data
                        morph_features = morph_features.to(self.device, non_blocking=True)
                    else:
                        images, labels = batch_data
                        morph_features = torch.zeros((images.size(0), num_features), device=self.device)
                    
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    
                    if self.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(images, morph_features)
                    else:
                        outputs = self.model(images, morph_features)
                    
                    _, predicted = outputs.max(1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()
            
            test_accuracy = 100. * test_correct / test_total
            epoch_time = time.time() - start_time
            
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(test_accuracy)
            
            print(f'Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s):')
            print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            print(f'  Val Acc: {test_accuracy:.2f}%')
            print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
            
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                patience_counter = 0
                
                checkpoint = {
                    'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'label_encoder': self.label_encoder,
                    'morph_scaler': self.morph_scaler,
                    'feature_extractor': self.feature_extractor,
                    'class_mapping': self.class_mapping,
                    'num_classes': num_classes,
                    'num_features': num_features,
                    'epoch': epoch + 1,
                    'best_accuracy': best_accuracy
                }
                
                if scaler:
                    checkpoint['scaler_state_dict'] = scaler.state_dict()
                
                torch.save(checkpoint, 'best_enhanced_breakhis_model.pth')
                print(f'  *** New best accuracy: {best_accuracy:.2f}% - Model saved!')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            print('-' * 60)
            
            if self.use_gpu:
                torch.cuda.empty_cache()
        
        print(f'\nTraining completed!')
        print(f'Best validation accuracy: {best_accuracy:.2f}%')
        
        return best_accuracy, train_losses, train_accuracies, val_accuracies
        
    def evaluate_model(self):
        if self.model is None:
            return None
        
        test_dataset = BreakHisDataset(
            self.X_test, self.y_test, self.morph_test,
            transform=self.test_transform, normalizer=self.normalizer
        )
        
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
        
        self.model.eval()
        y_true = []
        y_pred = []
        y_probs = []
        
        print("Evaluating model...")
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc="Evaluating"):
                if len(batch_data) == 3:
                    images, morph_features, labels = batch_data
                    morph_features = morph_features.to(self.device)
                else:
                    images, labels = batch_data
                    morph_features = torch.zeros((images.size(0), len(self.feature_extractor.feature_names)), device=self.device)
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images, morph_features)
                else:
                    outputs = self.model(images, morph_features)
                
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_probs.extend(probabilities.cpu().numpy())
        
        accuracy = accuracy_score(y_true, y_pred)
        
        try:
            auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
        except:
            auc = 0.0
        
        report = classification_report(y_true, y_pred, 
                                     target_names=self.label_encoder.classes_, 
                                     output_dict=True)
        
        cm = confusion_matrix(y_true, y_pred)
        
        results = {
            'accuracy': accuracy,
            'auc': auc,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_probs': y_probs
        }
        
        print(f"\nTest Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=self.label_encoder.classes_))
        
        return results
    
    def predict_single_image(self, image_path):
        if self.model is None:
            print("Model not trained yet!")
            return None
        
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            normalized_image = self.normalizer.normalize_he(image)
            binary_mask = self.processor.create_binary_mask(normalized_image)
            morph_features = self.feature_extractor.extract_all_features(normalized_image, binary_mask)
            
            morph_features = self.morph_scaler.transform([morph_features])
            morph_features = torch.tensor(morph_features, dtype=torch.float32).to(self.device)
            
            transformed_image = self.test_transform(normalized_image).unsqueeze(0).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(transformed_image, morph_features)
                else:
                    outputs = self.model(transformed_image, morph_features)
                
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = self.label_encoder.inverse_transform([predicted.cpu().item()])[0]
            confidence_score = confidence.cpu().item()
            
            class_probabilities = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                class_probabilities[class_name] = probabilities[0][i].cpu().item()
            
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence_score,
                'class_probabilities': class_probabilities,
                'readable_name': self.class_mapping.get(predicted_class, predicted_class)
            }
            
            return result
            
        except Exception as e:
            print(f"Error predicting image {image_path}: {e}")
            return None
    
    def load_model(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            num_classes = checkpoint['num_classes']
            num_features = checkpoint['num_features']
            
            self.model = AdvancedBreakHisClassifier(
                num_classes=num_classes,
                num_morphological_features=num_features,
                fusion_strategy='attention_weighted'
            ).to(self.device)
            
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
            
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.label_encoder = checkpoint['label_encoder']
            self.morph_scaler = checkpoint['morph_scaler']
            self.feature_extractor = checkpoint['feature_extractor']
            self.class_mapping = checkpoint['class_mapping']
            
            print(f"Model loaded successfully from {model_path}")
            print(f"Best accuracy: {checkpoint['best_accuracy']:.2f}%")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def save_model(self, filepath):
        if self.model is None:
            print("No model to save!")
            return False
        
        try:
            checkpoint = {
                'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
                'label_encoder': self.label_encoder,
                'morph_scaler': self.morph_scaler,
                'feature_extractor': self.feature_extractor,
                'class_mapping': self.class_mapping,
                'num_classes': len(self.label_encoder.classes_),
                'num_features': len(self.feature_extractor.feature_names)
            }
            
            torch.save(checkpoint, filepath)
            print(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def get_model_summary(self):
        if self.model is None:
            return "No model loaded"
        
        summary = {
            'num_classes': len(self.label_encoder.classes_),
            'class_names': list(self.label_encoder.classes_),
            'num_features': len(self.feature_extractor.feature_names),
            'feature_names': self.feature_extractor.feature_names,
            'total_images': len(self.image_paths),
            'train_images': len(self.X_train) if self.X_train else 0,
            'test_images': len(self.X_test) if self.X_test else 0,
            'device': str(self.device),
            'mixed_precision': self.use_mixed_precision
        }
        
        return summary

def run_complete_analysis(dataset_path, magnification='400X', epochs=30):
    """
    Complete analysis pipeline for BreakHis dataset
    """
    print("="*60)
    print("BREAKHIS CANCER CLASSIFICATION SYSTEM")
    print("="*60)
    
    analyzer = BreakHisAnalyzer(dataset_path)
    
    print("\n1. Loading dataset...")
    try:
        class_counts = analyzer.load_dataset(magnification, extract_morphological=True)
        print(f"Class distribution: {class_counts}")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None
    
    print("\n2. Splitting dataset...")
    analyzer.split_dataset(test_size=0.2)
    
    print("\n3. Training with cross-validation...")
    cv_scores, mean_cv_score, std_cv_score = analyzer.train_model_with_cv(epochs=epochs)
    
    print("\n4. Training final model...")
    best_accuracy, train_losses, train_accuracies, val_accuracies = analyzer.train_final_model(epochs=epochs)
    
    print("\n5. Evaluating model...")
    results = analyzer.evaluate_model()
    
    print("\n6. Model summary:")
    summary = analyzer.get_model_summary()
    for key, value in summary.items():
        if key not in ['feature_names', 'class_names']:
            print(f"{key}: {value}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    return analyzer, results

if __name__ == "__main__":
    dataset_path = "/path/to/breakhis/dataset"
    
    analyzer, results = run_complete_analysis(
        dataset_path=dataset_path,
        magnification='400X',
        epochs=30
    )
    
    if analyzer and results:
        print(f"\nFinal Test Accuracy: {results['accuracy']:.4f}")
        print(f"Final Test AUC: {results['auc']:.4f}")
        
        example_image = "/path/to/test/image.png"
        if os.path.exists(example_image):
            prediction = analyzer.predict_single_image(example_image)
            if prediction:
                print(f"\nExample Prediction:")
                print(f"Class: {prediction['readable_name']}")
                print(f"Confidence: {prediction['confidence']:.4f}")
    
    print("\nTraining completed successfully!")
    