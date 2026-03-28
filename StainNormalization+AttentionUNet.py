import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
from skimage import filters, morphology, segmentation
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import warnings
warnings.filterwarnings('ignore')

class MacenkoStainNormalizer:
    def __init__(self):
        # Reference stain matrix (typical H&E stain vectors)
        # These are typical values for hematoxylin and eosin
        self.target_stains = np.array([[0.5626, 0.2159],
                                       [0.7201, 0.8012], 
                                       [0.4062, 0.5581]])
        
        # Target concentrations (can be adjusted)
        self.target_concentrations = np.array([[1.9705, 1.0308]])
        
    def rgb_to_od(self, img):
        """Convert RGB image to optical density"""
        # Add small epsilon to avoid log(0)
        img = np.maximum(img, 1e-6)
        return -np.log(img / 255.0)
    
    def od_to_rgb(self, od):
        """Convert optical density back to RGB"""
        rgb = 255 * np.exp(-od)
        return np.clip(rgb, 0, 255).astype(np.uint8)
    
    def get_stain_matrix(self, od, beta=0.15, alpha=1):
        """Extract stain matrix using robust PCA approach"""
        # Reshape OD for processing
        od_flat = od.reshape(-1, 3)
        
        # Remove transparent pixels (low optical density)
        od_hat = od_flat[(od_flat > beta).any(axis=1)]
        
        if len(od_hat) == 0:
            return self.target_stains
            
        # Compute eigenvectors
        cov = np.cov(od_hat.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvals)[::-1]
        eigenvecs = eigenvecs[:, idx]
        
        # Project data onto plane
        proj = np.dot(od_hat, eigenvecs[:, :2])
        
        # Find robust extreme angles
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        
        # Find percentiles for robust estimation
        min_phi = np.percentile(phi, alpha)
        max_phi = np.percentile(phi, 100 - alpha)
        
        # Convert back to stain vectors
        v1 = np.dot(eigenvecs[:, :2], [np.cos(min_phi), np.sin(min_phi)])
        v2 = np.dot(eigenvecs[:, :2], [np.cos(max_phi), np.sin(max_phi)])
        
        # Normalize stain vectors
        if v1[0] > 0: v1 = -v1
        if v2[0] > 0: v2 = -v2
        
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        return np.column_stack([v1, v2])
    
    def separate_stains(self, od, stain_matrix):
        """Separate stains using non-negative matrix factorization"""
        od_flat = od.reshape(-1, 3)
        
        # Use least squares to get concentrations
        concentrations = np.linalg.lstsq(stain_matrix, od_flat.T, rcond=None)[0]
        concentrations = np.maximum(concentrations, 0)
        
        return concentrations.T.reshape(od.shape[:2] + (2,))
    
    def normalize_he(self, image):
        """Main function to normalize H&E staining"""
        # Convert to optical density
        od = self.rgb_to_od(image)
        
        # Get stain matrix from image
        source_stains = self.get_stain_matrix(od)
        
        # Separate stains
        concentrations = self.separate_stains(od, source_stains)
        
        # Reconstruct with target stain matrix
        normalized_od = np.dot(concentrations, self.target_stains.T)
        
        # Convert back to RGB
        normalized_rgb = self.od_to_rgb(normalized_od)
        
        return normalized_rgb
    
    def create_tissue_mask(self, image, method='otsu'):
        """Create binary mask for tissue segmentation"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        if method == 'otsu':
            # Otsu thresholding (good for tissue vs background)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            # Adaptive thresholding
            mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)
        else:  # 'manual'
            # Simple threshold (can be adjusted)
            _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return mask
    
    def create_nuclei_mask(self, image):
        """Create binary mask for nuclei detection using hematoxylin channel"""
        # Convert to optical density
        od = self.rgb_to_od(image)
        
        # Get stain matrix
        stain_matrix = self.get_stain_matrix(od)
        
        # Separate stains
        concentrations = self.separate_stains(od, stain_matrix)
        
        # Extract hematoxylin channel (usually first channel)
        hematoxylin = concentrations[:, :, 0]
        
        # Normalize to 0-255
        h_norm = ((hematoxylin - hematoxylin.min()) / 
                 (hematoxylin.max() - hematoxylin.min()) * 255).astype(np.uint8)
        
        # Apply threshold to get nuclei
        _, nuclei_mask = cv2.threshold(h_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = np.ones((2,2), np.uint8)
        nuclei_mask = cv2.morphologyEx(nuclei_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        nuclei_mask = cv2.morphologyEx(nuclei_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return nuclei_mask
    
    def create_eosin_mask(self, image):
        """Create binary mask for eosin-rich regions (cytoplasm, RBCs, etc.)"""
        # Convert to optical density
        od = self.rgb_to_od(image)
        
        # Get stain matrix
        stain_matrix = self.get_stain_matrix(od)
        
        # Separate stains
        concentrations = self.separate_stains(od, stain_matrix)
        
        # Extract eosin channel (usually second channel)
        eosin = concentrations[:, :, 1]
        
        # Normalize to 0-255
        e_norm = ((eosin - eosin.min()) / 
                 (eosin.max() - eosin.min()) * 255).astype(np.uint8)
        
        # Apply threshold
        _, eosin_mask = cv2.threshold(e_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean up
        kernel = np.ones((3,3), np.uint8)
        eosin_mask = cv2.morphologyEx(eosin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return eosin_mask

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
    """Attention U-Net with EfficientNet backbone"""
    def __init__(self, num_classes=1, efficientnet_version='b0'):
        super(AttentionUNet, self).__init__()
        
        # EfficientNet encoder
        self.backbone = EfficientNet.from_pretrained(f'efficientnet-{efficientnet_version}')
        
        # Get feature dimensions for different EfficientNet versions
        feature_dims = {
            'b0': [32, 24, 40, 112, 1280],
            'b1': [32, 24, 40, 112, 1280],
            'b2': [32, 24, 48, 120, 1408],
            'b3': [40, 32, 48, 136, 1536],
            'b4': [48, 32, 56, 160, 1792]
        }
        
        dims = feature_dims.get(efficientnet_version, feature_dims['b0'])
        
        # Decoder layers
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
        
        # Final convolution
        self.final_conv = nn.Conv2d(dims[0], num_classes, kernel_size=1)
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dims[4], 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256)
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
        """Extract features from EfficientNet encoder"""
        features = []
        
        # Stem
        x = self.backbone._conv_stem(x)
        x = self.backbone._bn0(x)
        x = self.backbone._swish(x)
        features.append(x)
        
        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            x = block(x)
            # Save features at specific stages
            if idx in [2, 4, 10, 16]:  # Adjust based on EfficientNet version
                features.append(x)
        
        # Head
        x = self.backbone._conv_head(x)
        x = self.backbone._bn1(x)
        x = self.backbone._swish(x)
        features.append(x)
        
        return features
    
    def forward(self, x, return_features=False):
        # Extract encoder features
        encoder_features = self.extract_encoder_features(x)
        
        # Extract deep features for analysis
        deep_features = self.feature_extractor(encoder_features[-1])
        
        if return_features:
            return deep_features
        
        # Decoder with attention
        d4 = self.up_conv4(encoder_features[-1])
        s4 = self.att4(d4, encoder_features[-2])
        d4 = torch.cat([d4, s4], dim=1)
        d4 = self.conv4(d4)
        
        d3 = self.up_conv3(d4)
        s3 = self.att3(d3, encoder_features[-3])
        d3 = torch.cat([d3, s3], dim=1)
        d3 = self.conv3(d3)
        
        d2 = self.up_conv2(d3)
        s2 = self.att2(d2, encoder_features[-4])
        d2 = torch.cat([d2, s2], dim=1)
        d2 = self.conv2(d2)
        
        d1 = self.up_conv1(d2)
        s1 = self.att1(d1, encoder_features[-5])
        d1 = torch.cat([d1, s1], dim=1)
        d1 = self.conv1(d1)
        
        output = self.final_conv(d1)
        
        return output, deep_features

class FeatureExtractor:
    """Feature extraction using Attention U-Net with EfficientNet"""
    def __init__(self, model_path=None, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AttentionUNet(num_classes=1, efficientnet_version='b0')
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        else:
            print("Using pretrained EfficientNet backbone (model not trained for specific task)")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image, mask=None):
        """Preprocess image for the model"""
        # Apply mask if provided
        if mask is not None:
            # Convert mask to 3 channels
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) / 255.0
            image = image * mask_3ch
        
        # Convert to tensor
        image_tensor = self.transform(image.astype(np.uint8))
        return image_tensor.unsqueeze(0).to(self.device)
    
    def extract_features(self, image, mask=None):
        """Extract deep features from the image"""
        with torch.no_grad():
            # Preprocess
            input_tensor = self.preprocess_image(image, mask)
            
            # Extract features
            features = self.model(input_tensor, return_features=True)
            
            # Convert to numpy
            features_np = features.cpu().numpy().flatten()
            
        return features_np
    
    def segment_and_extract(self, image, mask=None):
        """Perform segmentation and feature extraction"""
        with torch.no_grad():
            # Preprocess
            input_tensor = self.preprocess_image(image, mask)
            
            # Forward pass
            segmentation, features = self.model(input_tensor)
            
            # Process segmentation output
            seg_output = torch.sigmoid(segmentation).cpu().numpy()[0, 0]
            seg_output = (seg_output * 255).astype(np.uint8)
            seg_output = cv2.resize(seg_output, (image.shape[1], image.shape[0]))
            
            # Process features
            features_np = features.cpu().numpy().flatten()
            
        return seg_output, features_np
    
class StainNormalizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Macenko Stain Normalization for H&E Images")
        self.root.geometry("800x600")
        
        self.normalizer = MacenkoStainNormalizer()
        self.feature_extractor = FeatureExtractor()
        self.original_image = None
        self.normalized_image = None
        self.current_mask = None
        self.extracted_features = None
        self.segmentation_output = None
        self.mask_type = tk.StringVar(value="tissue")
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Macenko Stain Normalization", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Upload button
        upload_btn = ttk.Button(main_frame, text="Upload H&E Image", 
                            command=self.upload_image)
        upload_btn.grid(row=1, column=0, columnspan=3, pady=(0, 10))
        
        # Image display frame
        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=2, column=0, columnspan=3, pady=(0, 20))
        
        # Original image label
        self.orig_label = ttk.Label(image_frame, text="Original Image")
        self.orig_label.grid(row=0, column=0, padx=(0, 5))
        
        # Normalized image label  
        self.norm_label = ttk.Label(image_frame, text="Normalized Image")
        self.norm_label.grid(row=0, column=1, padx=(5, 5))
        
        # Mask label
        self.mask_label = ttk.Label(image_frame, text="Binary Mask")
        self.mask_label.grid(row=0, column=2, padx=(5, 0))
        
        # Process button
        self.process_btn = ttk.Button(main_frame, text="Normalize Staining", 
                                    command=self.process_image, state="disabled")
        self.process_btn.grid(row=3, column=0, columnspan=3, pady=(0, 10))
        
        # Mask options frame
        mask_frame = ttk.LabelFrame(main_frame, text="Binary Mask Options", padding="10")
        mask_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Mask type selection
        ttk.Label(mask_frame, text="Mask Type:").grid(row=0, column=0, sticky=tk.W)
        
        mask_types = [("Tissue Segmentation", "tissue"),
                      ("Nuclei Detection", "nuclei"), 
                      ("Eosin Regions", "eosin")]
        
        for i, (text, value) in enumerate(mask_types):
            ttk.Radiobutton(mask_frame, text=text, variable=self.mask_type, 
                           value=value).grid(row=0, column=i+1, padx=(10, 0))
        
        # Create mask button
        self.mask_btn = ttk.Button(mask_frame, text="Generate Binary Mask", 
                                  command=self.create_mask, state="disabled")
        self.mask_btn.grid(row=1, column=0, columnspan=4, pady=(10, 0))
        
        # Feature extraction frame
        feature_frame = ttk.LabelFrame(main_frame, text="AI Feature Extraction", padding="10")
        feature_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Feature extraction options
        self.use_mask_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(feature_frame, text="Apply mask during feature extraction", 
                       variable=self.use_mask_var).grid(row=0, column=0, sticky=tk.W)
        
        # Feature extraction buttons
        feature_btn_frame = ttk.Frame(feature_frame)
        feature_btn_frame.grid(row=1, column=0, columnspan=3, pady=(10, 0))
        
        self.extract_btn = ttk.Button(feature_btn_frame, text="Extract Deep Features", 
                                     command=self.extract_features, state="disabled")
        self.extract_btn.grid(row=0, column=0, padx=(0, 5))
        
        self.segment_btn = ttk.Button(feature_btn_frame, text="AI Segmentation + Features", 
                                     command=self.ai_segmentation, state="disabled")
        self.segment_btn.grid(row=0, column=1, padx=(5, 0))
        
        # Feature display
        self.feature_text = tk.Text(feature_frame, height=8, width=80, wrap=tk.WORD)
        self.feature_text.grid(row=2, column=0, columnspan=3, pady=(10, 0))
        
        # Scrollbar for feature text
        scrollbar = ttk.Scrollbar(feature_frame, orient="vertical", command=self.feature_text.yview)
        scrollbar.grid(row=2, column=3, sticky=(tk.N, tk.S), pady=(10, 0))
        self.feature_text.configure(yscrollcommand=scrollbar.set)
        
        # Save buttons frame
        save_frame = ttk.Frame(main_frame)
        save_frame.grid(row=6, column=0, columnspan=3, pady=(0, 10))
        
        # Save normalized image button
        self.save_norm_btn = ttk.Button(save_frame, text="Save Normalized", 
                                       command=self.save_normalized_image, state="disabled")
        self.save_norm_btn.grid(row=0, column=0, padx=(0, 3))
        
        # Save mask button
        self.save_mask_btn = ttk.Button(save_frame, text="Save Mask", 
                                       command=self.save_mask, state="disabled")
        self.save_mask_btn.grid(row=0, column=1, padx=(3, 3))
        
        # Save features button
        self.save_features_btn = ttk.Button(save_frame, text="Save Features", 
                                           command=self.save_features, state="disabled")
        self.save_features_btn.grid(row=0, column=2, padx=(3, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready to upload image")
        self.status_label.grid(row=8, column=0, columnspan=3)
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select H&E Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        )
        
        if file_path:
            try:
                # Load and display original image
                self.original_image = cv2.imread(file_path)
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                
                # Resize for display if too large
                display_img = self.resize_for_display(self.original_image)
                
                # Convert to PIL and display
                pil_img = Image.fromarray(display_img)
                photo = tk.PhotoImage(data=self.pil_to_base64(pil_img))
                
                self.orig_label.configure(image=photo)
                self.orig_label.image = photo
                
                self.process_btn.configure(state="normal")
                self.mask_btn.configure(state="normal")
                self.extract_btn.configure(state="normal")
                self.segment_btn.configure(state="normal")
                self.status_label.configure(text=f"Image loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                
    def resize_for_display(self, image, max_size=300):
        """Resize image for display while maintaining aspect ratio"""
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(image, (new_w, new_h))
        return image
    
    def pil_to_base64(self, pil_img):
        """Convert PIL image to base64 for tkinter display"""
        import io
        import base64
        
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue())
        return img_str
    
    def process_image(self):
        if self.original_image is None:
            return
            
        try:
            self.progress.start()
            self.status_label.configure(text="Processing image...")
            self.root.update()
            
            # Normalize the image
            self.normalized_image = self.normalizer.normalize_he(self.original_image)
            
            # Resize and display normalized image
            display_img = self.resize_for_display(self.normalized_image)
            pil_img = Image.fromarray(display_img)
            photo = tk.PhotoImage(data=self.pil_to_base64(pil_img))
            
            self.norm_label.configure(image=photo)
            self.norm_label.image = photo
            
            self.save_norm_btn.configure(state="normal")
            self.progress.stop()
            self.status_label.configure(text="Normalization complete!")
            
        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
            self.status_label.configure(text="Processing failed")
    
    def create_mask(self):
        """Create binary mask based on selected type"""
        if self.original_image is None:
            return
            
        try:
            self.progress.start()
            self.status_label.configure(text="Creating binary mask...")
            self.root.update()
            
            mask_type = self.mask_type.get()
            
            if mask_type == "tissue":
                self.current_mask = self.normalizer.create_tissue_mask(self.original_image)
            elif mask_type == "nuclei":
                self.current_mask = self.normalizer.create_nuclei_mask(self.original_image)
            elif mask_type == "eosin":
                self.current_mask = self.normalizer.create_eosin_mask(self.original_image)
            
            # Resize and display mask
            display_mask = self.resize_for_display(self.current_mask)
            
            # Convert grayscale mask to RGB for display
            mask_rgb = cv2.cvtColor(display_mask, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(mask_rgb)
            photo = tk.PhotoImage(data=self.pil_to_base64(pil_img))
            
            self.mask_label.configure(image=photo)
            self.mask_label.image = photo
            
            self.save_mask_btn.configure(state="normal")
            self.progress.stop()
            self.status_label.configure(text=f"Binary mask created: {mask_type}")
            
        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Error", f"Failed to create mask: {str(e)}")
            self.status_label.configure(text="Mask creation failed")
            
    def save_normalized_image(self):
        """Save the normalized image"""
        if self.normalized_image is None:
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Normalized Image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")]
        )
        
        if file_path:
            try:
                pil_img = Image.fromarray(self.normalized_image)
                pil_img.save(file_path)
                self.status_label.configure(text=f"Normalized image saved: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", "Normalized image saved successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save normalized image: {str(e)}")
    
    def save_mask(self):
        """Save the binary mask"""
        if self.current_mask is None:
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Binary Mask",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("TIFF files", "*.tif")]
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.current_mask)
                self.status_label.configure(text=f"Binary mask saved: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", "Binary mask saved successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save mask: {str(e)}")
    
    def extract_features(self):
        """Extract deep features using Attention U-Net"""
        image_to_use = self.normalized_image if self.normalized_image is not None else self.original_image
        
        if image_to_use is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        try:
            self.progress.start()
            self.status_label.configure(text="Extracting deep features...")
            self.root.update()
            
            # Determine if we should use mask
            mask_to_use = None
            if self.use_mask_var.get() and self.current_mask is not None:
                mask_to_use = self.current_mask
            
            # Extract features
            self.extracted_features = self.feature_extractor.extract_features(
                image_to_use, mask_to_use
            )
            
            # Display feature statistics
            self.display_feature_stats()
            
            self.save_features_btn.configure(state="normal")
            self.progress.stop()
            self.status_label.configure(text="Deep features extracted successfully!")
            
        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Error", f"Failed to extract features: {str(e)}")
            self.status_label.configure(text="Feature extraction failed")
    
    def ai_segmentation(self):
        """Perform AI segmentation and feature extraction"""
        image_to_use = self.normalized_image if self.normalized_image is not None else self.original_image
        
        if image_to_use is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        try:
            self.progress.start()
            self.status_label.configure(text="Running AI segmentation and feature extraction...")
            self.root.update()
            
            # Determine if we should use mask
            mask_to_use = None
            if self.use_mask_var.get() and self.current_mask is not None:
                mask_to_use = self.current_mask
            
            # Perform segmentation and feature extraction
            self.segmentation_output, self.extracted_features = self.feature_extractor.segment_and_extract(
                image_to_use, mask_to_use
            )
            
            # Display AI segmentation result
            display_seg = self.resize_for_display(self.segmentation_output)
            seg_rgb = cv2.cvtColor(display_seg, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(seg_rgb)
            photo = tk.PhotoImage(data=self.pil_to_base64(pil_img))
            
            # Update mask display with AI segmentation
            self.mask_label.configure(image=photo)
            self.mask_label.image = photo
            
            # Display feature statistics
            self.display_feature_stats(ai_seg=True)
            
            self.save_features_btn.configure(state="normal")
            self.save_mask_btn.configure(state="normal")
            self.progress.stop()
            self.status_label.configure(text="AI segmentation and feature extraction complete!")
            
        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Error", f"Failed to perform AI analysis: {str(e)}")
            self.status_label.configure(text="AI analysis failed")
    
    def display_feature_stats(self, ai_seg=False):
        """Display feature extraction statistics"""
        self.feature_text.delete(1.0, tk.END)
        
        if self.extracted_features is None:
            return
        
        # Feature statistics
        feature_info = f"{'='*60}\n"
        feature_info += f"ATTENTION U-NET + EFFICIENTNET FEATURE EXTRACTION\n"
        feature_info += f"{'='*60}\n\n"
        
        if ai_seg:
            feature_info += f"AI SEGMENTATION + FEATURE EXTRACTION COMPLETE\n\n"
        else:
            feature_info += f" DEEP FEATURE EXTRACTION COMPLETE\n\n"
        
        feature_info += f"Feature Vector Statistics:\n"
        feature_info += f"• Feature dimension: {len(self.extracted_features)}\n"
        feature_info += f"• Mean activation: {np.mean(self.extracted_features):.6f}\n"
        feature_info += f"• Standard deviation: {np.std(self.extracted_features):.6f}\n"
        feature_info += f"• Min activation: {np.min(self.extracted_features):.6f}\n"
        feature_info += f"• Max activation: {np.max(self.extracted_features):.6f}\n"
        feature_info += f"• Non-zero features: {np.count_nonzero(self.extracted_features)}\n"
        feature_info += f"• Sparsity: {(1 - np.count_nonzero(self.extracted_features)/len(self.extracted_features))*100:.2f}%\n\n"
        
        # Top activated features
        top_indices = np.argsort(np.abs(self.extracted_features))[-10:][::-1]
        feature_info += f"Top 10 Activated Features:\n"
        for i, idx in enumerate(top_indices):
            feature_info += f"• Feature {idx:3d}: {self.extracted_features[idx]:8.6f}\n"
        
        feature_info += f"\n"
        feature_info += f"Model Architecture:\n"
        feature_info += f"• Encoder: EfficientNet-B0 (pretrained)\n"
        feature_info += f"• Decoder: U-Net with Attention Gates\n"
        feature_info += f"• Feature extractor: 256-dimensional embedding\n"
        feature_info += f"• Device: {self.feature_extractor.device}\n\n"
        
        if self.use_mask_var.get() and self.current_mask is not None:
            feature_info += f"✓ Mask applied during feature extraction\n"
        else:
            feature_info += f"✗ No mask applied during feature extraction\n"
        
        feature_info += f"\nThese features can be used for:\n"
        feature_info += f"• Tissue classification\n"
        feature_info += f"• Disease detection\n"
        feature_info += f"• Morphological analysis\n"
        feature_info += f"• Similarity search\n"
        feature_info += f"• Clustering analysis\n"
        
        self.feature_text.insert(tk.END, feature_info)
    
    def save_features(self):
        """Save extracted features"""
        if self.extracted_features is None:
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Feature Vector",
            defaultextension=".npy",
            filetypes=[("NumPy files", "*.npy"), ("CSV files", "*.csv"), ("Text files", "*.txt")]
        )
        
        if file_path:
            try:
                if file_path.endswith('.npy'):
                    np.save(file_path, self.extracted_features)
                elif file_path.endswith('.csv'):
                    np.savetxt(file_path, self.extracted_features.reshape(1, -1), delimiter=',')
                else:  # .txt
                    np.savetxt(file_path, self.extracted_features)
                
                self.status_label.configure(text=f"Features saved: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", "Features saved successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save features: {str(e)}")

def main():
    root = tk.Tk()
    app = StainNormalizationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()