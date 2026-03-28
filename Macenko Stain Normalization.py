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
    
class StainNormalizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Macenko Stain Normalization for H&E Images")
        self.root.geometry("800x600")
        
        self.normalizer = MacenkoStainNormalizer()
        self.original_image = None
        self.normalized_image = None
        self.current_mask = None
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
        
        # Save buttons frame
        save_frame = ttk.Frame(main_frame)
        save_frame.grid(row=5, column=0, columnspan=3, pady=(0, 10))
        
        # Save normalized image button
        self.save_norm_btn = ttk.Button(save_frame, text="Save Normalized Image", 
                                       command=self.save_normalized_image, state="disabled")
        self.save_norm_btn.grid(row=0, column=0, padx=(0, 5))
        
        # Save mask button
        self.save_mask_btn = ttk.Button(save_frame, text="Save Binary Mask", 
                                       command=self.save_mask, state="disabled")
        self.save_mask_btn.grid(row=0, column=1, padx=(5, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready to upload image")
        self.status_label.grid(row=7, column=0, columnspan=3)
        
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

def main():
    root = tk.Tk()
    app = StainNormalizationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()