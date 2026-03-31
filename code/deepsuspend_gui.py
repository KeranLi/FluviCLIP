"""
DeepSuspend: GUI-based software for operational SSC monitoring.
Integrates knowledge-distilled lightweight model for real-time inference.
"""
import os
import sys
import torch
import numpy as np
from osgeo import gdal
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import json

from models.fluviclip import FluviCLIP
from models.distillation import LightweightStudent
from utils.uncertainty import mc_dropout_predict
from configs.FluviCLIP import Config


class DeepSuspendApp:
    """
    DeepSuspend GUI application for SSC estimation from satellite imagery.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("DeepSuspend - FluviCLIP SSC Estimation System")
        self.root.geometry("1200x800")
        
        self.config = Config()
        self.device = self.config.device
        self.model = None
        self.model_type = "teacher"  # or "student"
        self.input_files = []
        self.results = []
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="DeepSuspend: SSC Estimation from Remote Sensing",
            font=("Helvetica", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Step 1: Data Import
        step1_frame = ttk.LabelFrame(main_frame, text="Step 1: Data Import", padding="10")
        step1_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.import_btn = ttk.Button(
            step1_frame, 
            text="Import Satellite Images",
            command=self.import_images
        )
        self.import_btn.pack(side=tk.LEFT, padx=5)
        
        self.file_label = ttk.Label(step1_frame, text="No files selected")
        self.file_label.pack(side=tk.LEFT, padx=10)
        
        # Step 2: Model Selection
        step2_frame = ttk.LabelFrame(main_frame, text="Step 2: Model Selection", padding="10")
        step2_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.model_var = tk.StringVar(value="teacher")
        ttk.Radiobutton(
            step2_frame, 
            text="High-Precision Model (287M params, GPU)",
            variable=self.model_var,
            value="teacher",
            command=self.on_model_change
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            step2_frame,
            text="Lightweight Model (15M params, CPU/GPU)",
            variable=self.model_var,
            value="student",
            command=self.on_model_change
        ).pack(side=tk.LEFT, padx=5)
        
        # Step 3: Processing
        step3_frame = ttk.LabelFrame(main_frame, text="Step 3: Processing", padding="10")
        step3_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.process_btn = ttk.Button(
            step3_frame,
            text="Start Inference",
            command=self.start_inference,
            state=tk.DISABLED
        )
        self.process_btn.pack(pady=10)
        
        self.mc_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            step3_frame,
            text="Enable Uncertainty Quantification (Monte Carlo Dropout)",
            variable=self.mc_var
        ).pack(pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(step3_frame, mode='determinate', length=300)
        self.progress.pack(pady=10)
        
        self.status_label = ttk.Label(step3_frame, text="Ready")
        self.status_label.pack(pady=5)
        
        # Results display
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=3, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=10)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Treeview for results
        columns = ('filename', 'ssc', 'uncertainty', 'gate')
        self.tree = ttk.Treeview(results_frame, columns=columns, show='headings')
        self.tree.heading('filename', text='Filename')
        self.tree.heading('ssc', text='SSC (g/m³)')
        self.tree.heading('uncertainty', text='Uncertainty (±)')
        self.tree.heading('gate', text='Gating Weight')
        
        self.tree.column('filename', width=300)
        self.tree.column('ssc', width=100, anchor=tk.CENTER)
        self.tree.column('uncertainty', width=100, anchor=tk.CENTER)
        self.tree.column('gate', width=100, anchor=tk.CENTER)
        
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=4, column=0, columnspan=3, pady=10)
        
        ttk.Button(
            btn_frame,
            text="Export Results",
            command=self.export_results
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            btn_frame,
            text="View Visualization",
            command=self.view_visualization
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            btn_frame,
            text="Exit",
            command=self.root.quit
        ).pack(side=tk.LEFT, padx=5)
    
    def load_model(self):
        """Load the default model."""
        try:
            if self.model_var.get() == "teacher":
                self.model = FluviCLIP(
                    img_size=self.config.img_size,
                    patch_size=self.config.patch_size,
                    in_chans=self.config.in_chans,
                    embed_dim=self.config.embed_dim,
                    depths=self.config.depths,
                    num_heads=self.config.num_heads,
                ).to(self.device)
            else:
                self.model = LightweightStudent(
                    img_size=self.config.img_size,
                    patch_size=self.config.patch_size,
                    in_chans=self.config.in_chans,
                ).to(self.device)
            
            self.status_label.config(text="Model loaded successfully")
        except Exception as e:
            self.status_label.config(text=f"Error loading model: {str(e)}")
    
    def on_model_change(self):
        """Handle model type change."""
        self.model_type = self.model_var.get()
        self.load_model()
    
    def import_images(self):
        """Import satellite images for processing."""
        files = filedialog.askopenfilenames(
            title="Select Satellite Images",
            filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")]
        )
        if files:
            self.input_files = list(files)
            self.file_label.config(text=f"{len(self.input_files)} files selected")
            self.process_btn.config(state=tk.NORMAL)
            self.status_label.config(text=f"Ready to process {len(self.input_files)} images")
    
    def start_inference(self):
        """Start the inference process in a separate thread."""
        self.process_btn.config(state=tk.DISABLED)
        self.progress['maximum'] = len(self.input_files)
        self.progress['value'] = 0
        self.results = []
        
        thread = threading.Thread(target=self.run_inference)
        thread.start()
    
    def run_inference(self):
        """Run inference on all input files."""
        try:
            for i, file_path in enumerate(self.input_files):
                self.update_status(f"Processing {os.path.basename(file_path)}...")
                
                # Load and preprocess image
                image = self.load_image(file_path)
                
                # Run inference
                if self.mc_var.get() and self.model_type == "teacher":
                    mean_pred, std_pred, _ = mc_dropout_predict(
                        self.model, image, n_samples=30, device=self.device
                    )
                    pred = mean_pred
                    uncertainty = std_pred
                else:
                    with torch.no_grad():
                        image_batch = image.unsqueeze(0).to(self.device)
                        if self.model_type == "teacher":
                            pred, gate = self.model(image_batch, texts=None)
                            pred = pred.item()
                            gate = gate.item()
                        else:
                            pred = self.model(image_batch).item()
                            gate = None
                    uncertainty = None
                
                result = {
                    'filename': os.path.basename(file_path),
                    'filepath': file_path,
                    'ssc': pred,
                    'uncertainty': uncertainty,
                    'gate': gate
                }
                self.results.append(result)
                
                # Update UI
                self.progress['value'] = i + 1
                self.root.update_idletasks()
            
            self.update_status(f"Completed processing {len(self.input_files)} images")
            self.display_results()
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
        finally:
            self.process_btn.config(state=tk.NORMAL)
    
    def load_image(self, image_path):
        """Load and preprocess a satellite image."""
        image_ds = gdal.Open(image_path)
        if image_ds is None:
            raise FileNotFoundError(f"Cannot open {image_path}")
        
        image = []
        for b in range(1, image_ds.RasterCount + 1):
            band = image_ds.GetRasterBand(b)
            image.append(band.ReadAsArray())
        image = np.stack(image, axis=0)
        image = image.astype(np.float32)
        
        # Convert to tensor and resize
        image = torch.from_numpy(image).float()
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Normalize (using default stats if not computed)
        for i in range(image.shape[0]):
            mean = image[i].mean()
            std = image[i].std() + 1e-6
            image[i] = (image[i] - mean) / std
        
        return image
    
    def update_status(self, message):
        """Update status label thread-safely."""
        self.root.after(0, lambda: self.status_label.config(text=message))
    
    def display_results(self):
        """Display results in the treeview."""
        # Clear existing
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Insert new results
        for result in self.results:
            values = (
                result['filename'],
                f"{result['ssc']:.2f}",
                f"{result['uncertainty']:.2f}" if result['uncertainty'] else "N/A",
                f"{result['gate']:.3f}" if result['gate'] is not None else "N/A"
            )
            self.tree.insert('', tk.END, values=values)
    
    def export_results(self):
        """Export results to CSV file."""
        if not self.results:
            messagebox.showwarning("Warning", "No results to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            import csv
            with open(file_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['filename', 'ssc', 'uncertainty', 'gate'])
                writer.writeheader()
                for result in self.results:
                    writer.writerow({
                        'filename': result['filename'],
                        'ssc': result['ssc'],
                        'uncertainty': result['uncertainty'] if result['uncertainty'] else '',
                        'gate': result['gate'] if result['gate'] is not None else ''
                    })
            messagebox.showinfo("Success", f"Results exported to {file_path}")
    
    def view_visualization(self):
        """Open visualization window for selected result."""
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("Info", "Please select a result to visualize")
            return
        
        # Get selected item
        item = self.tree.item(selected[0])
        filename = item['values'][0]
        
        # Find corresponding result
        result = next((r for r in self.results if r['filename'] == filename), None)
        if result:
            self.show_visualization_window(result)
    
    def show_visualization_window(self, result):
        """Show detailed visualization window."""
        viz_window = tk.Toplevel(self.root)
        viz_window.title(f"Visualization - {result['filename']}")
        viz_window.geometry("800x600")
        
        # Load and display image
        image = self.load_image(result['filepath'])
        img_np = image[:3].numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        img_np = (img_np * 255).astype(np.uint8)
        
        img = Image.fromarray(img_np)
        img = img.resize((400, 400), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        
        img_label = ttk.Label(viz_window, image=img_tk)
        img_label.image = img_tk
        img_label.pack(pady=10)
        
        # Info frame
        info_frame = ttk.Frame(viz_window)
        info_frame.pack(pady=10)
        
        ttk.Label(info_frame, text=f"SSC Prediction: {result['ssc']:.2f} g/m³", 
                 font=("Helvetica", 12, "bold")).pack()
        if result['uncertainty']:
            ttk.Label(info_frame, text=f"Uncertainty: ±{result['uncertainty']:.2f} g/m³").pack()
        if result['gate'] is not None:
            ttk.Label(info_frame, text=f"Gating Weight: {result['gate']:.3f}").pack()


def main():
    root = tk.Tk()
    app = DeepSuspendApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
