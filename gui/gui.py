"""
GUI Application for Visualization

A simple GUI to select visualization types and data files from the data/ folder.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys

# Add project root to path (parent of gui/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src import parser
from src.plotter import UnifiedPlotter


class VisualizationGUI:
    """Main GUI application for visualization selection."""

    def __init__(self, root):
        self.root = root
        self.root.title("PyVista Visualization Tool")
        self.root.geometry("800x700")
        self.root.resizable(True, True)

        # Data directory (in project root)
        self.data_dir = os.path.join(PROJECT_ROOT, "data")

        # Store selected items
        self.selected_items = []

        # Categorize files
        self.file_categories = self._categorize_files()

        # Build GUI
        self._create_widgets()

    def _categorize_files(self):
        """Categorize files in data/ folder by type."""
        categories = {
            "XYZ Point Cloud (.dat)": [],
            "Boundary File (boundary.txt)": [],
            "VTK Mesh (.vtk)": [],
            "Patran Mesh (.msh)": [],
            "Vector Field (.dat)": [],
        }

        if not os.path.exists(self.data_dir):
            return categories

        for filename in sorted(os.listdir(self.data_dir)):
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.isfile(filepath):
                continue

            if filename == "boundary.txt":
                categories["Boundary File (boundary.txt)"].append(filename)
            elif filename.endswith(".vtk"):
                categories["VTK Mesh (.vtk)"].append(filename)
            elif filename.endswith(".msh"):
                categories["Patran Mesh (.msh)"].append(filename)
            elif filename.endswith(".dat") or filename.endswith(".txt"):
                if filename != "boundary.txt":
                    n_cols = parser.detect_file_columns(filepath)
                    if n_cols >= 6:
                        categories["Vector Field (.dat)"].append(filename)
                    else:
                        categories["XYZ Point Cloud (.dat)"].append(filename)

        return categories

    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="PyVista Visualization Tool",
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=(0, 10))

        # Instructions
        instructions = ttk.Label(
            main_frame,
            text="Select visualization items to add to the plot. Multiple items can be combined.",
            wraplength=750
        )
        instructions.pack(pady=(0, 10))

        # Create notebook for different visualization types
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Create tabs for each category
        self._create_point_cloud_tab()
        self._create_boundary_tab()
        self._create_vtk_tab()
        self._create_patran_tab()
        self._create_wire_tab()
        self._create_vector_field_tab()

        # Selected items list
        selected_frame = ttk.LabelFrame(main_frame, text="Selected Visualizations", padding="5")
        selected_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.selected_listbox = tk.Listbox(selected_frame, height=6)
        self.selected_listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        scrollbar = ttk.Scrollbar(selected_frame, orient=tk.VERTICAL, command=self.selected_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.selected_listbox.config(yscrollcommand=scrollbar.set)

        # Buttons for selected list
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(btn_frame, text="Remove Selected", command=self._remove_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear All", command=self._clear_all).pack(side=tk.LEFT, padx=5)

        # Clip plane options
        clip_frame = ttk.LabelFrame(main_frame, text="Clip Plane (Cross-Section View)", padding="5")
        clip_frame.pack(fill=tk.X, pady=(0, 10))

        self.clip_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            clip_frame,
            text="Enable Clip Plane",
            variable=self.clip_enabled
        ).pack(anchor=tk.W)

        clip_options_frame = ttk.Frame(clip_frame)
        clip_options_frame.pack(fill=tk.X, pady=5)

        ttk.Label(clip_options_frame, text="Clip Direction:").pack(side=tk.LEFT, padx=(0, 5))
        self.clip_direction = tk.StringVar(value="y")
        clip_combo = ttk.Combobox(
            clip_options_frame,
            textvariable=self.clip_direction,
            values=["x", "y", "z", "-x", "-y", "-z"],
            width=10,
            state="readonly"
        )
        clip_combo.pack(side=tk.LEFT, padx=5)

        # Plot button frame - make it very visible
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(fill=tk.X, pady=15)

        # Style the button to be more prominent
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Helvetica", 14, "bold"), padding=10)

        # Create a large, prominent button
        plot_btn = tk.Button(
            plot_frame,
            text="🚀 GENERATE PLOT",
            command=self._generate_plot,
            font=("Helvetica", 14, "bold"),
            bg="#4CAF50",
            fg="white",
            activebackground="#45a049",
            activeforeground="white",
            relief=tk.RAISED,
            bd=3,
            padx=30,
            pady=10,
            cursor="hand2"
        )
        plot_btn.pack(pady=10, expand=True)

    def _create_point_cloud_tab(self):
        """Create tab for XYZ point cloud files."""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Point Clouds")

        ttk.Label(frame, text="Select XYZ data file:").pack(anchor=tk.W)

        # File list
        file_frame = ttk.Frame(frame)
        file_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.pc_listbox = tk.Listbox(file_frame, selectmode=tk.SINGLE, height=8)
        self.pc_listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        scrollbar = ttk.Scrollbar(file_frame, orient=tk.VERTICAL, command=self.pc_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.pc_listbox.config(yscrollcommand=scrollbar.set)

        for f in self.file_categories["XYZ Point Cloud (.dat)"]:
            self.pc_listbox.insert(tk.END, f)

        # Options
        options_frame = ttk.Frame(frame)
        options_frame.pack(fill=tk.X, pady=5)

        ttk.Label(options_frame, text="Color:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.pc_color = tk.StringVar(value="red")
        color_combo = ttk.Combobox(
            options_frame,
            textvariable=self.pc_color,
            values=["red", "green", "blue", "yellow", "orange", "purple", "cyan", "magenta", "white", "black"],
            width=15
        )
        color_combo.grid(row=0, column=1, padx=5)

        ttk.Label(options_frame, text="Point Size:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.pc_size = tk.IntVar(value=5)
        size_spinbox = ttk.Spinbox(options_frame, from_=1, to=20, textvariable=self.pc_size, width=5)
        size_spinbox.grid(row=0, column=3, padx=5)

        ttk.Label(options_frame, text="Label:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.pc_label = tk.StringVar()
        ttk.Entry(options_frame, textvariable=self.pc_label, width=20).grid(row=1, column=1, padx=5, pady=5)

        # Add button
        ttk.Button(frame, text="Add Point Cloud", command=self._add_point_cloud).pack(pady=5)

    def _create_boundary_tab(self):
        """Create tab for boundary files."""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Boundary")

        ttk.Label(frame, text="Select boundary file:").pack(anchor=tk.W)

        # File list
        file_frame = ttk.Frame(frame)
        file_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.boundary_listbox = tk.Listbox(file_frame, selectmode=tk.SINGLE, height=4)
        self.boundary_listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        for f in self.file_categories["Boundary File (boundary.txt)"]:
            self.boundary_listbox.insert(tk.END, f)

        # Options
        options_frame = ttk.Frame(frame)
        options_frame.pack(fill=tk.X, pady=5)

        ttk.Label(options_frame, text="Color:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.boundary_color = tk.StringVar(value="cyan")
        color_combo = ttk.Combobox(
            options_frame,
            textvariable=self.boundary_color,
            values=["cyan", "lightblue", "blue", "green", "yellow", "orange", "red", "purple", "white", "gray"],
            width=15
        )
        color_combo.grid(row=0, column=1, padx=5)

        ttk.Label(options_frame, text="Opacity:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.boundary_opacity = tk.DoubleVar(value=1.0)
        opacity_spinbox = ttk.Spinbox(options_frame, from_=0.1, to=1.0, increment=0.1, textvariable=self.boundary_opacity, width=5)
        opacity_spinbox.grid(row=0, column=3, padx=5)

        ttk.Label(options_frame, text="Toroidal Res (n_phi):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.boundary_nphi = tk.IntVar(value=81)
        ttk.Spinbox(options_frame, from_=20, to=200, textvariable=self.boundary_nphi, width=5).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(options_frame, text="Poloidal Res (n_s):").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.boundary_ns = tk.IntVar(value=10)
        ttk.Spinbox(options_frame, from_=5, to=50, textvariable=self.boundary_ns, width=5).grid(row=1, column=3, padx=5, pady=5)

        self.boundary_edges = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Show Edges", variable=self.boundary_edges).grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)

        ttk.Label(options_frame, text="Label:").grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        self.boundary_label = tk.StringVar(value="Boundary")
        ttk.Entry(options_frame, textvariable=self.boundary_label, width=15).grid(row=2, column=3, padx=5, pady=5)

        # Add button
        ttk.Button(frame, text="Add Boundary", command=self._add_boundary).pack(pady=5)

    def _create_vtk_tab(self):
        """Create tab for VTK mesh files."""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="VTK Mesh")

        ttk.Label(frame, text="Select VTK mesh file:").pack(anchor=tk.W)

        # File list
        file_frame = ttk.Frame(frame)
        file_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.vtk_listbox = tk.Listbox(file_frame, selectmode=tk.SINGLE, height=6)
        self.vtk_listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        scrollbar = ttk.Scrollbar(file_frame, orient=tk.VERTICAL, command=self.vtk_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.vtk_listbox.config(yscrollcommand=scrollbar.set)

        for f in self.file_categories["VTK Mesh (.vtk)"]:
            self.vtk_listbox.insert(tk.END, f)

        # Options
        options_frame = ttk.Frame(frame)
        options_frame.pack(fill=tk.X, pady=5)

        ttk.Label(options_frame, text="Color:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.vtk_color = tk.StringVar(value="lightblue")
        color_combo = ttk.Combobox(
            options_frame,
            textvariable=self.vtk_color,
            values=["lightblue", "cyan", "blue", "green", "yellow", "orange", "red", "purple", "white", "gray"],
            width=15
        )
        color_combo.grid(row=0, column=1, padx=5)

        ttk.Label(options_frame, text="Opacity:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.vtk_opacity = tk.DoubleVar(value=1.0)
        opacity_spinbox = ttk.Spinbox(options_frame, from_=0.1, to=1.0, increment=0.1, textvariable=self.vtk_opacity, width=5)
        opacity_spinbox.grid(row=0, column=3, padx=5)

        self.vtk_edges = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Show Edges", variable=self.vtk_edges).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)

        ttk.Label(options_frame, text="Label:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.vtk_label = tk.StringVar(value="VTK Mesh")
        ttk.Entry(options_frame, textvariable=self.vtk_label, width=15).grid(row=1, column=3, padx=5, pady=5)

        # Add button
        ttk.Button(frame, text="Add VTK Mesh", command=self._add_vtk).pack(pady=5)

    def _create_patran_tab(self):
        """Create tab for Patran mesh files."""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Patran Mesh")

        ttk.Label(frame, text="Select Patran mesh file (.msh):").pack(anchor=tk.W)

        # File list
        file_frame = ttk.Frame(frame)
        file_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.patran_listbox = tk.Listbox(file_frame, selectmode=tk.SINGLE, height=6)
        self.patran_listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        scrollbar = ttk.Scrollbar(file_frame, orient=tk.VERTICAL, command=self.patran_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.patran_listbox.config(yscrollcommand=scrollbar.set)

        for f in self.file_categories["Patran Mesh (.msh)"]:
            self.patran_listbox.insert(tk.END, f)

        # Options
        options_frame = ttk.Frame(frame)
        options_frame.pack(fill=tk.X, pady=5)

        ttk.Label(options_frame, text="Color:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.patran_color = tk.StringVar(value="violet")
        color_combo = ttk.Combobox(
            options_frame,
            textvariable=self.patran_color,
            values=["violet", "purple", "blue", "green", "yellow", "orange", "red", "cyan", "white", "gray"],
            width=15
        )
        color_combo.grid(row=0, column=1, padx=5)

        ttk.Label(options_frame, text="Opacity:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.patran_opacity = tk.DoubleVar(value=0.7)
        opacity_spinbox = ttk.Spinbox(options_frame, from_=0.1, to=1.0, increment=0.1, textvariable=self.patran_opacity, width=5)
        opacity_spinbox.grid(row=0, column=3, padx=5)

        self.patran_edges = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Show Edges", variable=self.patran_edges).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)

        ttk.Label(options_frame, text="Label:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.patran_label = tk.StringVar(value="Patran Mesh")
        ttk.Entry(options_frame, textvariable=self.patran_label, width=15).grid(row=1, column=3, padx=5, pady=5)

        # Add button
        ttk.Button(frame, text="Add Patran Mesh", command=self._add_patran).pack(pady=5)

    def _create_vector_field_tab(self):
        """Create tab for vector field (arrow) files."""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Vector Field")

        ttk.Label(
            frame,
            text="Select a file with columns:  x  y  z  Bx  By  Bz",
            wraplength=700
        ).pack(anchor=tk.W, pady=(0, 5))

        # File list
        file_frame = ttk.Frame(frame)
        file_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.vf_listbox = tk.Listbox(file_frame, selectmode=tk.SINGLE, height=6)
        self.vf_listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        scrollbar = ttk.Scrollbar(file_frame, orient=tk.VERTICAL, command=self.vf_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.vf_listbox.config(yscrollcommand=scrollbar.set)

        for f in self.file_categories["Vector Field (.dat)"]:
            self.vf_listbox.insert(tk.END, f)

        # Options
        options_frame = ttk.Frame(frame)
        options_frame.pack(fill=tk.X, pady=5)

        ttk.Label(options_frame, text="Arrow Color:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.vf_color = tk.StringVar(value="crimson")
        ttk.Combobox(
            options_frame,
            textvariable=self.vf_color,
            values=["crimson", "red", "orange", "green", "blue", "cyan",
                    "magenta", "yellow", "purple", "white", "black"],
            width=12,
            state="readonly"
        ).grid(row=0, column=1, padx=5)

        ttk.Label(options_frame, text="Scale Factor:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.vf_scale = tk.DoubleVar(value=0.1)
        ttk.Spinbox(
            options_frame, from_=0.001, to=100.0, increment=0.01,
            textvariable=self.vf_scale, width=8, format="%.3f"
        ).grid(row=0, column=3, padx=5)

        ttk.Label(options_frame, text="Subsample:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.vf_sample = tk.DoubleVar(value=1.0)
        ttk.Spinbox(
            options_frame, from_=0.01, to=1.0, increment=0.05,
            textvariable=self.vf_sample, width=6, format="%.2f"
        ).grid(row=1, column=1, padx=5, pady=5)

        self.vf_color_by_mag = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame, text="Color by |B|", variable=self.vf_color_by_mag
        ).grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)

        ttk.Label(options_frame, text="Colormap:").grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)
        self.vf_cmap = tk.StringVar(value="plasma")
        ttk.Combobox(
            options_frame,
            textvariable=self.vf_cmap,
            values=["plasma", "viridis", "inferno", "magma", "jet",
                    "hot", "coolwarm", "rainbow"],
            width=10
        ).grid(row=1, column=4, padx=5, pady=5)

        ttk.Label(options_frame, text="Label:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.vf_label = tk.StringVar(value="B field")
        ttk.Entry(options_frame, textvariable=self.vf_label, width=20).grid(row=2, column=1, padx=5)

        ttk.Button(frame, text="Add Vector Field", command=self._add_vector_field).pack(pady=5)

    def _create_wire_tab(self):
        """Create tab for wire (current filament loop) elements."""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Wire")

        ttk.Label(
            frame,
            text="Define a wire (current filament loop) by its position and tilt angle.",
            wraplength=700
        ).pack(anchor=tk.W, pady=(0, 10))

        # Parameters frame
        params_frame = ttk.LabelFrame(frame, text="Wire Parameters", padding="10")
        params_frame.pack(fill=tk.X, pady=5)

        ttk.Label(params_frame, text="r0  (major radius) [m]:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.wire_r0 = tk.StringVar(value="1.99141779000833")
        ttk.Entry(params_frame, textvariable=self.wire_r0, width=25).grid(row=0, column=1, padx=5, pady=3)

        ttk.Label(params_frame, text="z0  (axial position) [m]:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.wire_z0 = tk.StringVar(value="0.0")
        ttk.Entry(params_frame, textvariable=self.wire_z0, width=25).grid(row=1, column=1, padx=5, pady=3)

        ttk.Label(params_frame, text="alfa_wire [deg]:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=3)
        self.wire_alfa = tk.StringVar(value="3.0")
        ttk.Entry(params_frame, textvariable=self.wire_alfa, width=25).grid(row=2, column=1, padx=5, pady=3)

        # Display options frame
        options_frame = ttk.LabelFrame(frame, text="Display Options", padding="10")
        options_frame.pack(fill=tk.X, pady=5)

        ttk.Label(options_frame, text="Color:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.wire_color = tk.StringVar(value="orange")
        ttk.Combobox(
            options_frame,
            textvariable=self.wire_color,
            values=["orange", "red", "green", "blue", "yellow", "purple", "cyan", "magenta", "white", "gray"],
            width=15,
            state="readonly"
        ).grid(row=0, column=1, padx=5)

        ttk.Label(options_frame, text="Tube Radius [m] (0 = auto):").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.wire_tube_radius = tk.StringVar(value="0.0")
        ttk.Entry(options_frame, textvariable=self.wire_tube_radius, width=10).grid(row=0, column=3, padx=5)

        ttk.Label(options_frame, text="Label:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.wire_label = tk.StringVar(value="Wire")
        ttk.Entry(options_frame, textvariable=self.wire_label, width=20).grid(row=1, column=1, padx=5, pady=5)

        # Add button
        ttk.Button(frame, text="Add Wire", command=self._add_wire).pack(pady=10)

    def _add_point_cloud(self):
        """Add selected point cloud to the list."""
        selection = self.pc_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a file first.")
            return

        filename = self.pc_listbox.get(selection[0])
        label = self.pc_label.get() or filename

        item = {
            'type': 'point_cloud',
            'file': filename,
            'color': self.pc_color.get(),
            'point_size': self.pc_size.get(),
            'label': label
        }

        self.selected_items.append(item)
        self._update_selected_listbox()

    def _add_boundary(self):
        """Add selected boundary to the list."""
        selection = self.boundary_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a file first.")
            return

        filename = self.boundary_listbox.get(selection[0])
        label = self.boundary_label.get() or "Boundary"

        item = {
            'type': 'boundary',
            'file': filename,
            'color': self.boundary_color.get(),
            'opacity': self.boundary_opacity.get(),
            'n_phi': self.boundary_nphi.get(),
            'n_s': self.boundary_ns.get(),
            'show_edges': self.boundary_edges.get(),
            'label': label
        }

        self.selected_items.append(item)
        self._update_selected_listbox()

    def _add_vtk(self):
        """Add selected VTK mesh to the list."""
        selection = self.vtk_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a file first.")
            return

        filename = self.vtk_listbox.get(selection[0])
        label = self.vtk_label.get() or filename

        item = {
            'type': 'vtk',
            'file': filename,
            'color': self.vtk_color.get(),
            'opacity': self.vtk_opacity.get(),
            'show_edges': self.vtk_edges.get(),
            'label': label
        }

        self.selected_items.append(item)
        self._update_selected_listbox()

    def _add_patran(self):
        """Add selected Patran mesh to the list."""
        selection = self.patran_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a file first.")
            return

        filename = self.patran_listbox.get(selection[0])
        label = self.patran_label.get() or filename

        item = {
            'type': 'patran',
            'file': filename,
            'color': self.patran_color.get(),
            'opacity': self.patran_opacity.get(),
            'show_edges': self.patran_edges.get(),
            'label': label
        }

        self.selected_items.append(item)
        self._update_selected_listbox()

    def _add_vector_field(self):
        """Add vector field file to the list."""
        selection = self.vf_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a file first.")
            return

        filename = self.vf_listbox.get(selection[0])
        label = self.vf_label.get() or filename

        try:
            scale = float(self.vf_scale.get())
            sample = float(self.vf_sample.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Scale and subsample must be numeric.")
            return

        item = {
            'type': 'vector_field',
            'file': filename,
            'color': self.vf_color.get(),
            'scale': scale,
            'sample_frac': sample,
            'color_by_magnitude': self.vf_color_by_mag.get(),
            'colormap': self.vf_cmap.get(),
            'label': label,
        }

        self.selected_items.append(item)
        self._update_selected_listbox()

    def _add_wire(self):
        """Add wire element to the list."""
        try:
            r0 = float(self.wire_r0.get())
            z0 = float(self.wire_z0.get())
            alfa = float(self.wire_alfa.get())
            tube_radius = float(self.wire_tube_radius.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values for wire parameters.")
            return

        if r0 <= 0:
            messagebox.showerror("Invalid Input", "r0 must be a positive value.")
            return

        label = self.wire_label.get() or "Wire"
        item = {
            'type': 'wire',
            'r0': r0,
            'z0': z0,
            'alfa_wire_deg': alfa,
            'color': self.wire_color.get(),
            'tube_radius': tube_radius if tube_radius > 0 else None,
            'label': label
        }

        self.selected_items.append(item)
        self._update_selected_listbox()

    def _update_selected_listbox(self):
        """Update the selected items listbox."""
        self.selected_listbox.delete(0, tk.END)
        for item in self.selected_items:
            if item['type'] == 'wire':
                display = (
                    f"[WIRE] r0={item['r0']:.6f} m, z0={item['z0']:.6f} m, "
                    f"alfa={item['alfa_wire_deg']:.4f}° - {item['label']}"
                )
            elif item['type'] == 'vector_field':
                display = (
                    f"[VECTOR FIELD] {item['file']} scale={item['scale']:.3f} - {item['label']}"
                )
            else:
                display = f"[{item['type'].upper()}] {item['file']} - {item['label']}"
            self.selected_listbox.insert(tk.END, display)

    def _remove_selected(self):
        """Remove selected item from the list."""
        selection = self.selected_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        del self.selected_items[idx]
        self._update_selected_listbox()

    def _clear_all(self):
        """Clear all selected items."""
        self.selected_items = []
        self._update_selected_listbox()

    def _generate_plot(self):
        """Generate the visualization plot."""
        if not self.selected_items:
            messagebox.showwarning("No Items", "Please add at least one visualization item.")
            return

        try:
            # Create plotter
            plotter = UnifiedPlotter(background="white", title="Visualization")

            # Set clip plane if enabled
            if self.clip_enabled.get():
                plotter.set_clip_plane(self.clip_direction.get())

            # Add each selected item
            for item in self.selected_items:
                filepath = os.path.join(self.data_dir, item['file']) if item['type'] not in ('wire',) else None

                if item['type'] == 'point_cloud':
                    plotter.add_point_cloud(
                        filepath,
                        color=item['color'],
                        point_size=item['point_size'],
                        label=item['label']
                    )

                elif item['type'] == 'boundary':
                    plotter.add_boundary(
                        filepath,
                        color=item['color'],
                        opacity=item['opacity'],
                        n_phi=item['n_phi'],
                        n_s=item['n_s'],
                        show_edges=item['show_edges'],
                        label=item['label']
                    )

                elif item['type'] == 'vtk':
                    plotter.add_vtk_mesh(
                        filepath,
                        color=item['color'],
                        opacity=item['opacity'],
                        show_edges=item['show_edges'],
                        label=item['label']
                    )

                elif item['type'] == 'patran':
                    df_nodes, elements_hex = parser.read_patran_neutral(filepath)
                    plotter.add_hex_mesh(
                        df_nodes,
                        elements_hex,
                        color=item['color'],
                        opacity=item['opacity'],
                        show_edges=item['show_edges'],
                        label=item['label']
                    )

                elif item['type'] == 'vector_field':
                    plotter.add_vector_field(
                        filepath,
                        scale=item['scale'],
                        color=item['color'],
                        colormap=item['colormap'],
                        color_by_magnitude=item['color_by_magnitude'],
                        sample_frac=item['sample_frac'],
                        label=item['label'],
                    )

                elif item['type'] == 'wire':
                    plotter.add_wire(
                        r0=item['r0'],
                        z0=item['z0'],
                        alfa_wire_deg=item['alfa_wire_deg'],
                        color=item['color'],
                        tube_radius=item['tube_radius'],
                        label=item['label']
                    )

            # Show the plot
            plotter.show(
                show_axes=True,
                show_bounds=True,
                show_legend=True
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate plot:\n{str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point for the GUI application."""
    root = tk.Tk()
    app = VisualizationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

