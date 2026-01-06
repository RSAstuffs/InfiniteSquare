#!/usr/bin/env python3
"""
Modern GUI for Geometric Lattice Factorization Tool
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import queue
import sys
from io import StringIO
from Squarer import factor_with_lattice_compression, LatticePoint, GeometricLattice

class FactorizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Geometric Lattice Factorization Tool")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')
        
        # Style configuration
        self.setup_styles()
        
        # Control variables
        self.is_running = False
        self.output_queue = queue.Queue()
        self.worker_thread = None
        
        # Create UI
        self.create_widgets()
        
        # Start output monitor
        self.monitor_output()
    
    def setup_styles(self):
        """Configure modern styling."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', 
                       background='#1e1e1e', 
                       foreground='#00ff88',
                       font=('Segoe UI', 16, 'bold'))
        
        style.configure('Heading.TLabel',
                       background='#1e1e1e',
                       foreground='#ffffff',
                       font=('Segoe UI', 11, 'bold'))
        
        style.configure('Info.TLabel',
                       background='#1e1e1e',
                       foreground='#cccccc',
                       font=('Segoe UI', 9))
        
        style.configure('Modern.TButton',
                       font=('Segoe UI', 10),
                       padding=10)
        
        style.configure('Modern.TFrame',
                       background='#2d2d2d',
                       relief='flat')
    
    def create_widgets(self):
        """Create and layout all GUI widgets."""
        # Main container
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title = ttk.Label(main_frame, 
                         text="🔷 Geometric Lattice Factorization Engine",
                         style='Title.TLabel')
        title.pack(pady=(0, 20))
        
        # Left panel - Controls
        left_panel = tk.Frame(main_frame, bg='#2d2d2d', relief='flat', bd=0)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10), expand=False)
        left_panel.config(width=400)
        
        # Right panel - Output
        right_panel = tk.Frame(main_frame, bg='#1e1e1e')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_control_panel(left_panel)
        self.create_output_panel(right_panel)
    
    def create_control_panel(self, parent):
        """Create the control panel with input fields."""
        # Input section
        input_frame = tk.Frame(parent, bg='#2d2d2d')
        input_frame.pack(fill=tk.X, padx=15, pady=15)
        
        ttk.Label(input_frame, 
                 text="Input Parameters",
                 style='Heading.TLabel').pack(anchor='w', pady=(0, 10))
        
        # N input
        n_frame = tk.Frame(input_frame, bg='#2d2d2d')
        n_frame.pack(fill=tk.X, pady=5)
        ttk.Label(n_frame, text="Number to Factor (N):", 
                 style='Info.TLabel').pack(anchor='w')
        
        self.n_entry = tk.Text(n_frame, height=4, width=45,
                              bg='#1e1e1e', fg='#00ff88',
                              font=('Consolas', 10),
                              insertbackground='#00ff88',
                              relief='flat', bd=2)
        self.n_entry.pack(fill=tk.X, pady=(5, 0))
        self.n_entry.insert('1.0', '35')
        
        # Load from file button
        load_btn = tk.Button(n_frame, text="📁 Load from File",
                            command=self.load_from_file,
                            bg='#3d3d3d', fg='#ffffff',
                            font=('Segoe UI', 9),
                            relief='flat', cursor='hand2',
                            activebackground='#4d4d4d')
        load_btn.pack(anchor='w', pady=(5, 0))
        
        # Lattice size
        lattice_frame = tk.Frame(input_frame, bg='#2d2d2d')
        lattice_frame.pack(fill=tk.X, pady=10)
        ttk.Label(lattice_frame, text="Initial Lattice Size:",
                 style='Info.TLabel').pack(anchor='w')
        
        self.lattice_size_var = tk.StringVar(value="100")
        lattice_spin = ttk.Spinbox(lattice_frame, from_=10, to=1000,
                                  textvariable=self.lattice_size_var,
                                  width=20, font=('Segoe UI', 10))
        lattice_spin.pack(anchor='w', pady=(5, 0))
        
        # Iterations
        iter_frame = tk.Frame(input_frame, bg='#2d2d2d')
        iter_frame.pack(fill=tk.X, pady=10)
        ttk.Label(iter_frame, text="Recursive Refinement Iterations:",
                 style='Info.TLabel').pack(anchor='w')
        
        self.iterations_var = tk.StringVar(value="10")
        iter_scale = tk.Scale(iter_frame, from_=1, to=1000,
                             orient=tk.HORIZONTAL,
                             variable=self.iterations_var,
                             bg='#2d2d2d', fg='#ffffff',
                             troughcolor='#1e1e1e',
                             activebackground='#00ff88',
                             font=('Segoe UI', 9),
                             length=350)
        iter_scale.pack(fill=tk.X, pady=(5, 0))
        
        iter_label = ttk.Label(iter_frame, 
                              textvariable=self.iterations_var,
                              style='Info.TLabel')
        iter_label.pack(anchor='w')
        
        # Control buttons
        button_frame = tk.Frame(input_frame, bg='#2d2d2d')
        button_frame.pack(fill=tk.X, pady=20)
        
        self.start_btn = tk.Button(button_frame, text="▶ Start Factorization",
                                   command=self.start_factorization,
                                   bg='#00aa55', fg='#ffffff',
                                   font=('Segoe UI', 11, 'bold'),
                                   relief='flat', cursor='hand2',
                                   activebackground='#00cc66',
                                   padx=20, pady=12)
        self.start_btn.pack(fill=tk.X, pady=(0, 10))
        
        self.stop_btn = tk.Button(button_frame, text="⏹ Stop",
                                  command=self.stop_factorization,
                                  bg='#cc3333', fg='#ffffff',
                                  font=('Segoe UI', 10),
                                  relief='flat', cursor='hand2',
                                  activebackground='#dd4444',
                                  state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X)
        
        # Clear button
        clear_btn = tk.Button(button_frame, text="🗑 Clear Output",
                             command=self.clear_output,
                             bg='#555555', fg='#ffffff',
                             font=('Segoe UI', 9),
                             relief='flat', cursor='hand2',
                             activebackground='#666666')
        clear_btn.pack(fill=tk.X, pady=(10, 0))
        
        # Status section
        status_frame = tk.Frame(parent, bg='#2d2d2d')
        status_frame.pack(fill=tk.X, padx=15, pady=15)
        
        ttk.Label(status_frame, text="Status",
                 style='Heading.TLabel').pack(anchor='w', pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame,
                                     text="Ready",
                                     style='Info.TLabel',
                                     foreground='#00ff88')
        self.status_label.pack(anchor='w')
        
        # Progress bar
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate',
                                      length=350)
        self.progress.pack(fill=tk.X, pady=(10, 0))
    
    def create_output_panel(self, parent):
        """Create the output display panel."""
        # Output section
        output_frame = tk.Frame(parent, bg='#1e1e1e')
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(output_frame, text="Output & Results",
                 style='Heading.TLabel').pack(anchor='w', pady=(0, 10))
        
        # Output text area with larger font for better readability
        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            bg='#1e1e1e',
            fg='#00ff88',
            font=('Consolas', 10),
            insertbackground='#00ff88',
            relief='flat',
            wrap=tk.WORD,
            padx=15,
            pady=15,
            tabs=('1c', '2c', '3c', '4c')  # Tab stops for formatting
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags for colored output with enhanced styling
        self.output_text.tag_config('success', foreground='#00ff88', font=('Consolas', 10))
        self.output_text.tag_config('error', foreground='#ff4444', font=('Consolas', 10, 'bold'))
        self.output_text.tag_config('warning', foreground='#ffaa00', font=('Consolas', 10))
        self.output_text.tag_config('info', foreground='#88ccff', font=('Consolas', 10))
        self.output_text.tag_config('factor', foreground='#ffff00', font=('Consolas', 11, 'bold'))
    
    def load_from_file(self):
        """Load N from a file."""
        filename = filedialog.askopenfilename(
            title="Select file containing N",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    content = f.read().strip()
                    self.n_entry.delete('1.0', tk.END)
                    self.n_entry.insert('1.0', content)
                    self.log_output(f"Loaded N from {filename}\n", 'info')
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
    
    def log_output(self, text, tag=''):
        """Add text to output area."""
        self.output_text.insert(tk.END, text, tag)
        self.output_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_output(self):
        """Clear the output area."""
        self.output_text.delete('1.0', tk.END)
    
    def start_factorization(self):
        """Start the factorization process."""
        if self.is_running:
            return
        
        # Get input
        n_str = self.n_entry.get('1.0', tk.END).strip()
        if not n_str:
            messagebox.showerror("Error", "Please enter a number to factor.")
            return
        
        try:
            N = int(n_str)
        except ValueError:
            messagebox.showerror("Error", "Invalid number format.")
            return
        
        try:
            lattice_size = int(self.lattice_size_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid lattice size.")
            return
        
        try:
            iterations = int(self.iterations_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid iteration count.")
            return
        
        # Update UI
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Running...", foreground='#00ff88')
        self.progress.start()
        
        # Clear output
        self.clear_output()
        self.log_output("="*80 + "\n", 'info')
        self.log_output("🔷 GEOMETRIC LATTICE FACTORIZATION ENGINE 🔷\n", 'info')
        self.log_output("="*80 + "\n\n", 'info')
        self.log_output(f"📊 Configuration:\n", 'info')
        self.log_output(f"   Target N: {N}\n", 'info')
        self.log_output(f"   Bit length: {N.bit_length()} bits\n", 'info')
        self.log_output(f"   Initial lattice size: {lattice_size}×{lattice_size}×{lattice_size} = {lattice_size**3:,} points\n", 'info')
        self.log_output(f"   Recursive refinement iterations: {iterations}\n", 'info')
        self.log_output(f"   Expected zoom factor: 10^{iterations * 6}\n", 'info')
        self.log_output(f"\n{'='*80}\n", 'info')
        self.log_output("🚀 Starting factorization process...\n\n", 'info')
        
        # Start worker thread
        self.worker_thread = threading.Thread(
            target=self.factorization_worker,
            args=(N, lattice_size, iterations),
            daemon=True
        )
        self.worker_thread.start()
    
    def stop_factorization(self):
        """Stop the factorization process."""
        self.is_running = False
        self.status_label.config(text="Stopping...", foreground='#ffaa00')
        # Note: Actual stopping would require more complex thread management
    
    def factorization_worker(self, N, lattice_size, iterations):
        """Worker thread for factorization with verbose real-time output."""
        try:
            # Create a custom stdout that writes to queue in real-time
            class QueueWriter:
                def __init__(self, queue):
                    self.queue = queue
                    self.buffer = ""
                
                def write(self, text):
                    self.buffer += text
                    # Send complete lines to queue
                    while '\n' in self.buffer:
                        line, self.buffer = self.buffer.split('\n', 1)
                        self.queue.put(('output', line + '\n'))
                    # Also send any remaining text
                    if self.buffer and len(self.buffer) > 100:
                        self.queue.put(('output', self.buffer))
                        self.buffer = ""
                
                def flush(self):
                    if self.buffer:
                        self.queue.put(('output', self.buffer))
                        self.buffer = ""
            
            # Redirect stdout to queue writer
            old_stdout = sys.stdout
            queue_writer = QueueWriter(self.output_queue)
            sys.stdout = queue_writer
            
            # Send initial status
            self.output_queue.put(('status', 'Initializing factorization...'))
            
            # Call factorization with custom iterations
            self.output_queue.put(('status', f'Starting factorization with {iterations} iterations...'))
            self.output_queue.put(('output', f'[CONFIG] Using {iterations} recursive refinement iterations\n'))
            result = factor_with_lattice_compression(N, lattice_size=lattice_size, zoom_iterations=iterations)
            
            # Flush any remaining output
            queue_writer.flush()
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # Send completion status
            self.output_queue.put(('status', 'Factorization complete'))
            self.output_queue.put(('result', result))
            self.output_queue.put(('done', None))
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.output_queue.put(('error', error_msg))
            self.output_queue.put(('status', 'Error occurred'))
            self.output_queue.put(('done', None))
    
    def monitor_output(self):
        """Monitor output queue and update UI with verbose real-time updates."""
        try:
            while True:
                msg_type, data = self.output_queue.get_nowait()
                
                if msg_type == 'output':
                    # Auto-detect message types for color coding
                    if '✓' in data or 'FACTORS FOUND' in data or 'Found factor' in data:
                        self.log_output(data, 'success')
                    elif 'ERROR' in data or 'Error' in data or 'Traceback' in data:
                        self.log_output(data, 'error')
                    elif 'Warning' in data or 'WARNING' in data:
                        self.log_output(data, 'warning')
                    elif 'Stage' in data or 'STAGE' in data or 'Iteration' in data:
                        self.log_output(data, 'info')
                    elif '=' in data and len(data.strip()) > 10:
                        # Section headers
                        self.log_output(data, 'info')
                    else:
                        self.log_output(data)
                elif msg_type == 'status':
                    self.status_label.config(text=data, foreground='#00ff88')
                    self.log_output(f"[STATUS] {data}\n", 'info')
                elif msg_type == 'result':
                    self.display_results(data)
                elif msg_type == 'error':
                    self.log_output(f"\n{'='*80}\n", 'error')
                    self.log_output(f"ERROR: {data}\n", 'error')
                    self.log_output(f"{'='*80}\n", 'error')
                elif msg_type == 'done':
                    self.factorization_done()
                    
        except queue.Empty:
            pass
        
        # Schedule next check (more frequent for real-time feel)
        self.root.after(50, self.monitor_output)
    
    def display_results(self, result):
        """Display factorization results with verbose details."""
        self.log_output("\n" + "="*80 + "\n", 'success')
        self.log_output("📈 FACTORIZATION RESULTS\n", 'success')
        self.log_output("="*80 + "\n\n", 'success')
        
        if result and 'factors' in result:
            factors = result['factors']
            if factors:
                self.log_output("✅ FACTORS FOUND:\n\n", 'success')
                for i, factor_pair in enumerate(factors, 1):
                    p, q = factor_pair
                    verified = p * q == result.get('N', 0)
                    self.log_output(f"  Factor Pair #{i}:\n", 'info')
                    self.log_output(f"    p = {p}\n", 'factor')
                    self.log_output(f"    q = {q}\n", 'factor')
                    self.log_output(f"    p × q = {p * q}\n", 'factor')
                    self.log_output(f"    Verification: {'✓ CORRECT' if verified else '✗ INCORRECT'}\n\n", 
                                  'success' if verified else 'error')
                
                # Display compression metrics if available
                if 'compression_metrics' in result:
                    metrics = result['compression_metrics']
                    self.log_output("\n" + "="*80 + "\n", 'info')
                    self.log_output("📊 COMPRESSION METRICS\n", 'info')
                    self.log_output("="*80 + "\n", 'info')
                    if 'volume_reduction' in metrics:
                        self.log_output(f"   Volume reduction: {metrics.get('volume_reduction', 0):.2f}%\n", 'info')
                    if 'surface_reduction' in metrics:
                        self.log_output(f"   Surface area reduction: {metrics.get('surface_reduction', 0):.2f}%\n", 'info')
                    if 'span_reduction' in metrics:
                        self.log_output(f"   Span reduction: {metrics.get('span_reduction', 0):.2f}%\n", 'info')
                    if 'unique_points' in metrics and 'total_points' in metrics:
                        self.log_output(f"   Points collapsed: {metrics['unique_points']} / {metrics['total_points']}\n", 'info')
            else:
                self.log_output("⚠️  No factors found.\n", 'warning')
                self.log_output("   This may indicate N is prime, or factors require different encoding.\n", 'warning')
        
        self.log_output("\n" + "="*80 + "\n", 'info')
    
    def factorization_done(self):
        """Called when factorization completes."""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="✓ Complete", foreground='#00ff88')
        self.progress.stop()
        self.log_output("\n✅ Factorization process completed!\n", 'success')


def main():
    root = tk.Tk()
    app = FactorizationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

