import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import threading
import os
import platform # For opening directories

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("RT-DETR Inference GUI")

        # Workspace path (assuming gui.py is in the project root)
        self.workspace_path = os.getcwd()
        self.input_dir_host = os.path.join(self.workspace_path, "input")
        self.output_dir_host = os.path.join(self.workspace_path, "output")

        # Ensure input/output directories exist on host
        os.makedirs(self.input_dir_host, exist_ok=True)
        os.makedirs(self.output_dir_host, exist_ok=True)

        # --- UI Elements ---
        # Frame for input/output management
        io_frame = ttk.LabelFrame(root, text="Input/Output Management")
        io_frame.pack(padx=10, pady=10, fill="x")

        ttk.Label(io_frame, text=f"Place your media files in: {self.input_dir_host}").pack(pady=5)
        ttk.Button(io_frame, text="Open Input Directory", command=self.open_input_dir).pack(pady=5)
        ttk.Label(io_frame, text=f"Output will be saved to: {self.output_dir_host}").pack(pady=5)
        ttk.Button(io_frame, text="Open Output Directory", command=self.open_output_dir).pack(pady=5)

        # Frame for parameters
        param_frame = ttk.LabelFrame(root, text="Inference Parameters")
        param_frame.pack(padx=10, pady=5, fill="x")

        ttk.Label(param_frame, text="Device:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.device_var = tk.StringVar(value="cuda:0") # Default to cuda:0 as per run_inference.sh
        device_options = ["cuda:0", "cpu"]
        device_menu = ttk.OptionMenu(param_frame, self.device_var, self.device_var.get(), *device_options)
        device_menu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(param_frame, text="Threshold:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.threshold_var = tk.StringVar(value="0.5") # Default to 0.5 as per run_inference.sh
        threshold_entry = ttk.Entry(param_frame, textvariable=self.threshold_var)
        threshold_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        param_frame.columnconfigure(1, weight=1)


        # Frame for controls and progress
        control_frame = ttk.Frame(root)
        control_frame.pack(padx=10, pady=10, fill="x")

        self.run_button = ttk.Button(control_frame, text="Start Inference", command=self.start_inference_thread)
        self.run_button.pack(pady=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, orient="horizontal", length=300, mode="determinate", variable=self.progress_var)
        self.progress_bar.pack(pady=5)
        self.progress_label = ttk.Label(control_frame, text="0%")
        self.progress_label.pack(pady=2)


        # Frame for logs
        log_frame = ttk.LabelFrame(root, text="Logs")
        log_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.log_text = tk.Text(log_frame, height=15, width=80, wrap=tk.WORD, bg="black", fg="lightgreen", font=("Consolas", 10))
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        log_scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=log_scrollbar.set)

        self.files_to_process_total = 0
        self.files_processed_count = 0

    def open_dir(self, path):
        try:
            if platform.system() == "Windows":
                os.startfile(os.path.normpath(path))
            elif platform.system() == "Darwin": # macOS
                subprocess.Popen(["open", os.path.normpath(path)])
            else: # Linux
                subprocess.Popen(["xdg-open", os.path.normpath(path)])
        except Exception as e:
            self.log_message_gui(f"Error opening directory {path}: {e}")
            messagebox.showerror("Error", f"Could not open directory: {path}\n{e}")

    def open_input_dir(self):
        self.open_dir(self.input_dir_host)

    def open_output_dir(self):
        self.open_dir(self.output_dir_host)

    def log_message_gui(self, message):
        # This function must be called from the main thread or scheduled with root.after
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        # self.root.update_idletasks() # Not always needed if called via root.after

    def count_processable_files(self, directory):
        count = 0
        # From README and common types for RT-DETR
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', # images
                              '.mp4', '.avi', '.mov', '.mkv', '.webm') # videos
        try:
            for item in os.listdir(directory):
                if os.path.isfile(os.path.join(directory, item)) and item.lower().endswith(valid_extensions):
                    count += 1
        except Exception as e:
            self.root.after(0, self.log_message_gui, f"Error counting files in {directory}: {e}")
        return count

    def start_inference_thread(self):
        self.run_button.config(state=tk.DISABLED)
        self.log_text.delete(1.0, tk.END)
        self.progress_var.set(0)
        self.progress_label.config(text="0%")
        self.files_processed_count = 0
        
        self.files_to_process_total = self.count_processable_files(self.input_dir_host)
        
        if self.files_to_process_total == 0:
            self.log_message_gui("No processable files found in the input directory. Script will still run.")
            self.progress_bar.config(mode="indeterminate")
            self.progress_bar.start(10) # Indeterminate animation
        else:
            self.log_message_gui(f"Found {self.files_to_process_total} processable file(s) in input directory.")
            self.progress_bar.config(mode="determinate", maximum=100) # Set for percentage

        self.log_message_gui("Starting inference...")
        
        thread = threading.Thread(target=self.run_inference_command)
        thread.daemon = True 
        thread.start()

    def run_inference_command(self):
        device = self.device_var.get()
        threshold_str = self.threshold_var.get()

        try:
            threshold_val = float(threshold_str)
            if not (0 <= threshold_val <= 1):
                self.root.after(0, self.log_message_gui, "Error: Threshold must be between 0.0 and 1.0.")
                self.root.after(0, self.on_inference_complete, False, "Invalid threshold value.")
                return
        except ValueError:
            self.root.after(0, self.log_message_gui, "Error: Threshold must be a valid number.")
            self.root.after(0, self.on_inference_complete, False, "Invalid threshold format.")
            return

        command = [
            "docker-compose", "exec", "tensorrt-container",
            "bash", "./run_inference.sh", device, threshold_str
        ]
        self.root.after(0, self.log_message_gui, f"Executing: {' '.join(command)}")

        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                       text=True, bufsize=1, universal_newlines=True, cwd=self.workspace_path,
                                       creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0)
            
            for line in iter(process.stdout.readline, ''):
                self.root.after(0, self.log_message_gui, line.strip())
                
                # --- Attempt to update progress based on output ---
                # This part is speculative and depends on the output of rtdetrv2_torch.py
                # If your script logs filenames or specific phrases upon processing each file,
                # this logic can be made more robust.
                # Example: "Processing file: image.jpg" or "Saved output for video.mp4"
                # For now, we'll use a very generic check or rely on the final "Inference complete".
                
                # A simple heuristic: if "Inference complete" is not yet seen,
                # and we see a line that might indicate one file is done.
                # This is VERY basic and might need tuning based on actual script output.
                if self.files_to_process_total > 0 and "Inference complete" not in line :
                    # Check for lines that might indicate a file has been processed
                    # This is highly dependent on your script's output.
                    # You might need to adjust these keywords.
                    processed_keywords = ["processing", "detecting", "saving result for", ".jpg", ".png", ".mp4"] # Add more if needed
                    if any(kw in line.lower() for kw in processed_keywords):
                        # This is a rough guess, it might increment multiple times for one file.
                        # A better approach is if the script explicitly states "Processed file X of Y".
                        # For now, we increment and cap.
                        if self.files_processed_count < self.files_to_process_total:
                           # self.files_processed_count +=1 # This might be too aggressive
                           # Let's assume the script prints "Inference complete" at the very end.
                           # The progress bar will jump to 100% then.
                           # For intermediate, we can just keep it moving slowly or based on some general output activity.
                           # A more accurate way would be to parse specific messages from the script.
                           # For simplicity, we will mainly rely on the "Inference complete" message for 100%.
                           # If files_to_process_total > 0, let's make small increments to show activity.
                           current_progress = self.progress_var.get()
                           if current_progress < 90 : # Don't let this heuristic fill it completely
                                self.progress_var.set(current_progress + (50.0 / self.files_to_process_total if self.files_to_process_total > 0 else 5))


                if "Inference complete" in line:
                    if self.files_to_process_total > 0 or self.progress_bar.cget("mode") == "indeterminate":
                        self.root.after(0, lambda: self.progress_var.set(100))
                        self.root.after(0, lambda: self.progress_label.config(text="100%"))
            
            process.wait()
            
            if process.returncode == 0:
                # Final update to 100% if not already set by "Inference complete" log line.
                self.root.after(0, lambda: self.progress_var.set(100))
                self.root.after(0, lambda: self.progress_label.config(text="100%"))
                self.root.after(0, self.on_inference_complete, True, "Inference process completed successfully.")
            else:
                error_msg = f"Inference process failed with exit code {process.returncode}."
                self.root.after(0, self.log_message_gui, error_msg)
                self.root.after(0, self.on_inference_complete, False, error_msg)

        except FileNotFoundError:
            msg = "Error: docker-compose command not found. Is Docker Desktop running and 'docker-compose' in system PATH?"
            self.root.after(0, self.log_message_gui, msg)
            self.root.after(0, self.on_inference_complete, False, msg)
        except Exception as e:
            error_msg = f"An error occurred: {e}"
            self.root.after(0, self.log_message_gui, error_msg)
            self.root.after(0, self.on_inference_complete, False, error_msg)
        finally:
            # This ensures cleanup if thread ends unexpectedly too
            if self.progress_bar.cget("mode") == "indeterminate":
                 self.root.after(0, self.progress_bar.stop)


    def on_inference_complete(self, success, message):
        self.run_button.config(state=tk.NORMAL)
        if self.progress_bar.cget("mode") == "indeterminate":
            self.progress_bar.stop()
        
        if success:
            # Ensure it's 100% on success, even if indeterminate was used due to 0 files
            self.progress_var.set(100)
            self.progress_label.config(text="100%")
            messagebox.showinfo("Success", message)
            self.open_output_dir() # Optionally open output dir on success
        else:
            # For failure, if it was determinate, don't force to 0 unless it makes sense
            # If it was indeterminate, it's already stopped.
            # Current state of progress_var is fine.
            self.progress_label.config(text="Error")
            messagebox.showerror("Error", f"Inference failed. Check logs for details.\nDetails: {message}")

if __name__ == "__main__":
    root = tk.Tk()
    # Simple styling
    style = ttk.Style()
    # print(style.theme_names()) # To see available themes
    # print(style.theme_use()) # To see current theme
    try:
        # Attempt to use a more modern theme if available
        if "clam" in style.theme_names():
            style.theme_use("clam")
        elif "vista" in style.theme_names() and platform.system() == "Windows":
            style.theme_use("vista")
    except tk.TclError:
        pass # Default theme is fine

    app = App(root)
    root.mainloop()

