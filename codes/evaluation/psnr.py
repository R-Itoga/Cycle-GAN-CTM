import cv2
import os
import numpy as np
import csv
import tkinter as tk
from tkinter import filedialog

def compare_images_with_psnr(folder_a, folder_b, output_csv="psnr_results.csv"):

    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        writer.writerow(['filename', 'PSNR'])
        
        
        files_a = os.listdir(folder_a)
        files_b = os.listdir(folder_b)

        
        for filename in files_a:
            if filename in files_b and filename.lower().endswith('.jpg'):
                image_a_path = os.path.join(folder_a, filename)
                image_b_path = os.path.join(folder_b, filename)

                try:
                    
                    image_a = cv2.imread(image_a_path)
                    image_b = cv2.imread(image_b_path)

                    
                    mse = np.mean((image_a - image_b) ** 2)
                    if mse == 0:
                        psnr = float('inf')
                    else:
                        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
                    
                    
                    writer.writerow([filename, f"{psnr:.2f}"])
                    print(f"画像 {filename} のPSNRスコア: {psnr:.2f}")

                except Exception as e:
                    print(f"error: {filename}  {e}")
                    writer.writerow([filename, "error"])

    print(f"saved to {output_csv} ")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw() 
    print("select ground truth folder")
    folder_a = filedialog.askdirectory(title="select ground truth folder")
    print("select prediction folder")
    folder_b = filedialog.askdirectory(title="select prediction folder")
    if folder_a and folder_b:
        compare_images_with_psnr(folder_a, folder_b)
    else:
        print("none selected")

