import cv2
import os
from skimage.metrics import structural_similarity as ssim
import tkinter as tk
from tkinter import filedialog

def compare_images_and_save_csv(folder_a, folder_b, output_csv="ssim_results.csv"):

    import csv
    
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
       
        writer.writerow(['filename', 'ssim_score'])
        
        
        files_a = os.listdir(folder_a)
        files_b = os.listdir(folder_b)

        
        for filename in files_a:
            if filename in files_b and filename.lower().endswith('.jpg'):
                image_a_path = os.path.join(folder_a, filename)
                image_b_path = os.path.join(folder_b, filename)

                try:
                    
                    image_a = cv2.imread(image_a_path)
                    image_b = cv2.imread(image_b_path)

                    
                    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
                    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

                    
                    score = ssim(gray_a, gray_b)
                    
                    
                    writer.writerow([filename, f"{score:.4f}"])
                    print(f" {filename} : {score:.4f}")

                except Exception as e:
                    print(f"error: {filename}  {e}")
                    writer.writerow([filename, "error"])

    print(f"saved to {output_csv}")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  

    print("select ground truth folder")
    folder_a = filedialog.askdirectory(title="select ground truth folder")
    print("select prediction folder")
    folder_b = filedialog.askdirectory(title="select prediction folder")

    if folder_a and folder_b:
        compare_images_and_save_csv(folder_a, folder_b)
    else:
        print("none selected")

