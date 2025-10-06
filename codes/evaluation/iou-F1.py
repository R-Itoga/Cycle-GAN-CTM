import cv2
import numpy as np
import pandas as pd
import os
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
import tkinter as tk
from tkinter import filedialog

def evaluate_images(ground_truth_folder, prediction_folders, output_csv_prefix, thresholds):


    for threshold_min, threshold_max in thresholds:
        for pred_folder in prediction_folders:
            results = []
            for filename in os.listdir(ground_truth_folder):
                if filename.endswith('.jpg') or filename.endswith('.jpeg'):
                    ground_truth_path = os.path.join(ground_truth_folder, filename)
                    prediction_path = os.path.join(pred_folder, filename)

                    if not os.path.exists(prediction_path):
                        print(f"warning: {prediction_path} was not found. skipped")
                        continue

                    
                    img_gt = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
                    img_pred = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)

                    
                    mask_gt = ((img_gt >= threshold_min) & (img_gt <= threshold_max)).astype(np.uint8)
                    mask_pred = ((img_pred >= threshold_min) & (img_pred <= threshold_max)).astype(np.uint8)

                    
                    
                    iou = jaccard_score(mask_gt.flatten(), mask_pred.flatten())
                    precision = precision_score(mask_gt.flatten(), mask_pred.flatten())
                    recall = recall_score(mask_gt.flatten(), mask_pred.flatten())
                    f1 = f1_score(mask_gt.flatten(), mask_pred.flatten())

                    results.append([filename, iou, precision, recall, f1])
            
            
            if results:
                df = pd.DataFrame(results, columns=['filename', 'IoU', 'Precision', 'Recall', 'F1-score'])
                output_csv_path = os.path.join(pred_folder, f'{output_csv_prefix}_{threshold_min}_{threshold_max}.csv')
                df.to_csv(output_csv_path, index=False)
                print(f"results output to {output_csv_path} ")


def select_folders():

    root = tk.Tk()
    root.withdraw()

    print("select ground truth")
    ground_truth_folder = filedialog.askdirectory(title="select ground truth")
    if not ground_truth_folder:
        print("none selected")
        return None, None
    
    prediction_folders = []
    while True:
        print("select prediction folder. if cancel, select none.")
        folder = filedialog.askdirectory(title="select prediction folder")
        if not folder:
            break
        prediction_folders.append(folder)

    if not prediction_folders:
        print("none selected")
        return None, None
    
    return ground_truth_folder, prediction_folders


if __name__ == "__main__":
    ground_truth_folder, prediction_folders = select_folders()
    if ground_truth_folder and prediction_folders:
        output_csv_prefix = 'results'
        thresholds = [(0, 20), (140, 160), (230, 255)]
        

        evaluate_images(ground_truth_folder, prediction_folders, output_csv_prefix, thresholds)
