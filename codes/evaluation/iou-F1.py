import cv2
import numpy as np
import pandas as pd
import os
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
import tkinter as tk
from tkinter import filedialog

def evaluate_images(ground_truth_folder, prediction_folders, output_csv_prefix, thresholds):
    """
    正解画像フォルダと複数の予測画像フォルダを比較し、評価指標を計算してCSVファイルに出力します。

    Args:
        ground_truth_folder (str): 正解画像フォルダのパス
        prediction_folders (list): 予測画像フォルダのパスのリスト
        output_csv_prefix (str): 出力CSVファイルのプレフィックス
        thresholds (list): 評価する階調幅のリスト (例: [(0, 20), (140, 160), (240, 255)])
    """

    for threshold_min, threshold_max in thresholds:
        for pred_folder in prediction_folders:
            results = []
            for filename in os.listdir(ground_truth_folder):
                if filename.endswith('.jpg') or filename.endswith('.jpeg'):
                    ground_truth_path = os.path.join(ground_truth_folder, filename)
                    prediction_path = os.path.join(pred_folder, filename)

                    if not os.path.exists(prediction_path):
                        print(f"警告: {prediction_path} が見つかりませんでした。スキップします。")
                        continue

                    # 画像を読み込み、グレースケールに変換
                    img_gt = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
                    img_pred = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)

                    # 指定した階調幅のマスクを作成
                    mask_gt = ((img_gt >= threshold_min) & (img_gt <= threshold_max)).astype(np.uint8)
                    mask_pred = ((img_pred >= threshold_min) & (img_pred <= threshold_max)).astype(np.uint8)

                    # 評価指標を計算
                    # ここで、y_trueに正解画像（mask_gt）を、y_predに予測画像（mask_pred）を渡しています
                    iou = jaccard_score(mask_gt.flatten(), mask_pred.flatten())
                    precision = precision_score(mask_gt.flatten(), mask_pred.flatten())
                    recall = recall_score(mask_gt.flatten(), mask_pred.flatten())
                    f1 = f1_score(mask_gt.flatten(), mask_pred.flatten())

                    results.append([filename, iou, precision, recall, f1])
            
            # 結果をDataFrameに変換し、各予測フォルダ内にCSVファイルとして保存
            if results:
                df = pd.DataFrame(results, columns=['filename', 'IoU', 'Precision', 'Recall', 'F1-score'])
                output_csv_path = os.path.join(pred_folder, f'{output_csv_prefix}_{threshold_min}_{threshold_max}.csv')
                df.to_csv(output_csv_path, index=False)
                print(f"結果を {output_csv_path} に出力しました。")


def select_folders():
    """
    tkinterを使用してフォルダ選択ダイアログを表示し、正解フォルダと複数の予測フォルダを選択します。
    """
    root = tk.Tk()
    root.withdraw()

    print("正解画像フォルダを選択してください。")
    ground_truth_folder = filedialog.askdirectory(title="正解画像フォルダを選択")
    if not ground_truth_folder:
        print("フォルダが選択されませんでした。")
        return None, None
    
    prediction_folders = []
    while True:
        print("比較画像フォルダを1つ選択してください。（終了する場合はキャンセルを押してください）")
        folder = filedialog.askdirectory(title="比較画像フォルダを選択")
        if not folder:
            break
        prediction_folders.append(folder)

    if not prediction_folders:
        print("比較画像フォルダが選択されませんでした。")
        return None, None
    
    return ground_truth_folder, prediction_folders


if __name__ == "__main__":
    ground_truth_folder, prediction_folders = select_folders()
    if ground_truth_folder and prediction_folders:
        output_csv_prefix = 'results'
        thresholds = [(0, 20), (140, 160), (230, 255)]
        
        evaluate_images(ground_truth_folder, prediction_folders, output_csv_prefix, thresholds)