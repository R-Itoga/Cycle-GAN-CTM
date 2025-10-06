import cv2
import os
import numpy as np
import csv
import tkinter as tk
from tkinter import filedialog

def compare_images_with_psnr(folder_a, folder_b, output_csv="psnr_results.csv"):
    """
    2つのフォルダ内の同名画像をPSNRで比較し、結果をCSVファイルに保存する。

    Args:
        folder_a (str): 比較元画像が格納されたフォルダのパス
        folder_b (str): 比較先画像が格納されたフォルダのパス
        output_csv (str): 出力するCSVファイルのパス
    """
    # CSVファイルを書き込みモードで開く
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # ヘッダーを書き込む
        writer.writerow(['ファイル名', 'PSNRスコア'])
        
        # フォルダ内のファイル一覧を取得
        files_a = os.listdir(folder_a)
        files_b = os.listdir(folder_b)

        # 同名ファイルを探して処理
        for filename in files_a:
            if filename in files_b and filename.lower().endswith('.jpg'):
                image_a_path = os.path.join(folder_a, filename)
                image_b_path = os.path.join(folder_b, filename)

                try:
                    # 画像を読み込み
                    image_a = cv2.imread(image_a_path)
                    image_b = cv2.imread(image_b_path)

                    # PSNRスコアを計算
                    mse = np.mean((image_a - image_b) ** 2)
                    if mse == 0:
                        psnr = float('inf')
                    else:
                        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
                    
                    # CSVに結果を書き込む
                    writer.writerow([filename, f"{psnr:.2f}"])
                    print(f"画像 {filename} のPSNRスコア: {psnr:.2f}")

                except Exception as e:
                    print(f"エラー: {filename} の処理中にエラーが発生しました。 {e}")
                    writer.writerow([filename, "エラー"])

    print(f"比較結果を {output_csv} に保存しました。")

# 使用例：
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # メインウィンドウを表示しない
    print("比較元画像フォルダを選択してください。")
    folder_a = filedialog.askdirectory(title="比較元画像フォルダを選択")
    print("比較先画像フォルダを選択してください。")
    folder_b = filedialog.askdirectory(title="比較先画像フォルダを選択")
    if folder_a and folder_b:
        compare_images_with_psnr(folder_a, folder_b)
    else:
        print("フォルダが選択されませんでした。処理を中止します。")
