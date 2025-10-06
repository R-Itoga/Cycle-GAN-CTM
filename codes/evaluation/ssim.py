import cv2
import os
from skimage.metrics import structural_similarity as ssim
import tkinter as tk
from tkinter import filedialog

def compare_images_and_save_csv(folder_a, folder_b, output_csv="ssim_results.csv"):
    """
    2つのフォルダ内の同名画像をSSIMで比較し、結果をCSVファイルに保存する。

    Args:
        folder_a (str): 比較元画像が格納されたフォルダのパス
        folder_b (str): 比較先画像が格納されたフォルダのパス
        output_csv (str): 出力するCSVファイルのパス
    """
    import csv
    
    # CSVファイルを書き込みモードで開く
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # ヘッダーを書き込む
        writer.writerow(['filename', 'ssim_score'])
        
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

                    # グレースケールに変換
                    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
                    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

                    # SSIMスコアを計算
                    score = ssim(gray_a, gray_b)
                    
                    # CSVに結果を書き込む
                    writer.writerow([filename, f"{score:.4f}"])
                    print(f"画像 {filename} のSSIMスコア: {score:.4f}")

                except Exception as e:
                    print(f"エラー: {filename} の処理中にエラーが発生しました。 {e}")
                    writer.writerow([filename, "エラー"])

    print(f"比較結果を {output_csv} に保存しました。")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # メインウィンドウを表示しない

    print("比較元画像フォルダを選択してください。")
    folder_a = filedialog.askdirectory(title="比較元画像フォルダを選択")
    print("比較先画像フォルダを選択してください。")
    folder_b = filedialog.askdirectory(title="比較先画像フォルダを選択")

    if folder_a and folder_b:
        compare_images_and_save_csv(folder_a, folder_b)
    else:
        print("フォルダが選択されませんでした。")
