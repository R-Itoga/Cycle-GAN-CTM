import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
import glob

def find_transform_from_reference(image_a_path, image_b_path):
    """
    画像Aと画像Bを比較し、画像Bが切り出された元の位置とサイズを特定する。
    Args:
        image_a_path (str): 元画像Aのパス
        image_b_path (str): 参照用画像Bのパス
    Returns:
        dict: 成功した場合、変換情報(x, y, w, h, a_height, a_width)の辞書。失敗した場合はNone。
    """
    print(f"基準を計算中: '{os.path.basename(image_a_path)}'と'{os.path.basename(image_b_path)}'をマッチングします...")
    
    img_a = cv2.imread(image_a_path)
    img_b = cv2.imread(image_b_path)

    if img_a is None or img_b is None:
        print(f"❌ エラー: 画像ファイル({image_a_path} or {image_b_path})を読み込めませんでした。")
        return None

    # ROIを設定してオーバーレイを回避
    roi_y_start, roi_y_end = 0, img_a.shape[0] - 30 # 下部のテキストを避ける
    roi_x_start, roi_x_end = 80, img_a.shape[1] # 左側のバーを避ける
    roi_a = img_a[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    
    orb = cv2.ORB_create(nfeatures=2000)
    kp_a, des_a = orb.detectAndCompute(roi_a, None)
    kp_b, des_b = orb.detectAndCompute(img_b, None)
    
    if des_a is None or des_b is None:
        print("❌ エラー: 特徴点を検出できませんでした。")
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_b, des_a)
    matches = sorted(matches, key=lambda x: x.distance)
    
    if len(matches) < 10:
        print("❌ エラー: 十分な数のマッチングが見つかりませんでした。")
        return None

    good_matches = matches[:50]
    
    # マッチングの様子を実行ファイルと同じ場所に保存
    match_img_path = 'matches_visualization.jpg'
    img_matches = cv2.drawMatches(img_b, kp_b, roi_a, kp_a, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(match_img_path, img_matches)
    print(f"💡 基準計算時のマッチングの様子を '{match_img_path}' に保存しました。")

    src_pts = np.float32([kp_b[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_a[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if M is None:
        print("❌ エラー: 幾何学的な変換を計算できませんでした。")
        return None

    h_b, w_b = img_b.shape[:2]
    pts_b = np.float32([[0, 0], [0, h_b - 1], [w_b - 1, h_b - 1], [w_b - 1, 0]]).reshape(-1, 1, 2)
    dst_corners_in_roi = cv2.perspectiveTransform(pts_b, M)
    dst_corners_in_a = dst_corners_in_roi + [roi_x_start, roi_y_start]
    
    points_for_rect = np.int32(dst_corners_in_a)
    x, y, w, h = cv2.boundingRect(points_for_rect)
    
    h_a, w_a = img_a.shape[:2]

    return {'x': x, 'y': y, 'w': w, 'h': h, 'a_height': h_a, 'a_width': w_a}

def batch_process_folder(folder_path, transform_params):
    """
    指定されたフォルダ内の全画像を、与えられた変換情報に基づいて処理する。
    """
    if not transform_params:
        print("❌ エラー: 変換情報が無効なため、一括処理を実行できません。")
        return

    # 保存用フォルダを作成
    output_folder = os.path.join(folder_path, 'restored')
    os.makedirs(output_folder, exist_ok=True)
    print(f"📂 保存先フォルダ: '{output_folder}'")

    # 処理対象の画像ファイルを取得 (jpg, jpeg, png, bmp, tiff)
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
    files_to_process = []
    for ext in image_extensions:
        files_to_process.extend(glob.glob(os.path.join(folder_path, ext)))

    if not files_to_process:
        print(f"⚠️ フォルダ '{folder_path}' に処理対象の画像が見つかりませんでした。")
        return

    print(f"🏞️ {len(files_to_process)} 個の画像を処理します...")
    
    # 変換情報を展開
    ref_x, ref_y = transform_params['x'], transform_params['y']
    ref_w, ref_h = transform_params['w'], transform_params['h']
    canvas_h, canvas_w = transform_params['a_height'], transform_params['a_width']
    
    count = 0
    for file_path in files_to_process:
        img = cv2.imread(file_path)
        if img is None:
            print(f"  - 警告: '{os.path.basename(file_path)}' は読み込めませんでした。スキップします。")
            continue

        # 基準のサイズにリサイズ
        resized_img = cv2.resize(img, (ref_w, ref_h))

        # 黒い背景を作成
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # 背景の基準位置に画像を貼り付け
        canvas[ref_y:ref_y+ref_h, ref_x:ref_x+ref_w] = resized_img

        # 保存
        base_filename = os.path.basename(file_path)
        save_path = os.path.join(output_folder, base_filename)
        cv2.imwrite(save_path, canvas)
        count += 1
        print(f"  ({count}/{len(files_to_process)}) ✔️  '{base_filename}' の処理が完了しました。")

    print(f"\n✅ 一括処理が完了しました。{count}個のファイルを '{output_folder}' に保存しました。")


# ==============================================================================
# メインの実行部分
# ==============================================================================
if __name__ == '__main__':
    # Tkinterのウィンドウを準備（画面には表示しない）
    root = tk.Tk()
    root.withdraw()

    print("これよりファイル選択ダイアログが3回開きます。順番に選択してください。\n")
    
    # 1. 画像Aを選択
    image_a_path = filedialog.askopenfilename(title="ステップ1/3: 元となる全体画像 (画像A) を選択してください")
    if not image_a_path:
        print("キャンセルされました。処理を終了します。")
        exit()

    # 2. 画像Bを選択
    image_b_path = filedialog.askopenfilename(title="ステップ2/3: 基準となる切り抜き画像 (画像B) を選択してください")
    if not image_b_path:
        print("キャンセルされました。処理を終了します。")
        exit()

    # 3. フォルダDを選択
    folder_d_path = filedialog.askdirectory(title="ステップ3/3: 一括処理したい画像が入ったフォルダ (フォルダD) を選択してください")
    if not folder_d_path:
        print("キャンセルされました。処理を終了します。")
        exit()

    print("-" * 50)
    
    # 基準となる変換情報を計算
    transform = find_transform_from_reference(image_a_path, image_b_path)
    
    print("-" * 50)

    # 一括処理を実行
    if transform:
        batch_process_folder(folder_d_path, transform)
    else:
        print("基準の計算に失敗したため、一括処理は行いませんでした。")