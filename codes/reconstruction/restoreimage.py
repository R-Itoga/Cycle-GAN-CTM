import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
import glob

def find_transform_from_reference(image_a_path, image_b_path):

    print(f"matching '{os.path.basename(image_a_path)}'and'{os.path.basename(image_b_path)}'...")
    
    img_a = cv2.imread(image_a_path)
    img_b = cv2.imread(image_b_path)

    if img_a is None or img_b is None:
        print(f" error: ({image_a_path} or {image_b_path}) not found")
        return None

    
    roi_y_start, roi_y_end = 0, img_a.shape[0] - 30 
    roi_x_start, roi_x_end = 80, img_a.shape[1] 
    roi_a = img_a[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    
    orb = cv2.ORB_create(nfeatures=2000)
    kp_a, des_a = orb.detectAndCompute(roi_a, None)
    kp_b, des_b = orb.detectAndCompute(img_b, None)
    
    if des_a is None or des_b is None:
        print("error: could not match the images")
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_b, des_a)
    matches = sorted(matches, key=lambda x: x.distance)
    
    if len(matches) < 10:
        print("error: could not match the images")
        return None

    good_matches = matches[:50]
    
    
    match_img_path = 'matches_visualization.jpg'
    img_matches = cv2.drawMatches(img_b, kp_b, roi_a, kp_a, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(match_img_path, img_matches)
    print(f"💡matching image saved to '{match_img_path}'")

    src_pts = np.float32([kp_b[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_a[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if M is None:
        print("error")
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

    if not transform_params:
        print("error")
        return

    
    output_folder = os.path.join(folder_path, 'restored')
    os.makedirs(output_folder, exist_ok=True)
    print(f"output folder: '{output_folder}'")

    
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
    files_to_process = []
    for ext in image_extensions:
        files_to_process.extend(glob.glob(os.path.join(folder_path, ext)))

    if not files_to_process:
        print(f"no findings in '{folder_path}'")
        return

    print(f"🏞️ {len(files_to_process)} images found...")
    
    
    ref_x, ref_y = transform_params['x'], transform_params['y']
    ref_w, ref_h = transform_params['w'], transform_params['h']
    canvas_h, canvas_w = transform_params['a_height'], transform_params['a_width']
    
    count = 0
    for file_path in files_to_process:
        img = cv2.imread(file_path)
        if img is None:
            print(f"  - warning: '{os.path.basename(file_path)}' skipped")
            continue

        
        resized_img = cv2.resize(img, (ref_w, ref_h))

        
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        
        canvas[ref_y:ref_y+ref_h, ref_x:ref_x+ref_w] = resized_img

        
        base_filename = os.path.basename(file_path)
        save_path = os.path.join(output_folder, base_filename)
        cv2.imwrite(save_path, canvas)
        count += 1
        print(f"  ({count}/{len(files_to_process)}) ✔️  '{base_filename}' finished")

    print(f"\n✅ finished. {count}images in '{output_folder}'")



if __name__ == '__main__':
    
    root = tk.Tk()
    root.withdraw()

    print("select file or folder in order.\n")
    
    
    image_a_path = filedialog.askopenfilename(title="1/3: select base image (whole image)")
    if not image_a_path:
        print("canceled")
        exit()

    
    image_b_path = filedialog.askopenfilename(title="2/3: select croppped image (partial image)")
    if not image_b_path:
        print("キャンセルされました。処理を終了します。")
        exit()

    
    folder_d_path = filedialog.askdirectory(title="2/3: select working folder")
    if not folder_d_path:
        print("canceled")
        exit()

    print("-" * 50)
    
    
    transform = find_transform_from_reference(image_a_path, image_b_path)
    
    print("-" * 50)

    
    if transform:
        batch_process_folder(folder_d_path, transform)
    else:

        print("error")
