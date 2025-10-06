import os
import pydicom
import cv2

def embed_dicom_tags_into_jpeg(jpeg_folder, dicom_folder, output_folder):
    """
    JPEGファイルにDICOMデータのタグ情報を埋め込み、新しいDICOMファイルを作成します。
    白黒画像のJPEGからモノクロのDICOMに変換し、ウィンドウレベルと幅を設定します。

    Args:
        jpeg_folder: JPEGファイルが入っているフォルダのパス。
        dicom_folder: DICOMファイルが入っているフォルダのパス。
        output_folder: 新しいDICOMファイルを出力するフォルダのパス。
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(jpeg_folder):
        if filename.lower().endswith((".jpg", ".jpeg")):
            base_name, _ = os.path.splitext(filename)
            jpeg_path = os.path.join(jpeg_folder, filename)
            dicom_path = os.path.join(dicom_folder, base_name + ".dcm")

            if os.path.exists(dicom_path):
                try:
                    # DICOMファイルを読み込む
                    ds = pydicom.dcmread(dicom_path)

                    # JPEG画像をグレースケールで読み込む
                    img = cv2.imread(jpeg_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Error: Could not read JPEG file: {jpeg_path}")
                        continue

                    # --- メタデータの更新 ---

                    # 1. Pixel DataをJPEG画像のピクセルで置き換え
                    ds.PixelData = img.tobytes()
                    ds.PhotometricInterpretation = "MONOCHROME2"

                    # 2. 画像のサイズを更新 (img.shapeは (Rows, Columns) の順)
                    ds.Rows, ds.Columns = img.shape

                    # 3. 8bit画像用の設定を追加
                    ds.BitsAllocated = 8
                    ds.BitsStored = 8
                    ds.HighBit = 7
                    ds.PixelRepresentation = 0  # 0: 符号なし整数

                    # 4. ウィンドウレベルと幅を設定 (ご要望の箇所)
                    ds.WindowCenter = 400
                    ds.WindowWidth = 1000

                    # 5. Transfer Syntax UIDを非圧縮形式に設定
                    # (元のDICOMが圧縮形式だった場合に備える)
                    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

                    # 新しいDICOMファイル名を生成
                    output_path = os.path.join(output_folder, base_name + "_embedded.dcm")

                    # DICOMファイルを保存
                    ds.save_as(output_path)
                    print(f"Successfully created: {output_path}")

                except Exception as e:
                    print(f"Error processing {filename}: {e}")
            else:
                print(f"DICOM file not found for {filename} in {dicom_folder}")


# --- 使用例 ---
# 以下のパスを実際の環境に合わせて変更してください
jpeg_folder = "jpeg"
dicom_folder = "dicom"
output_folder = "modified_dicom"

embed_dicom_tags_into_jpeg(jpeg_folder, dicom_folder, output_folder)