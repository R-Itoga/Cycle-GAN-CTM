import os
import pydicom
import cv2

def embed_dicom_tags_into_jpeg(jpeg_folder, dicom_folder, output_folder):


    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(jpeg_folder):
        if filename.lower().endswith((".jpg", ".jpeg")):
            base_name, _ = os.path.splitext(filename)
            jpeg_path = os.path.join(jpeg_folder, filename)
            dicom_path = os.path.join(dicom_folder, base_name + ".dcm")

            if os.path.exists(dicom_path):
                try:
                    
                    ds = pydicom.dcmread(dicom_path)

                    
                    img = cv2.imread(jpeg_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Error: Could not read JPEG file: {jpeg_path}")
                        continue

                   

                    
                    ds.PixelData = img.tobytes()
                    ds.PhotometricInterpretation = "MONOCHROME2"

                    
                    ds.Rows, ds.Columns = img.shape

                    
                    ds.BitsAllocated = 8
                    ds.BitsStored = 8
                    ds.HighBit = 7
                    ds.PixelRepresentation = 0  

                    
                    ds.WindowCenter = 400
                    ds.WindowWidth = 1000

                   
                    
                    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

                    
                    output_path = os.path.join(output_folder, base_name + "_embedded.dcm")

                    
                    ds.save_as(output_path)
                    print(f"Successfully created: {output_path}")

                except Exception as e:
                    print(f"Error processing {filename}: {e}")
            else:
                print(f"DICOM file not found for {filename} in {dicom_folder}")



jpeg_folder = "jpeg"
dicom_folder = "dicom"
output_folder = "modified_dicom"


embed_dicom_tags_into_jpeg(jpeg_folder, dicom_folder, output_folder)
