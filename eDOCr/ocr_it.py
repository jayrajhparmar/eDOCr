import os
import argparse
from pdf2image import convert_from_path
import numpy as np
import keras_ocr
from eDOCr import tools
import cv2
import string
from skimage import io
##############################


##############################
parser=argparse.ArgumentParser(description='Pre-process the engineering drawing, it returns the information from the title block')
parser.add_argument('file_path', help='specify the file path to the drawing. Supported formats: .pdf, .png, .jpg')
parser.add_argument('--dest_folder', help='specify the destination folder')
parser.add_argument('--water', action='store_true', help='Does the drawing have watermark you want to remove?')                  
parser.add_argument('--cluster', help='Set a custom threshold distance (in px.) for grouping detections')   
args = parser.parse_args()

if os.path.exists(args.file_path):
    pass
else:
    raise NotADirectoryError (args.file_path)

if args.dest_folder:
    if os.path.exists(args.dest_folder):
        os.makedirs(os.path.join(args.dest_folder),'Results')
        dest_DIR=os.path.join(args.dest_folder),'Results'
    else:
        raise NotADirectoryError (args.dest_folder)
else:
    os.makedirs('Results', exist_ok=True)
    dest_DIR='Results'
if args.cluster:
    cluster_t=int(args.cluster)
else:
    cluster_t=20
###############################################################
GDT_symbols='⏤⏥○⌭⌒⌓⏊∠⫽⌯⌖◎↗⌰'
FCF_symbols='ⒺⒻⓁⓂⓅⓈⓉⓊ'
Extra='(),.+-±:/°"⌀'

alphabet_dimensions=string.digits + 'AaBCDRGHhMmnx'+ Extra
alphabet_infoblock=string.digits+string.ascii_letters+',.:-/'
alphabet_gdts=string.digits+',.⌀ABCD'+GDT_symbols #+FCF_symbols


model_infoblock=keras_ocr.tools.download_and_verify(
                        url="https://github.com/javvi51/eDOCr/releases/download/v1.0.0/recognizer_infoblock.h5",
                        filename="recognizer_infoblock.h5",
                        sha256="e0a317e07ce75235f67460713cf1b559e02ae2282303eec4a1f76ef211fcb8e8",
                    )

model_dimensions=keras_ocr.tools.download_and_verify(
                        url="https://github.com/javvi51/eDOCr/releases/download/v1.0.0/recognizer_dimensions.h5",
                        filename="recognizer_dimensions.h5",
                        sha256="a1c27296b1757234a90780ccc831762638b9e66faf69171f5520817130e05b8f",
                    )

model_gdts=keras_ocr.tools.download_and_verify(
                        url="https://github.com/javvi51/eDOCr/releases/download/v1.0.0/recognizer_gdts.h5",
                        filename="recognizer_gdts.h5",
                        sha256="58acf6292a43ff90a344111729fc70cf35f0c3ca4dfd622016456c0b29ef2a46",
                    )

color_palette={'infoblock':(180,220,250),'gdts':(94,204,243),'dimensions':(93,206,175),'frame':(167,234,82),'flag':(241,65,36)}
filename=os.path.splitext(os.path.basename(args.file_path))[0]
###############################################################
#pdf to image
if os.path.splitext(args.file_path)[1]=='.pdf':
    images = convert_from_path(args.file_path, dpi=300)
else:
    images=[cv2.imread(args.file_path)]

index = 0
#for pages in pdf, usually 1
for img in images:
    index += 1
    img = np.array(img)
    #watermark?
    if args.water==True:
        img = tools.watermark.handle(img)

    class_list, img_boxes=tools.box_tree.findrect(img)
    boxes_infoblock,gdt_boxes,cl_frame,process_img=tools.img_process.process_rect(class_list,img)
    io.imsave(os.path.join(dest_DIR, filename+'_'+str(index)+'_process.jpg'),process_img)

    infoblock_dict=tools.pipeline_infoblock.read_infoblocks(boxes_infoblock,img,alphabet_infoblock,model_infoblock)
    gdt_dict=tools.pipeline_gdts.read_gdtbox1(gdt_boxes,alphabet_gdts,model_gdts,alphabet_dimensions,model_dimensions )
    
    process_img=os.path.join(dest_DIR, filename+'_'+str(index)+'_process.jpg')

    dimension_dict=tools.pipeline_dimensions.read_dimensions(process_img,alphabet_dimensions,model_dimensions,cluster_t)
    mask_img=tools.output.mask_the_drawing(img, infoblock_dict, gdt_dict, dimension_dict,cl_frame,color_palette)

    #Record the results
    io.imsave(os.path.join(dest_DIR, filename+'_'+str(index)+'_boxes.jpg'),img_boxes)
    io.imsave(os.path.join(dest_DIR, filename+'_'+str(index)+ '_mask.jpg'),mask_img)

    tools.output.record_data(dest_DIR,filename,infoblock_dict, gdt_dict,dimension_dict)
    


    