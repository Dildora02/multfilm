# # File: cartoonizer_app.py
# import os
# import numpy as np
# import cv2
# import streamlit as st
# from PIL import Image

# # Define paths
# UPLOAD_FOLDER = "uploads"
# CARTOON_FOLDER = "cartoon_images"

# # Ensure folders exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(CARTOON_FOLDER, exist_ok=True)

# # Cartoonize functions
# def cartoonize_1(img, k):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.adaptiveThreshold(
#         gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8)
#     data = np.float32(img).reshape((-1, 3))
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
#     _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#     center = np.uint8(center)
#     result = center[label.flatten()].reshape(img.shape)
#     blurred = cv2.medianBlur(result, 3)
#     cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
#     return cartoon

# def cartoonize_2(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return cv2.stylization(img, sigma_s=150, sigma_r=0.25)

# def cartoonize_3(img):
#     _, imout = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
#     return imout

# def cartoonize_4(img):
#     imout_gray, _ = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
#     return imout_gray

# def cartoonize_5(img, k):
#     img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img1g = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
#     img1b = cv2.medianBlur(img1g, 3)
#     imgf = np.float32(img1).reshape(-1, 3)
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
#     _, label, center = cv2.kmeans(imgf, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#     center = np.uint8(center)
#     final_img = center[label.flatten()].reshape(img1.shape)
#     edges = cv2.adaptiveThreshold(img1b, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 3)
#     final = cv2.bitwise_and(final_img, final_img, mask=edges)
#     return final

# def cartoonize_6(img):
#     return cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

# # Streamlit UI
# st.title("Cartoonizer App")
# st.write("Upload an image to apply cartoon effects!")

# uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

# if uploaded_file:
#     # Load and display image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)
#     img_array = np.array(image)
#     style = st.selectbox("Choose Cartoon Style", ["Style1", "Style2", "Style3", "Style4", "Style5", "Style6"])

#     if st.button("Cartoonize"):
#         if style == "Style1":
#             cartoon = cartoonize_1(img_array, 8)
#         elif style == "Style2":
#             cartoon = cartoonize_2(img_array)
#         elif style == "Style3":
#             cartoon = cartoonize_3(img_array)
#         elif style == "Style4":
#             cartoon = cartoonize_4(img_array)
#         elif style == "Style5":
#             cartoon = cartoonize_5(img_array, 5)
#         elif style == "Style6":
#             cartoon = cartoonize_6(img_array)

#         # Save and display cartoonized image
#         result_path = os.path.join(CARTOON_FOLDER, f"cartoonized_{style}.jpg")
#         cv2.imwrite(result_path, cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR))
#         st.image(cartoon, caption=f"Cartoonized Image ({style})", use_column_width=True)
#         st.success(f"Cartoonized image saved to {result_path}!")

import os
import numpy as np
import cv2
import streamlit as st
from PIL import Image

download_folder = "cartoon_images"
os.makedirs(download_folder, exist_ok=True)

def cartoonize_1(img, k):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8)
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()].reshape(img.shape)
    blurred = cv2.medianBlur(result, 3)
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
    return cartoon

st.title("Cartoonizer App")
st.write("Upload an image to apply cartoon effects!")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=False)  # Original size
    img_array = np.array(image)
    
    style = st.selectbox("Choose Cartoon Style", ["Style1"])
    
    if st.button("Cartoonize"):
        cartoon = cartoonize_1(img_array, 8)
        
        result_path = os.path.join(download_folder, "cartoonized_image.jpg")
        cv2.imwrite(result_path, cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR))
        
        st.image(cartoon, caption="Cartoonized Image", use_column_width=False)
        
        with open(result_path, "rb") as file:
            btn = st.download_button(
                label="Download Cartoonized Image",
                data=file,
                file_name="cartoonized_image.jpg",
                mime="image/jpeg"
            )

# # coding=utf-8
# import sys
# import os, shutil
# import glob
# import re
# import numpy as np
# import cv2
# import subprocess



# # Flask ca
# from flask import Flask,flash, request, render_template,send_from_directory
# from werkzeug.utils import secure_filename


# # Define a flask app
# app = Flask(__name__, static_url_path='')
# app.secret_key = os.urandom(24)

# app.config['CARTOON_FOLDER'] = 'cartoon_images'
# app.config['UPLOAD_FOLDER'] = 'uploads'


# @app.route('/uploads/<filename>')
# def upload_img(filename):
    
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# @app.route('/cartoon_images/<filename>')
# def cartoon_img(filename):
    
#     return send_from_directory(app.config['CARTOON_FOLDER'], filename)


# def cartoonize_1(img, k):

#     # Convert the input image to gray scale 
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Peform adaptive threshold
#     edges  = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8)

#     # cv2.imshow('edges', edges)

#     # Defining input data for clustering
#     data = np.float32(img).reshape((-1, 3))

  

#     # Defining criteria
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

#     # Applying cv2.kmeans function
#     _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#     center = np.uint8(center)
#     # print(center)

#     # Reshape the output data to the size of input image
#     result = center[label.flatten()]
#     result = result.reshape(img.shape)
#     #cv2.imshow("result", result)

#     # Smooth the result
#     blurred = cv2.medianBlur(result, 3)

#     # Combine the result and edges to get final cartoon effect
#     cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)

#     return cartoon

# def cartoonize_2(img):

#     # Convert the input image to gray scale 
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     # stylization of image
#     img_style = cv2.stylization(img, sigma_s=150,sigma_r=0.25)
    
#     return img_style

# def cartoonize_3(img):

#     # Convert the input image to gray scale 
    
    
#     # pencil sketch  of image
    
#     imout_gray, imout = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    
#     return imout_gray

# def cartoonize_4(img):

#     # Convert the input image to gray scale 
    
    
#     # pencil sketch  of image
    
#     imout_gray, imout = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    
#     return imout

# def cartoonize_5(img, k):

#     # Convert the input image to gray scale 
#     img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img1g=cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
#     img1b=cv2.medianBlur(img1g,3)
#     #Clustering - (K-MEANS)
#     imgf=np.float32(img1).reshape(-1,3)
#     criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
#     compactness,label,center=cv2.kmeans(imgf,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
#     center=np.uint8(center)
#     final_img=center[label.flatten()]
#     final_img=final_img.reshape(img1.shape)
#     edges=cv2.adaptiveThreshold(img1b,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,3)
#     final=cv2.bitwise_and(final_img,final_img,mask=edges)

#     return final

# def cartoonize_6(img):

#     # Convert the input image to gray scale 
    
    
#     # pencil sketch  of image
    
#     dst = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    
#     return dst


# @app.route('/', methods=['GET'])
# def index():
#     # Main page
#     return render_template('index.html')


# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         # Get the file from post request
        
#         f = request.files['file']
#         style = request.form.get('style')
#         print(style)
#         # Save the file to ./uploads
#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(
#             basepath, 'uploads', secure_filename(f.filename))
        
#         f.save(file_path)
#         file_name=os.path.basename(file_path)
        
#         # reading the uploaded image
        
#         img = cv2.imread(file_path)
#         if style =="Style1":
#             cart_fname =file_name + "_style1_cartoon.jpg"
#             cartoonized = cartoonize_1(img, 8)
#             cartoon_path = os.path.join(
#                 basepath, 'cartoon_images', secure_filename(cart_fname))
#             fname=os.path.basename(cartoon_path)
#             print(fname)
#             cv2.imwrite(cartoon_path,cartoonized)
#             return render_template('predict.html',file_name=file_name, cartoon_file=fname)
#         elif style =="Style2":
#             cart_fname =file_name + "_style2_cartoon.jpg"
#             cartoonized = cartoonize_2(img)
#             cartoon_path = os.path.join(
#                 basepath, 'cartoon_images', secure_filename(cart_fname))
#             fname=os.path.basename(cartoon_path)
#             print(fname)
#             cv2.imwrite(cartoon_path,cartoonized)
#             return render_template('predict.html',file_name=file_name, cartoon_file=fname)
#         elif style=="Style3":
#             cart_fname =file_name + "_style3_cartoon.jpg"
#             cartoonized = cartoonize_3(img)
#             cartoon_path = os.path.join(
#                 basepath, 'cartoon_images', secure_filename(cart_fname))
#             fname=os.path.basename(cartoon_path)
#             print(fname)
#             cv2.imwrite(cartoon_path,cartoonized)
#             return render_template('predict.html',file_name=file_name, cartoon_file=fname)
#         elif style=="Style4":
#             cart_fname =file_name + "_style4_cartoon.jpg"
#             cartoonized = cartoonize_4(img)
#             cartoon_path = os.path.join(
#                 basepath, 'cartoon_images', secure_filename(cart_fname))
#             fname=os.path.basename(cartoon_path)
#             print(fname)
#             cv2.imwrite(cartoon_path,cartoonized)
#             return render_template('predict.html',file_name=file_name, cartoon_file=fname)
#         elif style=="Style5":
#             cart_fname =file_name + "_style5_cartoon.jpg"
#             cartoonized = cartoonize_5(img,5)
#             cartoon_path = os.path.join(
#                 basepath, 'cartoon_images', secure_filename(cart_fname))
#             fname=os.path.basename(cartoon_path)
#             print(fname)
#             cv2.imwrite(cartoon_path,cartoonized)
#             return render_template('predict.html',file_name=file_name, cartoon_file=fname)
#         elif style=="Style6":
#             cart_fname =file_name + "_style6_cartoon.jpg"
#             cartoonized = cartoonize_6(img)
#             cartoon_path = os.path.join(
#                 basepath, 'cartoon_images', secure_filename(cart_fname))
#             fname=os.path.basename(cartoon_path)
#             print(fname)
#             cv2.imwrite(cartoon_path,cartoonized)
#             return render_template('predict.html',file_name=file_name, cartoon_file=fname)
#         else:
#              flash('Please select style')
#              return render_template('index.html')
            
       
              
        
#     return ""



# if __name__ == '__main__':
#         app.run(debug=True, host="localhost", port=5000)

