import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_model():
  model=tf.keras.models.load_model('CPE 019 Final Exam_ Model Deployment in the Cloud.hdf5')
  return model
model=load_model()
st.write("""
# Image Car Classifier"""
)
file=st.file_uploader("Please note that only .JPG files work with the program.",type=["jpg","png"])

from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    size=(128,128)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload a car image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['Audi', 'Hyundai Creta', 'Mahindra Scorpio', 'Rolls Royce', 'Swift', 'Tata Safari', 'Toyota Innova']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)
