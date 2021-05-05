

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
        model = model_set
    
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)
        
        return prediction

    
def import_and_predict_face(image_data, model):
        model = model_set
    
        size = (128, 48,48, 1)    
        image = ImageOps.fit(image_data, size, Image.NEAREST)
        image = ImageOps.grayscale(image)
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)
        #print(image.shape)

        #img_reshape = np.reshape(image.shape[0], image.shape[1], 1) #[np.newaxis,...]

        prediction = model.predict(image)
        
        return prediction
    
    
#model = tf.keras.models.load_model('models//skin_model_tl_1')
model_set = None

def model_select(model_invoked):
    model = None
    file = None
    if model_invoked == 'Eyes':
        model = tf.keras.models.load_model('models//eye_model_2')
        #file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])
    elif model_invoked == 'Skin':
        model = tf.keras.models.load_model('models//skin_model_tl_1')
        #file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])
    else:
        model = tf.keras.models.load_model('models//face_model')
        #file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])     
    return model



#file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])
#

def predict_skin(file, model):
    classes=['akiec','bcc','bkl','df','mel','nv','vasc']
    if file is None:
        st.text("You haven't uploaded an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
    
        if np.argmax(prediction) == 0:
            st.write("It is akiec - Actinic keratosis / Bowen’s disease")
        elif np.argmax(prediction) == 1:
            st.write("It is bcc - Basal cell carcinoma")
        elif np.argmax(prediction) == 2:
            st.write("It is bkl - Benign keratosis")
        elif np.argmax(prediction) == 3:
            st.write("It is df - Dermatofibroma")
        elif np.argmax(prediction) == 4:
            st.write("It is mel - Melanoma")
        elif np.argmax(prediction) == 5:
            st.write("It is nv - Melanocytic nevus")
        else:
            st.write("It is vasc - Vascular lesion")
    
        st.text("Probability (0: akiec, 1: bcc, 2: bkl, 3: df, 4: mel, 5: nv, 6: vasc) in %")
        st.write(prediction*100)
    

def predict_eyes(file, model):
    classes=['Crossed Eye','Glaucoma', 'Healthy']
    if file is None:
        st.text("You haven't uploaded an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
    
        if np.argmax(prediction) == 0:
            st.write("It is Crossed Eye class")
        elif np.argmax(prediction) == 1:
            st.write("It is Glaucoma class")
        else:
            st.write("It is Healthy class")
    
        st.text("Probability (0: Crossed Eye, 1: Glaucoma, 2: Healthy) in %")
        st.write(prediction*100)

def predict_face(file, model):
    classes=['Negative Emotion','Neutral','Positive Emotion']
    if file is None:
        st.text("You haven't uploaded an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict_face(image, model)
    
        if np.argmax(prediction) == 0:
            st.write("It is Negative Emotion")
        elif np.argmax(prediction) == 1:
            st.write("It is Neutral")
        else:
            st.write("It is Positive Emotion")
    
        st.text("Probability (0: Negative Emotion, 1: Neutral, 2: Positive Emotion) in %")
        st.write(prediction*100)

        
st.write("""
         # Classifier For Medical Image Processing
         """
         )

st.write("This is a simple image classification web app to predict labels of Medical Images")

choice_model = st.selectbox("Select from the options - Eyes, Skin and Facial Images", ("Eyes","Skin","Facial Expressions"))
model_set = model_select(choice_model)

if choice_model == 'Skin':
    file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])
    predict_skin(file, model_set)
elif choice_model == 'Eyes':
    file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])
    predict_eyes(file, model_set)
else:
    file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])
    predict_face(file, model_set)
    
