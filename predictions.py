import streamlit as st
from PIL import Image
from keras.preprocessing import image
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow import keras
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
st.set_page_config(layout="wide")


def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("https://wallpaperbat.com/img/161069-neural-network-wallpaper.gif");
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# Function to load uploaded image


def load_image(image_file):
    img = Image.open(image_file)
    return img
# Function to check the image


# Updated decorator for Streamlit versions >= 1.18.0
@st.cache_resource(ttl=48*3600)
def check():

    # FIX: Load the model without attempting to recompile it.
    # This prevents the ValueError caused by the old 'reduction=auto' loss function setting.
    lr = keras.models.load_model('weights.hdf5', compile=False)

    # Prediction Pipeline
    class Preprocessor(BaseEstimator, TransformerMixin):
        def fit(self, img_object):
            return self

        def transform(self, img_object):
            # The 'keras.preprocessing.image' module is often slightly separate from 'tensorflow.keras.utils'
            # If the line below fails, you might need to try 'tf.keras.utils.img_to_array(img_object)'
            img_array = image.img_to_array(img_object)
            expanded = (np.expand_dims(img_array, axis=0))
            return expanded

    class Predictor(BaseEstimator, TransformerMixin):
        def fit(self, img_array):
            return self

        def predict(self, img_array):
            # Model prediction still works fine even if the model wasn't recompiled
            probabilities = lr.predict(img_array)
            predicted_class = ['P_Deficiency', 'Healthy',
                               'N_Deficiency', 'K_Deficiency'][probabilities.argmax()]
            return predicted_class

    full_pipeline = Pipeline([('preprocessor', Preprocessor()),
                              ('predictor', Predictor())])
    return full_pipeline


def output(full_pipeline, img):
    a = img
    # Ensure the image is resized to the expected input shape for the model
    a = a.resize((224, 224))
    predic = full_pipeline.predict(a)
    return (predic)


def main():
    # giving a title
    set_bg_hack_url()
    col1, col2 = st.columns(2)

    with col1:
        st.title('H.A.R.N.')
        st.subheader(
            'Image Classification Using CNN for identifying Plant Nutrient Deficiencies')
        image_file = st.file_uploader(
            "Upload Images", type=["png", "jpg", "jpeg"])
        # code for Prediction
        prediction = ''

        # creating a button for Prediction

        if st.button('Predict'):
            if image_file is not None:
                # To See details
                with st.spinner('Loading Image and Model...'):
                    full_pipeline = check()

                # Check for compatibility of st.cache decorator and switch to st.cache_resource
                # file_details = {"filename":image_file.name, "filetype":image_file.type,"filesize":image_file.size}
                # st.write(file_details)

                img = load_image(image_file)
                w = img.size[0]
                h = img.size[1]

                # Display image with responsive sizing
                if w > h:
                    w = 600
                    st.image(img, width=w)
                else:
                    w = w*(600.0/h)
                    st.image(img, width=int(w))

                with st.spinner('Predicting...'):
                    prediction = output(full_pipeline, img)

                # Format the success message for clarity
                st.success(f'Prediction: {prediction}')
            else:
                st.warning(
                    "Please upload an image file to proceed with prediction.")


if __name__ == '__main__':
    main()
