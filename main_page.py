import os
import numpy as np
import json
import streamlit as st
from keras.models import load_model
from PIL import Image

# Set the page config
st.set_page_config(layout='wide')

# Set the title of the app
st.markdown("<h1 style='text-align: center;'>Eye Disease Detection App</h1>", unsafe_allow_html=True)

st.markdown('''
<div style="text-align: center">
Please follow the following instructions:
<ul>
<li>1. Upload the Fundus image.</li>
<li>2. Wait for the file to be uploaded.</li>
<li>3. Click the Predict button to predict the disease.</li>
</ul>
</div>
''', unsafe_allow_html=True)

# Add a description
st.markdown('<div style="text-align: center">Upload an image and click on the Predict button to classify the image.</div>', unsafe_allow_html=True)

image = st.file_uploader('Choose an image file', type=['jpeg', 'jpg', 'png'])

if image:
    im = Image.open(image)
    im = im.resize((150,150))
        
    test = np.array(im)
    test = np.expand_dims(test, axis=0)
        
    model = load_model('./final.h5')

    # Display the uploaded image
    st.image(image, caption='Uploaded Image',  width=300)

    # Create a button
    if st.button('Predict'):
        prediction = model.predict(test)
        predictions = prediction.tolist()[0]
        prediction = np.argmax(predictions)
        percentage = predictions[prediction]

        # Create a dictionary for the response
        responses = {"prediction": str(prediction), "percentage": percentage}

        # Map the prediction to the disease name
        CLASSES = {'Cataract': 0, 'Diabetes': 1, 'Glaucoma': 2, 'Normal': 3, 'Other': 4}
        disease = [key for key, value in CLASSES.items() if value == int(responses["prediction"])][0]

        # Display the disease and probability in a stylish way
        st.markdown(f"<div style='text-align: left; color: blue;'><h2>Eye Disease : {disease}</h2></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: left; color: green;'><h3>Probability: {responses['percentage']*100:.2f}%</h3></div>", unsafe_allow_html=True)

