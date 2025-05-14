import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Function to load dataset
@st.cache_data
def load_data():
    dataset = pd.read_csv('/content/Crop and fertilizer dataset.csv')  # Replace 'your_dataset.csv' with your dataset file path
    return dataset

# Function to update Soil Color options based on selected District
def update_soil_color_options(dataset, district_name):
    return dataset[dataset['District_Name'] == district_name]['Soil_color'].unique()

# Function to train model and make recommendations
def train_model(dataset, input_data):
    # Perform one-hot encoding for District_Name and Soil_color columns
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_encoded = encoder.fit_transform(dataset[['District_Name', 'Soil_color']])
    input_data_encoded = encoder.transform(input_data[['District_Name', 'Soil_color']])

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, dataset['Crop'], test_size=0.2, random_state=42)

    # Train the random forest model
    model_crop = RandomForestClassifier(n_estimators=100, random_state=42)
    model_crop.fit(X_train, y_train)

    # Make predictions
    predicted_crop = model_crop.predict(input_data_encoded)

    # Find the fertilizer associated with the recommended crop
    recommended_fertilizer = dataset[dataset['Crop'] == predicted_crop[0]]['Fertilizer'].values[0]
    link = dataset[(dataset['Crop'] == predicted_crop[0]) & (dataset['Fertilizer'] == recommended_fertilizer)]['Link'].values[0]

    return predicted_crop[0], recommended_fertilizer, link

# Function to predict leaf disease
def predict_leaf_disease(image) :
    with st.spinner('Loading Model...'):
        classifier_model = keras.models.load_model(r'final_model.h5', compile = False)

    shape = ((256,256,3))
    model = keras.Sequential([hub.KerasLayer(classifier_model, input_shape = shape)])     # ye bhi kaam kar raha he
    test_image = image.resize((256, 256))
    test_image = keras.preprocessing.image.img_to_array(test_image)
    test_image /= 255.0
    test_image = np.expand_dims(test_image, axis = 0)
    class_name = ['Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy']

    prediction = model.predict(test_image)
    confidence = round(100 * (np.max(prediction[0])), 2)
    final_pred = class_name[np.argmax(prediction)]
    return final_pred, confidence

# Load dataset
dataset = load_data()

# Main title
st.title(' AgroAI ')

# Sidebar title and background color
st.sidebar.title('Select Parameters:')
st.sidebar.markdown('For Crop Recommendation:')

# Select District for crop recommendation
district_name = st.sidebar.selectbox('District:', dataset['District_Name'].unique())

# Update Soil Color options based on selected District
soil_colors = update_soil_color_options(dataset, district_name)
soil_color = st.sidebar.selectbox('Soil Color:', soil_colors)

# Select parameters for crop recommendation
st.sidebar.subheader('Soil Nutrients Levels:')
nitrogen = st.sidebar.slider('Nitrogen (in %)', min_value=0.0, max_value=100.0, step=0.1)
phosphorus = st.sidebar.slider('Phosphorus (in %)', min_value=0.0, max_value=100.0, step=0.1)
potassium = st.sidebar.slider('Potassium (in %)', min_value=0.0, max_value=100.0, step=0.1)

st.sidebar.subheader('Environmental Conditions:')
pH = st.sidebar.slider('pH', min_value=0.0, max_value=14.0, step=0.1)
rainfall = st.sidebar.slider('Rainfall (in mm)', min_value=0.0, max_value=5000.0, step=1.0)
temperature = st.sidebar.slider('Temperature (in Celsius)', min_value=0.0, max_value=50.0, step=0.1)

# For potato leaf disease prediction
file_uploaded = st.sidebar.file_uploader('Upload an image for Leaf Disease Detection...', type = ['jpg', 'jpeg', 'png'])

# Train the model and make recommendations for crop
input_data = pd.DataFrame([[nitrogen, phosphorus, potassium, pH, rainfall, temperature, district_name, soil_color]],
                           columns=['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature', 'District_Name', 'Soil_color'])

# Crop recommendation section
st.sidebar.markdown('---')
if st.sidebar.button('Get Crop Recommendation'):
    recommended_crop, recommended_fertilizer, link = train_model(dataset, input_data)
    
    # Display recommendation
    st.markdown('## Crop Recommendation')
    st.write(f"**Recommended Crop:** {recommended_crop}")
    st.write(f"**Recommended Fertilizer:** {recommended_fertilizer}")
    st.write(f"**Link:** [{recommended_crop} Info]({link})")

# Leaf disease detection section
if file_uploaded is not None:
    st.markdown('---')
    st.write("Uploaded Image for Leaf Disease Detection:")
    image = Image.open(file_uploaded)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    result, confidence = predict_leaf_disease(image)
    st.markdown('## Leaf Disease Prediction')
    st.write('Prediction : {}'.format(result))
    st.write('Confidence : {}%'.format(confidence))

footer = """<style>
a:link , a:visited{
    color: white;
    background-color: transparent;
    text-decoration: None;
}

a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: None;
}

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #1a1a1a;
    color: white;
    text-align: center;
    padding: 10px;
}
</style>

<div class="footer">
<p align="center"> <a> Designed by Yash Katke</a></p>
</div>
        """

st.markdown(footer, unsafe_allow_html = True)
