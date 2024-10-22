import streamlit as st
import tensorflow as tf
import numpy as np 

#tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model(r'C:\Users\vivek\OneDrive\Desktop\ML PROJECT Apple Classification\trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=[256, 256])
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    
    # Define class names
    class_names = ['Fresh_State', 'Last_stage', 'Mid_Stage', 'labels']
    
    # Define calorie prediction based on apple stage
    calorie_prediction = 72 if result_index == 0 else (50 if result_index == 2 else 0)
    
    return class_names[result_index], calorie_prediction


#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Stage Recognition"])

#Home Page
if app_mode == "Home":
    st.header("APPLE STAGES DETECTION SYSTEM")
    st.markdown("""
    Welcome to the Apple Stage Recognition System! 🌿🔍
    
    Our mission is to help in identifying apple stages whether it is fresh or rotten. Upload an image of an apple, and our system will analyze it to detect the stage of the apple. 

    ### How It Works
    1. **Upload Image:** Go to the **Stages Recognition** page and upload an image of an apple.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify stages.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate stage detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Apple Stages Recognition** page in the sidebar to upload an image and experience the power of our Apple Stages Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
""")

#About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
               The Fresh and Rotten/Stale Apple Classification Dataset is a comprehensive collection of high-quality 
               images specifically curated for the purpose of training and evaluating classification models. 
               This dataset is designed to aid in the development of computer vision algorithms that can accurately distinguish between fresh and rotten/stale produce.
               
                #### Content
                1. train (933 images)
                2. test (223 images)
                3. validation (320 images)

                """)

#Prediction Page
elif app_mode == "Stage Recognition":
    st.header("Stage Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image") and test_image:
        st.image(test_image, use_column_width=True)
    #Predict Button
    if st.button("Predict") and test_image:
        st.balloons()
        st.write("Our Prediction")
        apple_stage, calorie_prediction = model_prediction(test_image)
        st.success("Model is Predicting an apple's stage as: {}".format(apple_stage))
        st.success("Predicted Calorie Content: {} calories".format(calorie_prediction))




