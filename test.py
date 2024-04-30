import streamlit as st
from PIL import Image
from skimage.feature import hog
import joblib as joblib
clf = joblib.load('fish_classifier_model.pkl')
lables= {0:"Healthy",1:"Unhealthy"}

def extract_features(image):
# Convert the image to grayscale
    gray_image = image.convert('L')
# Resize the image to a fixed size
    resized_image = gray_image.resize((64, 64))
    # Extract HOG features
    hog_features = hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
    return hog_features
def main():
    st.title("Fish Healthiness Assurance")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image file
        image = Image.open(uploaded_file)
        y_pred = clf.predict([extract_features(image)])
        # Display the image
        st.image(image, caption=lables[y_pred[0]],width=300)
        st.title(lables[y_pred[0]])

if __name__ == "__main__":
    main()