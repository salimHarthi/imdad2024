import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns 
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

st.set_page_config(layout="wide")

# df = pd.read_csv('detailed_journeys.csv')
st.header("Fish Tracking Dashboard")

tabs = st.tabs(["Journeys","Track Journey","Fish Healthiness","About Us"])

with tabs[0]:
    df = pd.read_csv("journey_df.csv").drop("Unnamed: 0",axis=1)
    filters = st.columns(4)
    selected_fisherman = filters[0].selectbox("Select Fisherman",options=["All"] + list(df['fisherman_id'].unique()))
    selected_location = filters[1].selectbox("Select Customer",options=["All"] +list(df['delivered_location'].unique()))
    selected_fishing = filters[2].selectbox("Select Fishing Location",options=["All"] +list(df['fishing_location'].unique()))
    selected_status = filters[3].selectbox("Select Status",options=["All"] +list(df['status'].unique()))

    if selected_fisherman != "All":
        df = df[df['fisherman_id'] == selected_fisherman]

    if selected_location != "All":
        df = df[df['delivered_location'] == selected_location]

    if selected_fishing != "All":
        df = df[df['fishing_location'] == selected_fishing]

    if selected_status != "All":
        df = df[df['status'] == selected_status]

    with st.expander("View All Journeys"):
        st.dataframe(df,use_container_width=True)


    st.metric("Number of Journeys", len(df))
    config = {'displayModeBar': False}



    
    n_colors = len(df['status'].unique())
    color_sequence = sns.color_palette("husl", n_colors=n_colors).as_hex()

    cols = st.columns(2)
    fig = px.histogram(df, 'status',color_discrete_sequence=color_sequence[3:])
    cols[0].plotly_chart(fig, config=config)
    fig = px.histogram(df, 'fisherman_id',color_discrete_sequence=color_sequence[3:])
    cols[1].plotly_chart(fig, config=config)

    cols = st.columns(2)
    fig = px.histogram(df, 'fishing_location',color_discrete_sequence=color_sequence[3:])
    cols[0].plotly_chart(fig, config=config)
    fig = px.histogram(df, 'delivered_location',color_discrete_sequence=color_sequence[3:])
    cols[1].plotly_chart(fig, config=config)

    


with tabs[1]:
    df = pd.read_csv('detailed_journeys.csv')
    filters = st.columns(3)
    selected_journey = filters[2].selectbox("Select Journey",options=df['id'].unique())
    journey_data = df[df['id'] == selected_journey].iloc[0]

    
    header_cols = st.columns([9,3])
    header_cols[0].header(journey_data['id'])

    # st.write(list(journey_data['history']))

    for hist in eval(journey_data['history']):

        with st.expander(f"{str(hist['time']).split('.')[0]} - - {hist['stage']}"):
            metrics = st.columns(4)
            metrics[2].metric("Avg Temp",hist['avg_temp'] )
            metrics[3].metric("Location",hist['location'] )

with tabs[2]:
    
    st.title("Fish Healthiness Assurance")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image file
        image = Image.open(uploaded_file)
        y_pred = clf.predict([extract_features(image)])
        # Display the image
        st.image(image, caption=lables[y_pred[0]],width=300)
        st.title(lables[y_pred[0]])

with tabs[3]:
    st.title("About Us")
    st.text(
        "At Cold Chain, we are passionate about revolutionizing the way fish tracking is done.\n Our journey began with a vision to address the challenges faced by fisheries and aquaculture\n industries in monitoring and managing their fish populations effectively."
    )
    container = st.container()
    team_members = [
    {"name": "Salim", "job_title": "Software Engineer", "image_url": "salim.jpeg"},
    {"name": "Ahmed", "job_title": "AI Engineer / Designer", "image_url": "ahmed.jpeg"},
    {"name": "Madona", "job_title": "CEO", "image_url": "madona.jpg"},
    {"name": "Saud", "job_title": "Blockchain Engineer/the GOAT", "image_url": "saud.jpeg"},
    {"name": "Ghadeer", "job_title": "Software Engineer ", "image_url": "ghadeer.jpg"},
    {"name": "Noor", "job_title": "Marine Scientist", "image_url": "noor.jpg"},
    {"name": "Isehaq", "job_title": "Markiting Speciallist", "image_url": "isehaq.jpg"},
]
    st.markdown("""
    <style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    </style>
""", unsafe_allow_html=True)
    num_columns = 3  # Number of columns to display
    
    # Calculate number of rows based on number of team members and columns
    num_rows = -(-len(team_members) // num_columns)

    # Create a grid layout with specified number of columns
    columns = st.columns(num_columns)

    for i, member in enumerate(team_members):
        with columns[i % num_columns]:
            image = member["image_url"]
            st.image(image,  width=200,)
            st.header(member["name"])
            st.subheader(member["job_title"])
    
