import streamlit as st

st.set_page_config(layout='wide', page_title='Object Detection')

st.title("Object Detection using Yolov4-tiny")
st.markdown('''
This project was created as part of my :red[*Masters Degree*]. I had to develop a software that could detect
objects in :red[*images*].
The main objective of this project was that it should be able to run on any laptop, regardless of the computational power.
The entire project is on my github account which you can access from here:
            https://github.com/shahumSultan/EasyFind

This is just a demo of that product to showcase how it works. Happy Detecting :sunglasses:
''')

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

with col2:
    if uploaded_file is not None:
        st.write("Image exists")