from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.header("TIW")

# ใช้ไฟล์ภาพที่คุณอัปโหลดใน ChatGPT
st.image("/mnt/data/6a8a1184-98e0-4342-b9d8-34d6b60753ef.png")

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Versicolor")
    st.image("/mnt/data/iris1.jpg")

with col2:
    st.header("Verginica")
    st.image("/mnt/data/iris2.jpg")

with col3:
    st.header("Setosa")
    st.image("/mnt/data/iris3.jpg")


