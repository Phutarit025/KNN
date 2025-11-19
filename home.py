import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# --- Header ---
st.header("Tiw")
st.image("./img/Tiw.jpg")

# --- Columns แสดงรูปดอกไม้ ---
col1, col2, col3 = st.columns(3)
with col1:
    st.header("Versicolor")
    st.image("./img/iris1.jpg")
with col2:
    st.header("Verginiga")
    st.image("./img/iris2.jpg")
with col3:
    st.header("Setosa")
    st.image("./img/iris3.jpg")

# --- HTML Section สถิติข้อมูล ---
html_7 = """
<div style="background-color:#EC7063;padding:15px;border-radius:15px;border-style:solid;border-color:black">
<center><h5>สถิติข้อมูลดอกไม้</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)

# --- โหลดข้อมูล iris.csv ---
dt = pd.read_csv("./data/iris.csv")
st.write(dt.head(10))

# --- สรุปข้อมูลเพื่อแสดง bar chart ---
dx = [dt['petallength'].sum(), dt['petalwidth'].sum(), dt['sepallength'].sum(), dt['sepalwidth'].sum()]
dx2 = pd.DataFrame(dx, index=["petallength", "petalwidth", "sepallength", "sepalwidth"], columns=["Sum"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
    st.bar_chart(dx2)
else:
    st.write("ไม่แสดงข้อมูล")

# --- HTML Section ทำนายข้อมูล ---
html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px;border-style:solid;border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)

# --- Input ข้อมูลจากผู้ใช้ ---
pt_len = st.slider("กรุณาเลือกข้อมูล petal.length", float(dt['petallength'].min()), float(dt['petallength'].max()))
pt_wd = st.slider("กรุณาเลือกข้อมูล petal.width", float(dt['petalwidth'].min()), float(dt['petalwidth'].max()))
sp_len = st.number_input("กรุณาเลือกข้อมูล sepal.length", float(dt['sepallength'].min()), float(dt['sepallength'].max()))
sp_wd = st.number_input("กรุณาเลือกข้อมูล sepal.width", float(dt['sepalwidth'].min()), float(dt['sepalwidth'].max()))

# --- ปุ่มทำนายผล ---
if st.button("ทำนายผล"):
    X = dt.drop('variety', axis=1)
    y = dt['variety']
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X, y)

    x_input = np.array([[pt_len, pt_wd, sp_len, sp_wd]])
    out = knn_model.predict(x_input)
    st.write("ผลการทำนาย:", out[0])

    # แสดงรูปดอกไม้ตามผลทำนาย
    if out[0] == 'Setosa':
        st.image(os.path.join("img", "iris1.jpg"))
    elif out[0] == 'Versicolor':
        st.image(os.path.join("img", "iris2.jpg"))
    else:
        st.image(os.path.join("img", "iris3.jpg"))
else:
    st.write("ไม่ทำนาย")
