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

html_7 = """
<div style="background-color:#EC7063;padding:15px;border-radius:15px;border-style:solid;border-color:black">
<center><h5>สถิติข้อมูลดอกไม้</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")

# อ่านข้อมูล CSV
dt = pd.read_csv("/mnt/data/iris.csv")
st.write(dt.head(10))

# สรุปข้อมูล
dt1 = dt['petallength'].sum()
dt2 = dt['petalwidth'].sum()
dt3 = dt['sepallength'].sum()
dt4 = dt['sepalwidth'].sum()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["d1", "d2", "d3", "d4"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
    st.bar_chart(dx2)
else:
    st.write("ไม่แสดงข้อมูล")

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px;border-style:solid;border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

pt_len = st.slider("กรุณาเลือกข้อมูล petal.length")
pt_wd = st.slider("กรุณาเลือกข้อมูล petal.width")

sp_len = st.number_input("กรุณาเลือกข้อมูล sepal.length")
sp_wd = st.number_input("กรุณาเลือกข้อมูล sepal.width")

if st.button("ทำนายผล"):

    dt = pd.read_csv("/mnt/data/iris.csv")
    X = dt.drop('variety', axis=1)
    y = dt['variety']

    Knn_model = KNeighborsClassifier(n_neighbors=3)
    Knn_model.fit(X, y)

    x_input = np.array([[pt_len, pt_wd, sp_len, sp_wd]])
    out = Knn_model.predict(x_input)
    st.write(out)

    # แสดงรูปที่ทำนาย
    if out[0] == 'Setosa':
        st.image("/mnt/data/iris1.jpg")
    elif out[0] == 'Versicolor':
        st.image("/mnt/data/iris2.jpg")
    else:
        st.image("/mnt/data/iris3.jpg")

else:
    st.write("ไม่ทำนาย")
