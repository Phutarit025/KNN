from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Header และภาพโปรไฟล์ ----------------
st.header("TIW")
st.image("/mnt/data/6a8a1184-98e0-4342-b9d8-34d6b60753ef.png")

# ---------------- 3 Columns ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Versicolor")
    st.image("./img/iris1.jpg")

with col2:
    st.header("Virginica")
    st.image("./img/iris2.jpg")

with col3:
    st.header("Setosa")
    st.image("./img/iris3.jpg")

# ---------------- Title สถิติข้อมูล ----------------
html_7 = """
<div style="background-color:#EC7063;padding:15px;border-radius:15px;border-style:solid;border-color:black">
<center><h5>สถิติข้อมูลดอกไม้</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)

# ---------------- Load Data ----------------
dt = pd.read_csv("./data/iris.csv")
st.write(dt.head(10))

# ---------------- สถิติโดยใช้ sum() ----------------
dt1 = dt['petal.length'].sum()
dt2 = dt['petal.width'].sum()
dt3 = dt['sepal.length'].sum()
dt4 = dt['sepal.width'].sum()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["petal.length", "petal.width", "sepal.length", "sepal.width"])

# ---------------- ปุ่มแสดง/ไม่แสดง chart ----------------
if st.button("แสดงการจินตทัศน์ข้อมูล"):
    st.bar_chart(dx2)
else:
    st.write("ไม่แสดงข้อมูล")

# ---------------- Title ทำนายข้อมูล ----------------
html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px;border-style:solid;border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)

# ---------------- Input Fields ----------------
pt_len = st.slider("กรุณาเลือกข้อมูล petal.length", 0.0, 7.0, 1.0)
pt_wd = st.slider("กรุณาเลือกข้อมูล petal.width", 0.0, 3.0, 0.2)
sp_len = st.number_input("กรุณาเลือกข้อมูล sepal.length")
sp_wd = st.number_input("กรุณาเลือกข้อมูล sepal.width")

# ---------------- ปุ่มทำนาย ----------------
if st.button("ทำนายผล"):

    X = dt.drop('variety', axis=1)
    y = dt['variety']

    Knn_model = KNeighborsClassifier(n_neighbors=3)
    Knn_model.fit(X, y)

    # *** ต้องเรียง feature ให้ตรงกับลำดับใน dataset ***
    x_input = np.array([[sp_len, sp_wd, pt_len, pt_wd]])

    out = Knn_model.predict(x_input)
    st.write(out[0])

    # แสดงรูป
    if out[0] == 'Setosa':
        st.image("./pic/iris1.jpg")
    elif out[0] == 'Versicolor':
        st.image("./pic/iris2.jpg")
    else:
        st.image("./pic/iris3.jpg")

else:
    st.write("ไม่ทำนาย")
