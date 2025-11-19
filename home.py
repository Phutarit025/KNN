from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


st.title('โปรเจคทำนายชนิดของดอกไอริสด้วย KNN')
st.subheader('เว็บแอปพลิเคชันสำหรับทำนายสายพันธุ์ดอกไอริส')

st.sidebar.image("./img/Tiw.jpg")

st.markdown("---")

st.write("## สายพันธุ์ดอกไอริส (Iris Species)")
col1, col2, col3 = st.columns(3)

with col1:
   st.subheader("Versicolor")
   st.image("./img/iris1.jpg", caption="Iris Versicolor")

with col2:
   st.subheader("Virginica")
   st.image("./img/iris2.jpg", caption="Iris Virginica")

with col3:
   st.subheader("Setosa")
   st.image("./img/iris3.jpg", caption="Iris Setosa")



# Use caching to load data
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# Expander for data statistics
with st.expander("ดูสถิติและกราฟข้อมูลดอกไม้ (View Data Statistics and Chart)"):
    dt = load_data("./data/iris.csv")
    
    st.write("#### ข้อมูล 10 แถวแรก (First 10 rows)")
    st.write(dt.head(10))

    # Prepare data for the chart with meaningful labels
    feature_sums = dt[['petal.length', 'petal.width', 'sepal.length', 'sepal.width']].sum()
    sum_df = pd.DataFrame(feature_sums, columns=["ผลรวม (Total Sum)"])
    
    st.write("#### กราฟแท่งแสดงผลรวมของแต่ละคุณลักษณะ (Bar Chart of Feature Sums)")
    st.bar_chart(sum_df)



@st.cache_resource
def train_knn_model():
    dt = load_data("./data/iris.csv")
    X = dt.drop('variety', axis=1)
    y = dt.variety
    Knn_model = KNeighborsClassifier(n_neighbors=3)
    Knn_model.fit(X, y)
    return Knn_model

st.markdown("---")

with st.expander("ทำนายชนิดของดอกไอริส (Predict Iris Species)"):
    st.write("### ป้อนค่าเพื่อทำนาย (Enter values for prediction)")

    col_input1, col_input2 = st.columns(2)
    with col_input1:
        pt_len = st.slider("ความยาวกลีบดอก (petal.length)", min_value=0.0, max_value=8.0, value=4.0, step=0.1, key='pt_len')
        pt_wd = st.slider("ความกว้างกลีบดอก (petal.width)", min_value=0.0, max_value=3.0, value=1.5, step=0.1, key='pt_wd')
    with col_input2:
        sp_len = st.number_input("ความยาวกลีบเลี้ยง (sepal.length)", min_value=0.0, max_value=8.0, value=5.0, step=0.1, key='sp_len')
        sp_wd = st.number_input("ความกว้างกลีบเลี้ยง (sepal.width)", min_value=0.0, max_value=5.0, value=3.0, step=0.1, key='sp_wd')

    if st.button("ทำนายผล (Predict)"):
        Knn_model = train_knn_model()
        x_input = np.array([[pt_len, pt_wd, sp_len, sp_wd]])
        prediction = Knn_model.predict(x_input)
        
        st.subheader(f"ผลการทำนาย: {prediction[0]}")
        
        if prediction[0] == 'Setosa':
            st.image("./img/iris3.jpg", caption=f"Predicted: {prediction[0]}")
        elif prediction[0] == 'Versicolor':       
            st.image("./img/iris1.jpg", caption=f"Predicted: {prediction[0]}")
        else: # Virginica
            st.image("./img/iris2.jpg", caption=f"Predicted: {prediction[0]}")