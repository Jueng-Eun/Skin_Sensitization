
import streamlit as st
import numpy as np
import pickle

with open("rf_np_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("NP 클래스 예측기")
st.markdown("### 물질의 특성을 입력하세요:")

mw = st.number_input("MW (분자량)", min_value=0.0, value=200.0)
mp = st.number_input("MP (녹는점)", min_value=-100.0, value=50.0)
st_val = st.number_input("ST (표면장력)", min_value=0.0, value=20.0)

if st.button("예측하기"):
    features = np.array([[mw, mp, st_val]])
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    st.success(f"예측된 NP 클래스: **{prediction}**")
    st.bar_chart(proba)
