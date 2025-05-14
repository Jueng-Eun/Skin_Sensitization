
import streamlit as st
import numpy as np
import pickle

with open("ghs_new_rf_model.pkl", "rb") as f:
    ghs_model = pickle.load(f)

mw = st.number_input("MW (분자량)", min_value=0.0, value=200.0)
mp = st.number_input("MP (녹는점)", min_value=-100.0, value=50.0)
st_val = st.number_input("ST (표면장력)", min_value=0.0, value=20.0)

if st.button("예측하기"):
    features = np.array([[mw, mp, st_val]])
    prediction = ghs_model.predict(features)[0]
    proba = ghs_model.predict_proba(features)[0]

    label_map = {0: "N", 1: "1B", 2: "1A"}
    final_label = prediction

    st.markdown(f"### 최종 예측 피부감작성 (GHS): **{label_map[final_label]}**")
    
    # 시각화를 위한 라벨 적용
    proba_series = pd.Series(proba, index=[label_map[i] for i in range(len(proba))])
    st.bar_chart(proba_series)
