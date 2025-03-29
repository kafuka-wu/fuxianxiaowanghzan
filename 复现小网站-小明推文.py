import streamlit as st
import pandas as pd
import joblib
import base64

# 加载模型
models = {
    'LR': joblib.load('LR_best_model.pkl')  # 只加载LR模型
}

# 定义输入字段的映射
def get_input_fields():
    return {
        'gender': st.selectbox("Gender", ['男', '女']),
        'satisfaction': st.selectbox("Satisfaction", ['满意', '一般', '不满意']),
        'health_status': st.selectbox("Health Status", ['良好', '一般', '差']),
        'pain': st.selectbox("Pain", ['有', '无']),
        'adl_score': st.slider("ADL Score", min_value=0, max_value=100, value=50),
        'sleep_hours': st.slider("Sleep Hours", min_value=0, max_value=24, value=8),
        'digestive_disease': st.selectbox("Digestive Disease", ['有', '无']),
        'arthritis': st.selectbox("Arthritis", ['有', '无']),
        'hearing_impairment': st.selectbox("Hearing Impairment", ['有', '无']),
        'vision_impairment': st.selectbox("Vision Impairment", ['有', '无']),
        'fall_history': st.selectbox("Fall History", ['有', '无'])
    }

# 将输入转换为模型所需的格式
def preprocess_input(data):
    processed_data = {}
    for key, value in data.items():
        if isinstance(value, str):  # 如果字段是分类的
            values = input_fields[key]
            processed_data[key] = values.index(value)
        else:  # 如果字段是数字的
            processed_data[key] = value
    return pd.DataFrame([processed_data])

# 预测函数
def predict(data):
    model = models['LR']
    processed_data = preprocess_input(data)
    prediction = model.predict(processed_data)
    probability = model.predict_proba(processed_data)[:, 1][0]
    return prediction[0], probability

# 读取静态图片并转换为 Base64 编码
with open(r'C:\Users\86173\Desktop\微信图片_20250329192221.jpg', 'rb') as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# 设置静态图片背景
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('data:image/jpeg;base64,{encoded_string}');
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit 应用
st.title("COPD Depression Risk Predictor")
st.markdown("☁☁🌈欢迎来到健康小站！☁☁")
st.markdown("这里是由卡夫阿卡开发的在线网站，用于帮助患有COPD（慢性阻塞性肺气肿）的人预测自己是否有抑郁的风险。")
st.markdown("请您根据以下的提示选出符合自身条件的选项，最后点击“预测”，即能出现患抑郁症的风险。")

# 获取用户输入
input_fields = get_input_fields()
data = {key: value for key, value in input_fields.items()}

# 预测按钮
if st.button("预测"):
    prediction, probability = predict(data)
    st.write(f"预测结果: {prediction}")
    st.write(f"概率: {probability:.2f}")

# 结束语
st.markdown("结果解读：")
st.markdown("预测结果为1，则代表您已经是抑郁的状态；预测结果为0，则代表您目前不是抑郁的状态。概率的多少（0-1）代表您有多大可能性患抑郁。")
st.markdown("如果您觉得这个网站能帮到您及有需要的人的话，请帮忙转发，希望能帮助到更多需要的人！😀😀")