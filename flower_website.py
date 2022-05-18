import streamlit as st
import re
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit.elements.image import image_to_url
import pandas as pd
import requests

data = pd.read_csv('./102flowers/imagelabels.csv', encoding='gbk')

@st.cache(suppress_st_warning=True)
def img_process(image):
    image = np.array(Image.open(image).resize((80, 80)))                # 修改图片形状
    image = image.reshape((-1, 80, 80, 3))                              # reshape以输入model
    image = image / 255                                                 # 归一化
    out = model.predict(image)
    if np.max(out)<0.5:                                                 # 最大概率小于 0.5的丢掉
        out = '>>>暂未识别出花朵'
    elif np.max(out) >= 0.5:
        out = tf.argmax(out, axis=1)
        out = data.loc[data['label']==out, '种类'].to_string(index=False)
        out = f'>>>识别结果:这是{out}'
    return image, out

st.set_page_config(
    page_title='花卉识别',    #页面标题
    page_icon='🍁',          #图标
    layout='wide',                #页面布局
    initial_sidebar_state="auto"  #侧边栏
)

# 第一次点击网页的话，放气球
if 'first_visit' not in st.session_state:
    st.session_state.first_visit=True
else:
    st.session_state.first_visit=False
# 初始化全局配置
if st.session_state.first_visit:
    st.balloons()

#设置背景
# image_url = image_to_url('./data/gif.gif', width=-1, clamp=False ,channels='RGB', output_format='auto', image_id='', allow_emoji=False)
image_url = 'https://i.gifer.com/embedded/download/816Q.gif'

st.markdown('''<style>.css-fg4pbf{background-image:url(''' + image_url + ''');background-size:100% 100%;background-attachment:fixed;}</style>
''', unsafe_allow_html=True)        # markdown可以拿来写 html,这里表示用 css写入背景
# st.markdown('''<style>.css-fg4pbf{background-image:url(''' + image_url + ''');background-size:100% 100%;background-attachment:fixed;}</style>''', unsafe_allow_html=True)

st.markdown('''<h1 align="center" style="color:rgb(197,64,179)" > 🌸 花 卉 识 别 网 页 APP </h1>
            <h5 align="center" style="color:rgb(255,255,255)"  font-size:"5px"><i>点击下方上传  / 拍照</i></h5>''', unsafe_allow_html=True)
# h1:最大标题   h5:最小标题     位置、颜色   <i>:斜体
img = st.file_uploader(' ')
try:
    st.image(np.array(Image.open(img).resize((150, 150))))
except:
    st.text('')

model_select = st.selectbox('💡点击择识别方法：',['Inception (正确率40%)', 'Resnet50V2 (正确率55%)', 'VGG16 (正确率60%)', 'Xception (正确率65%)'])
# st.markdown('''<style>#root > div:nth-child(1) > div > div > div > div > section > div > div:nth-child(1) >
# div > div > div.row-widget.stSelectbox > label {width: 571px; background:rgba(255,255,255,0.9); color:rgb(0,0,0);
#  border-radius:5px; filter:alpha(opacity=50)}</style>''', unsafe_allow_html=True)
# background: 背景颜色 rgba:( RGB三原色 + 不透明度 ) color: 字体颜色
# #号表示 id, 如 (#root)就是 (id:root)的意思，  (div.row-widget.stSelectbox) 表示 ( div class="row-widget.stSelectbox")
# div:nth-child(-n+9):nth-child(n+6){} 第六个到第九个子节点

st.markdown('''<style>#root > div:nth-child(1) > div > div > div > div > section > div > div:nth-child(1) > 
div > div.row-widget.stButton > div > button{color:#FF8C00; #  border-radius:5px;}</style>''', unsafe_allow_html=True)
#root > div:nth-child(1) > div > div > div > div > section > div > div:nth-child(1) > div > div:nth-child(6) > div
#root > div:nth-child(1) > div > div > div > div > section > div > div:nth-child(1) > div > div:nth-child(8) > div
if st.button('👉点此识别花朵'):
    try:
        model = re.findall(r'^\w+', model_select)[0]
        model = tf.keras.models.load_model(f'./models/flower_{model}.h5')
        img, label = img_process(img)
        st.write(label)
    except:
        st.write('>>>暂未识别出花朵')
        st.markdown('''<style>#root > div:nth-child(1) > div > div > div > div > section > div 
        > div:nth-child([last()]) > div > div:nth-child(8) {width: 571px ;background:rgba(220,220,192,0.9); 
        color:red; border-radius:5px;} <\style>''',
                        unsafe_allow_html=True)

st.markdown('''<style>#root > div:nth-child(1) > div > div > div > div > section > div > 
        div > div {background-size:100% 100% ;background:rgba(207,207,207,0.9); 
        color:red; border-radius:5px;} </style>''', unsafe_allow_html=True)
