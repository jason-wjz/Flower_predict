import streamlit as st
import matplotlib.pyplot as plt
from streamlit.elements.image import image_to_url

st.set_page_config(
    page_title='数据观察和增强',    #页面标题
    page_icon='🍁',          #图标
    layout='wide',                #页面布局
    initial_sidebar_state="auto"  #侧边栏
)

image_url = 'https://i.gifer.com/embedded/download/816Q.gif'
st.markdown('''<style>.css-fg4pbf{background-image:url(''' + image_url + ''');background-size:100% 100%;background-attachment:fixed;}</style>
''', unsafe_allow_html=True)

st.markdown('''<h1 align="center" style="color:rgb(240,248,255)" > ①数据观察和增强 </h1>''', unsafe_allow_html=True)
st.markdown('''<style>#root > div:nth-child(1) > div > div > div > div > section > div > 
        div > div {background-size:100% 100% ;background:rgba(207,207,207,0.9); 
        color:red; border-radius:5px;} </style>''', unsafe_allow_html=True)

st.subheader('1、数据观察 & 数据获取')
st.subheader(' ')
st.subheader(' ')
st.markdown('''<p align="center" style="color:rgb(0,0,0)">对每一类花拥有的图片数量进行统计,发现原始数据很不均衡</p>''', unsafe_allow_html=True)
col1, col2 = st.columns(2)  # 分成两列
with col1:
    st.image('./images/before_clawer.png')
    st.markdown('''<h5 align="center" style="color:rgb(0,0,0)" > 原始数据集 8200张左右 </h5>''', unsafe_allow_html=True)
with col2:
    st.image('./images/after_clawer.png')
    st.markdown('''<h5 align="center" style="color:rgb(0,0,0)" > 爬取图片后 12000张左右 </h5>''', unsafe_allow_html=True)
st.subheader(' ')
st.subheader(' ')
st.markdown('''<p align="center" style="color:rgb(0,0,0)">通过selenium库控制浏览器模拟人的输入，点击。获取图片网址后下载图片</p>''', unsafe_allow_html=True)
st.image('./images/clawer.jpg')
st.markdown('''<h5 align="center" style="color:rgb(0,0,0)" > 爬取图片 </h5>''', unsafe_allow_html=True)
st.header(' ')
st.header(' ')
st.header(' ')
st.header(' ')
st.subheader('2、数据清洗')
st.subheader(' ')
st.markdown('''<p align="center" style="color:rgb(0,0,0)">部分网图的花会出现手绘/拼图的情况，需要手动剔除</p>''', unsafe_allow_html=True)
st.image('./images/img_process2.png')
st.markdown('''<h5 align="center" style="color:rgb(0,0,0)" > 数据清洗 删掉了约200张 </h5>''', unsafe_allow_html=True)
st.header(' ')
st.header(' ')
st.header(' ')
st.header(' ')
st.subheader('3、数据增强')
st.markdown('''<p align="center" style="color:rgb(0,0,0)">自定义四个图像处理函数，
再用random模块中的random.choice进行随机选择，实现随机处理</p>''', unsafe_allow_html=True)
st.markdown('''<p align="center" style="color:rgb(0,0,0)">花卉与人脸相比，数据增强以图像旋转、放大、裁剪为主</p>''', unsafe_allow_html=True)
st.image('./images/img_process.jpg')
st.markdown('''<h5 align="center" style="color:rgb(0,0,0)" > 数据增强思路图解 </h5>''', unsafe_allow_html=True)
st.markdown('''<p align="center" style="color:rgb(0,0,0)">不宜增加噪声（模糊/增加像素点），花卉本身颜色就很复杂</p>''', unsafe_allow_html=True)
st.header(' ')
st.image('./images/after_process.png')
st.markdown('''<h5 align="center" style="color:rgb(0,0,0)" > 数据增强后 20000张左右 </h5>''', unsafe_allow_html=True)
st.header(' ')
st.header(' ')
st.markdown('[>>>>>>>>>>>>>>>>>>>>点此跳转至：2.模型建立和训练](http://localhost:8503)')

st.markdown('''<style>#root > div:nth-child(1) > div > div > div > div > section > div > 
                div:nth-child(1) > div > div:nth-child(6) > div > div > div > img {position: absolute;}</style>''', unsafe_allow_html=True)


