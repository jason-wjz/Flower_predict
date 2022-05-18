import streamlit as st
import matplotlib.pyplot as plt
from streamlit.elements.image import image_to_url
import tensorflow as tf
import numpy as np

st.set_page_config(
    page_title='模型建立和训练',    #页面标题
    page_icon='🍁',          #图标
    layout='wide',                #页面布局
    initial_sidebar_state="auto"  #侧边栏
)
image_url = 'https://i.gifer.com/embedded/download/816Q.gif'
st.markdown('''<style>.css-fg4pbf{background-image:url(''' + image_url + ''');background-size:100% 100%;background-attachment:fixed;}</style>
''', unsafe_allow_html=True)

st.markdown('''<h1 align="center" style="color:rgb(240,248,255)" > ③模型建立和训练 </h1>''', unsafe_allow_html=True)
st.markdown('''<style>#root > div:nth-child(1) > div > div > div > div > section > div > 
        div > div {background-size:100% 100% ;background:rgba(207,207,207,0.9); 
        color:red; border-radius:5px;} </style>''', unsafe_allow_html=True)
st.sidebar.subheader('模型参数')
st.markdown('''<style>#root > div:nth-child(1) > div > div > div > div > section.css-1lcbmhc.e1fqkh3o3 > div.css-1adrfps.e1fqkh3o2
                {width: 200px; background:rgba(240,248,255,0.7);}</style>''', unsafe_allow_html=True)
st.sidebar.subheader(' ')
with st.sidebar.expander('点击展示模型图像'):
    st.image('./images/model.png')
l2 = st.sidebar.slider('设置L2参数：',min_value=0.01, max_value=0.1)
dropout = st.sidebar.slider('设置Dropout参数：',min_value=0.1, max_value=0.5)
epo = st.sidebar.slider('设置训练次数：',min_value=1, max_value=15)
bat = st.sidebar.slider('设置batch大小：',min_value=1, max_value=128)
tap = st.sidebar.radio('是否加入验证集',['是','否'])

st.markdown('''<h5 align="center" style="color:rgb(0,0,0)" > Xception </h5>''', unsafe_allow_html=True)
@st.cache(allow_output_mutation=True)
def load_data():
    x_train = tf.constant(np.load('./data/x_train.npy')[:1000, :, :, :], dtype=tf.float32)
    y_train = tf.constant(np.load('./data/y_train.npy')[:1000], dtype=tf.float32)
    x_test = tf.constant(np.load('./data/x_test.npy')[:1000, :, :, :], dtype=tf.float32)
    y_test = tf.constant(np.load('./data/y_test.npy')[:1000], dtype=tf.float32)
    x_valid = tf.constant(np.load('./data/x_valid.npy')[:1000, :, :, :], dtype=tf.float32)
    y_valid = tf.constant(np.load('./data/y_valid.npy')[:1000], dtype=tf.float32)
    outsize = len(set(y_train.numpy()))
    return outsize, x_train, y_train, x_test, y_test, x_valid, y_valid
outsize, x_train, y_train, x_test, y_test, x_valid, y_valid = load_data()

@st.cache(allow_output_mutation=True)
def ten(l2_num, dropout_size):
    tf.keras.backend.clear_session()
    base_model = tf.keras.applications.xception.Xception(weights='imagenet', include_top=False, input_shape=(80, 80, 3))
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_num))(x)
    x = tf.keras.layers.Dropout(dropout_size)(x)
    y = tf.keras.layers.Dense(units=outsize, activation=tf.keras.activations.softmax)(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=y)
    for layer in base_model.layers[:-3]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics='accuracy')
    return model
model_Xception = ten(l2, dropout)
if tap == '是':
    hist = model_Xception.fit(x_train, y_train, epochs=epo, batch_size=bat, validation_data=(x_valid, y_valid))
    loss, val_loss = hist.history['loss'], hist.history['val_loss']
    acc, val_acc = hist.history['accuracy'], hist.history['val_accuracy']
else:
    hist = model_Xception.fit(x_train, y_train, epochs=epo, batch_size=bat)
    loss, acc = hist.history['loss'], hist.history['accuracy']

plt.style.use('seaborn')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# plt.plot(loss); plt.plot(val_loss)
# plt.legend(['训练集','验证集'])
# plt.subplot(1, 2, 2)
# plt.plot(acc); plt.plot(val_acc)
# plt.legend(['训练集','验证集']); plt.title('正确率')
# plt.show()
# col1, col2 = st.columns(2)
fig, [ax1, ax2] = plt.subplots(1, 2)  # 一个 叫 fig 的图表, 有 ax1 和 ax2 两个子图 （subplot输入的)
ax1.plot(loss)
ax2.plot(acc)
if tap == '是':
    ax1.plot(val_loss)
    ax1.legend(['训练集','验证集'])
    ax2.plot(val_acc)
    ax2.legend(['训练集','验证集'])
ax1.set_title('损失函数')  # 设置图表的 title
ax2.set_title('正确率')  # 设置图表的 title
st.pyplot(fig)  # 展示

st.markdown('''<style>#root > div:nth-child(1) > div > div > div > div > section > div > 
                div:nth-child(1) > div > div:nth-child(6) > div > div > div > img {position: absolute;}</style>''', unsafe_allow_html=True)