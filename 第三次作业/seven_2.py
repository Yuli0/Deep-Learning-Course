import tensorflow as tf
import matplotlib.pyplot as plt
import random as rd


plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
mnist=tf.keras.datasets.mnist
(train_x,train_y),(test_x,test_y)=mnist.load_data("mnist.npz")
# mnist = input_data.read_data_sets("mnist.npz",one_hot=True)

fig = plt.figure()
st = fig.suptitle("MNIST测试集样本",fontsize="20",color="red")
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
# 调整子图间距
plt.subplots_adjust(wspace =0.5, hspace =0.5)

for i in range(16):
    num=rd.randint(1,50000)
    plt.subplot(4,4,i+1)
    plt.axis("off")
    plt.imshow(train_x[num],cmap="gray")
    plt.title("标签值："+str(train_y[num]))

plt.show()

