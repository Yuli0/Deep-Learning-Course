 #coding=utf-8
from PIL import Image
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

img=Image.open("lena.tiff")
img_r,img_g,img_b=img.split()


fig = plt.figure()
st = fig.suptitle("图像基本操作", fontsize="20",color="blue")
  
img_sf=img_r.resize((50,50))
plt.subplot(221)
plt.axis("off")
plt.title("R-缩放",fontsize=14) 
plt.imshow(img_sf,cmap="gray")


plt.subplot(222)
img_jx=img_g.transpose(Image.FLIP_LEFT_RIGHT)
img_xz=img.transpose(Image.ROTATE_270)
plt.axis("off")
plt.title("G-镜像+旋转",fontsize=14)
plt.imshow(img_xz,cmap="gray")

plt.subplot(223)
plt.axis("off")
plt.title("B-裁剪",fontsize=14)
img_cj=img.crop((0,0,150, 150))
plt.imshow(img_cj,cmap="gray")

 
img_rbg=Image.merge("RGB",[img_r,img_b,img_g])
img_rbg.save("test.png")

plt.subplot(224)
plt.axis("off")
plt.title("RGB",fontsize=14)
plt.imshow(img_rbg )

# plt.subtitle("图像基本操作",fontsize=20,color="blur")
plt.show()
