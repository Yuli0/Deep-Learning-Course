import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


load_data=np.loadtxt("商品房销售数据表.csv",delimiter=",",skiprows=1)



plt.figure(figsize=(5,5))
plt.scatter(load_data[:,1],load_data[:,2],color="red")
plt.xlabel("面积(平方米)",fontsize="14")
plt.ylabel("价格(万元)",fontsize="14")
plt.title("商品房销售记录",fontsize="16",color="blue")
plt.tight_layout()
plt.show()