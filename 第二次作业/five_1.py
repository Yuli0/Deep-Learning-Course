import numpy as np

np.random.seed(612)
a=np.random.random(1000)
n=int(input("请输入一个1-100之间的整数："))
print("序号\t索引值\t随机数")
j=1
i=n

while(i-1<1000):
    print("%d\t%d\t%f\t"%(j,i,a[i-1]))
    j+=1
    i+=n