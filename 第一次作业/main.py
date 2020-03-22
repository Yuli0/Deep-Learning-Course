import math

a=int(input(""))
b=int(input(""))
c=int(input(""))
jie=int(b*b-4*a*c)
if jie<0:
    print("无解")
elif jie==0:
    x=-b/2/a
    print(x)
else:
    x1=(-b+math.sqrt(jie))/2/a
    x2=(-b-math.sqrt(jie))/2/a
    print("%d %d"%(x1,x2))
