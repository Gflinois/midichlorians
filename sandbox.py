import sys
import numpy

sys.path.insert(1,"./NN")
sys.path.insert(2,"./data_MI")


from Sig_Cv2d_Lstm import neur_net_struct
from DataInRam import DataInRam
from translator import translator


nnc2l=neur_net_struct()
nnc2=neur_net_struct()

dd = DataInRam(CLA=True,PathToFiles="./data_MI")
d1 = dd["CLA"][0]["data"]
"""
print(nnc2l(d1).size())
dlim=d1[:1000,:]
print(dlim.size())
print(nnc2(dlim).size())
"""


m1 = dd["CLA"][0]["marker"]
n=0
lh=0
rh=0
o=0
IRP=0
ISRP=0
EE=0
for i in m1 :
	n+=i[0][0]
	lh+=i[0][1]
	rh+=i[0][2]
	o+=i[0][3]
	IRP+=i[0][4]
	ISRP+=i[0][5]
	EE+=i[0][6]

print("repartition :\n\tnothing = ",n/m1.size()[0]*100,"\n\tLH = ",lh/m1.size()[0]*100,"\n\tRH = ",rh/m1.size()[0]*100,"\n\tO = ",o/m1.size()[0]*100,"\n\tIRP = ",IRP/m1.size()[0]*100,"\n\tISRP = ",ISRP/m1.size()[0]*100,"\n\tEE = ",EE/m1.size()[0]*100)		


i=0

while not m1[i][0][3]:
	i+=1	

start=i
while m1[i][0][3]:
	i+=1

length = i-start
		
print("length of a O = ",length)		


