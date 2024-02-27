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
print(nnc2l(d1).size())
dlim=d1[:1000,:]
print(dlim.size())
print(nnc2(dlim).size())
