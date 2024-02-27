import sys
import numpy

sys.path.insert(1,"./NN")
sys.path.insert(2,"./data_MI")


from Sig_Cv2d_Lstm import neur_net_struct
from DataInRam import DataInRam
from translator import translator


nn=neur_net_struct()


dd = DataInRam(CLA=True,PathToFiles="./data_MI")
d1 = dd["CLA"][0]["data"]
print(nn(d1).size())
