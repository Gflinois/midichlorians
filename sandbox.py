import sys
if not sys.warnoptions:
    import os, warnings
    warnings.simplefilter("ignore")
import numpy as np
import shutil
import pytorch_lightning
import torch
sys.path.insert(1,"./NN")
sys.path.insert(2,"./data_MI")


import Sig_Cv2d_Lstm
import Snap_Cv2d
from DataInRam import DataInRam
from translator import translator



import sys
if not sys.warnoptions:
    import os, warnings
    warnings.simplefilter("ignore")



class MetricsCallback(pytorch_lightning.Callback):
	def __init__(self):
		super().__init__()
		self.metrics = []

	def on_validation_end(self, trainer, pl_module):
		self.metrics.append(trainer.callback_metrics)




nnc2l=Sig_Cv2d_Lstm.neur_net_struct()
nnc2=Snap_Cv2d.neur_net_struct()





#this is for datatype = Dataloader


dd = DataInRam(CLA=True,PathToFiles="./data_MI",datatype="Dataloader",MNE=True)
Train = dd["CLA"]["Train"]
Val = dd["CLA"]["Validation"]
Test = dd["CLA"]["Test"]
print(len(Val)," validation batches")
print(len(Train)," training batches")




#this is the training

model = nnc2l

try :
	shutil.rmtree("lightning_logs")
except FileNotFoundError :
	print("File not suppressed")

checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(monitor="val_loss",mode='min')

trainer = pytorch_lightning.Trainer(
	logger=True,
	max_epochs=30,
	devices=1, accelerator="auto",
	callbacks=[pytorch_lightning.callbacks.LearningRateMonitor(logging_interval='step'), MetricsCallback(), pytorch_lightning.callbacks.ModelCheckpoint(monitor="val_loss",mode='min')]
	)

trainer.fit(model,Train, Val)


fle = os.listdir("lightning_logs/version_0/checkpoints")[0]

shutil.copyfile("lightning_logs/version_0/checkpoints/" + fle, "model.ckpt")

if model == nnc2:
	model= Snap_Cv2d.neur_net_struct.load_from_checkpoint("model.ckpt")
if model == nnc2l :
	print("worked")
	model= Sig_Cv2d_Lstm.neur_net_struct.load_from_checkpoint("model.ckpt")


model.eval()
s=model.test_loop(Test)

print(model.pred_table)


"""
#this is for datatype = tuple

dd = DataInRam(CLA=True,PathToFiles="./data_MI",datatype="tuple")

cla0 = dd["CLA"][0]
d1 = cla0[0]
m1 = cla0[1]

print(len(dd["CLA"]))
print(m1.size())
print(d1.size())
"""





"""
#this is for datatype = dico
dd = DataInRam(CLA=True,PathToFiles="./data_MI")

cla0 = dd["CLA"][0]
d1 = cla0["data"]
m1 = cla0["marker"]

print(len(dd["CLA"]))
print(m1.size())
print(d1.size())
"""

"""
print(nnc2l(d1).size())
dlim=d1[:1000,:]
print(dlim.size())
print(nnc2(dlim).size())

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

"""
