import os
import torch
from scipy.io import loadmat
import numpy as np

def DataInRam(CLA=False,HaLT=False,fiveF=False,PathToFiles = '.',precutting=True,datatype="dico"):
	l = os.listdir(PathToFiles)

	for nf in l :
		if CLA & (nf[:3]=='CLA'):
			fpath = PathToFiles+'/'+nf
			load = loadmat(fpath)
			load = load['o'][0][0]
			
			
			markers = torch.LongTensor(load[4])
			datas = torch.FloatTensor(load[5])
			testname = nf[3:]
			
			
			if precutting :
				for j in range(4):
					idx = np.where(markers == j)[0]
					i=0
					while i<len(idx)-1 :
						start = idx[i]
						while idx[i]+1==idx[i+1] and i<len(idx)-2:
							i+=1
						end = idx[i]
						marker = torch.nn.functional.one_hot(markers[start],num_classes=4) 
						local_datas = datas[start:start+200]
						try: 
							CLA_data_list
						except NameError:CLA_data_list=[]
						if datatype == "dico":
							CLA_data_list.append({"marker":marker, "data":local_datas, "testname":testname})
						if datatype == "tuple" or datatype == "Dataloader":
							CLA_data_list.append((local_datas,marker))
						i+=1
						
				

				
			
			else :
				#excluding uninteresting data to be able to onehot it
				for j in [99,92,91,90] :
					idx = np.where(markers == j)[0]
					#print("will remove ",len(idx)," values for ",j)
					markers = np.delete(markers,idx,0)
					datas = np.delete(datas,idx,0)
					#print("removed values for ",j)
				
				marker = torch.nn.functional.one_hot(marker,num_classes=4)
				
				try: 
					CLA_data
				except NameError:CLA_data=[]
				CLA_data_list.append({"marker":markers, "data":datas, "testname":testname})
				
			print("loaded : ",fpath)
			
			

		if HaLT & (nf[:4]=='HaLT'):
			print(nf)
		if fiveF & (nf[:2]=='5F'):
			print(nf)
	
	
	if datatype == "Dataloader" :
		if CLA:
			np.random.shuffle(CLA_data_list)
			Train = CLA_data_list[ : len(CLA_data_list)*9//10]
			Val = CLA_data_list[len(CLA_data_list)*9//10 : ]
			
			dataset_train=torch.utils.data.DataLoader(Train,batch_size=100,shuffle=True)
			dataset_val=torch.utils.data.DataLoader(Val,batch_size=100,shuffle=True)
			
			CLA_data = {"Train" : dataset_train,"Validation":dataset_val}
		if HaLT:
			print("pas pret")
		if fiveF:
			print("pas pret")
	else : 
		if CLA:
			CLA_data = CLA_data_list
		if HaLT:
			print("pas pret")
		if fiveF:
			print("pas pret")			
	
	dico_data = dict()
	dico_data["CLA"]  = CLA_data   if CLA   else None
	dico_data["HalT"] = Halt_data  if HaLT  else None
	dico_data["5F"]   = fiveF_data if fiveF else None

	return dico_data



if __name__ == '__main__':
	dd=DataInRam(CLA =True)
	print(dd["CLA"])
