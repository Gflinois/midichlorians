import os
import torch
from scipy.io import loadmat
import numpy as np

def DataInRam(CLA=False,HaLT=False,fiveF=False,PathToFiles = '.',prebatching=True):
	l = os.listdir(PathToFiles)

	for nf in l :
		if CLA & (nf[:3]=='CLA'):
			fpath = PathToFiles+'/'+nf
			load = loadmat(fpath)
			load = load['o'][0][0]
			
			
			markers = torch.LongTensor(load[4])
			datas = torch.FloatTensor(load[5])
			testname = nf[3:]
			
			
			if prebatching :
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
						CLA_data_list.append({"marker":marker, "data":local_datas, "testname":testname})
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
					CLA_data_list
				except NameError:CLA_data_list=[]
				CLA_data_list.append({"marker":markers, "data":datas, "testname":testname})
			
			
				
				
				
				
				
			print("loaded : ",fpath)
			

		if HaLT & (nf[:4]=='HaLT'):
			print(nf)
		if fiveF & (nf[:2]=='5F'):
			print(nf)
		
	
	dico_data = dict()
	dico_data["CLA"]  = CLA_data_list   if CLA   else None
	dico_data["HalT"] = Halt_data_list  if HaLT  else None
	dico_data["5F"]   = fiveF_data_list if fiveF else None

	return dico_data



if __name__ == '__main__':
	dd=DataInRam(CLA =True)
	print(dd["CLA"])
