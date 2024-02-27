import os
import torch
from scipy.io import loadmat

def DataInRam(CLA=False,HaLT=False,fiveF=False,PathToFiles = '.'):
	l = os.listdir(PathToFiles)

	for nf in l :
		if CLA & (nf[:3]=='CLA'):
			fpath = PathToFiles+'/'+nf
			load = loadmat(fpath)
			load = load['o'][0][0]
			
			marker = torch.FloatTensor(load[4])
			data = torch.FloatTensor(load[5])
			testname = nf[3:]
			try: 
				CLA_data_list
			except NameError:CLA_data_list=[]
			CLA_data_list.append({"marker":marker, "data":data, "testname":testname})
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
