import os,sys
import mne
import torch
from scipy.io import loadmat
import numpy as np
sys.path.insert(2,"./data_MI")
from translator import electrodes


def DataInRam(CLA=False, HaLT=False, fiveF=False, PathToFiles = '.'):
	l = os.listdir(PathToFiles+'/npy/')
	lmark = ["ng","LH","RH","_O"]
	nce = 22
	npts = 200
	freqs = 35
	CLA_data_list = []
	for nf in l[:10] :
		string_marker = nf[-6:-4]
		marker = torch.LongTensor([lmark.index(string_marker)])
		marker = torch.nn.functional.one_hot(marker,num_classes=4)
		d_l = np.load(PathToFiles+'/npy/'+nf,allow_pickle=True)
		for d in d_l:
			local_datas = torch.FloatTensor(d[range(nce),:,:npts]).reshape([npts,freqs,nce])
			CLA_data_list.append((local_datas,marker))
	if CLA:
		np.random.shuffle(CLA_data_list)
		Train = CLA_data_list[ : len(CLA_data_list)*9//10]
		Val = CLA_data_list[len(CLA_data_list)*9//10 : ]
		
		if torch.cuda.is_available():
			for i in range(len(Train)):
				Train[i] = (Train[i][0].cuda(),Train[i][1].cuda())
			for i in range(len(Val)):
				Val[i] = (Val[i][0].cuda(),Val[i][1].cuda())
		dataset_train=torch.utils.data.DataLoader(Train,batch_size=150,shuffle=True)
		dataset_val=torch.utils.data.DataLoader(Val,batch_size=5,shuffle=True)
		dataset_test=torch.utils.data.DataLoader(Val,batch_size=1,shuffle=True)
		
		CLA_data = {"Train" : dataset_train,"Validation":dataset_val,"Test":dataset_test}
	
	data = dict()
	data["CLA"] = CLA_data
	
	return data



if __name__ == '__main__':
	dd=DataInRam(CLA =True,PathToFiles = './data_MI')
	print(dd["CLA"])
