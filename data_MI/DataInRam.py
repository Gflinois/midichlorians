import os,sys
import mne
import torch
from scipy.io import loadmat
import numpy as np
sys.path.insert(2,"./data_MI")
from translator import electrodes


def DataInRam(CLA=False, HaLT=False, fiveF=False, PathToFiles = '.', precutting=True, MNE=False, datatype="dico",l_chan = (False,range(22))):
	l = os.listdir(PathToFiles)
	nce = len(l_chan[1])


	
	
	for nf in l :
		if CLA & (nf[:3]=='CLA'):
			fpath = PathToFiles+'/'+nf
			load = loadmat(fpath)
			load = load['o'][0][0]
			
			markers = load[4]
			datas = load[5] * 10**(-6)
			testname = nf[3:]
			if not MNE :
				markers = torch.LongTensor(markers)
				datas = torch.FloatTensor(datas)
			if precutting and not MNE :
				for j in range(4):
					idx = np.where(markers == j)[0]
					i=0
					while i<len(idx)-1 :
						start = idx[i]
						while idx[i]+1==idx[i+1] and i<len(idx)-2:
							i+=1
						end = idx[i]
						marker = torch.nn.functional.one_hot(markers[start],num_classes=4) 
						local_datas = datas[start:start+npts]
						try: 
							CLA_data_list
						except NameError:CLA_data_list=[]
						if datatype == "dico":
							CLA_data_list.append({"marker":marker, "data":local_datas, "testname":testname})
						if datatype == "tuple" or datatype == "Dataloader":
							CLA_data_list.append((local_datas,marker))
						i+=1
						
			elif MNE :
				datas = datas.reshape([datas.shape[1],datas.shape[0]])
				datas = np.pad(datas, [(0, 1), (0, 0)], mode='constant')
				ch_names = list(electrodes.keys())[:datas.shape[0]-1]
				ch_names.append('STI 014')
				ch_types = {}
				for chn in ch_names:
					ch_types[chn] = "eeg"
				ch_types['STI 014'] = "stim"
				sfreq = load[2][0][0]
				info = mne.create_info(ch_names,sfreq)
				markers = torch.LongTensor(load[4])
				events = []
				for j in range(4):
					idx = np.where(markers == j)[0]
					i=0
					while i<len(idx)-1 :
						if j == 0:
							start = idx[i+200]
						else:
							start = idx[i-30]
						while idx[i]+1==idx[i+1] and i<len(idx)-2:
							i+=1
						end = idx[i]
						#print("j = ",j,"start = ",start,"end = ",end)
						events.append([start, 0,j+1])
						i+=1
				events = sorted(events, key=lambda event: event[0])
				
				#filt_passhaut = mne.filter.create_filter(datas[:nce], sfreq,15,None)
				#filt_coupebande = mne.filter.create_filter(datas[:nce], sfreq,90,30)
				#datas = np.convolve(datas,filt_passhaut)
				
				
				raw = mne.io.RawArray(data = datas,info = info)
				raw.set_channel_types(ch_types)
				raw.add_events(events)
				
				return raw
				
				treated = raw.copy()
				treated.load_data()
				treated.filter(1,None) #highpass
				#treated.filter(90,30)#notch
				treated.filter(None,30)#lowpass
				

				treated.compute_psd(fmax=50,picks="eeg")#.plot(picks="eeg",exclude="bads")

				ica = mne.preprocessing.ICA(n_components=4, random_state=97, max_iter=800)
				ica.fit(raw,picks="eeg")
				
				ica.apply(treated)
				
				try :
					event_dict = {"nothing": 1, "LH": 2, "RH": 3, "O": 4}
					npts = int(0.6*sfreq)
					epochs.equalize_event_counts(["LH", "RH", "O", "nothing"]) 
					O_epochs = epochs["O"]
					O_d = O_epochs.get_data(copy=True)
					O=True
				except:
					event_dict = {"nothing": 1, "LH": 2, "RH": 3}
					epochs = mne.Epochs(treated, events, event_id=event_dict, tmin=-0.2, tmax=0.4, preload=True)
					npts = int(0.6*sfreq)
					O=False
				LH_epochs = epochs["LH"]
				RH_epochs = epochs["RH"]
				nothing_epochs = epochs["nothing"]
				RH_d = RH_epochs.get_data(copy=True)
				LH_d = LH_epochs.get_data(copy=True)
				nothing_d = nothing_epochs.get_data(copy=True)
				
				
				#debugging#################################################################################
				lh = LH_epochs.average()
				n = nothing_epochs.average()
				diff = mne.combine_evoked((lh,-n),weights="equal")
				mne.viz.plot_events(events, event_id=event_dict, sfreq=sfreq)
				lh.plot()
				n.plot()
				diff.plot()
				dd["LH"].plot_psd() ==  dd["nothing"].plot_psd()
				###########################################################################################
				return 
				try: 
					CLA_data_list
				except NameError:CLA_data_list=[]
				
				if not precutting : 
					CLA_data_list.append(epochs)
				
				elif datatype == "dico":
					marker = torch.nn.functional.one_hot(torch.LongTensor([2]),num_classes=4)
					for  d in RH_d:
						CLA_data_list.append({"marker":marker, "data":local_datas, "testname":testname})
					marker = torch.nn.functional.one_hot(torch.LongTensor([1]),num_classes=4)
					for  d in LH_d:
						local_datas = torch.FloatTensor(d[:nce])
						CLA_data_list.append({"marker":marker, "data":local_datas, "testname":testname})
					if O:
						marker = torch.nn.functional.one_hot(torch.LongTensor([3]),num_classes=4)
						for  d in O_d:
							local_datas = torch.FloatTensor(d[:nce])
							CLA_data_list.append({"marker":marker, "data":local_datas, "testname":testname})
					marker = torch.nn.functional.one_hot(torch.LongTensor([0]),num_classes=4)
					for  d in nothing_d:
						local_datas = torch.FloatTensor(d[:nce])
						CLA_data_list.append({"marker":marker, "data":local_datas, "testname":testname})
						
				elif datatype == "tuple" or datatype == "Dataloader":
					marker = torch.nn.functional.one_hot(torch.LongTensor([2]),num_classes=4)
					for  d in RH_d:
						local_datas = torch.FloatTensor(d[l_chan[1],:npts]).reshape([npts,nce])
						CLA_data_list.append((local_datas,marker))

					marker = torch.nn.functional.one_hot(torch.LongTensor([1]),num_classes=4)
					for  d in LH_d:
						local_datas = torch.FloatTensor(d[l_chan[1],:npts]).reshape([npts,nce])
						CLA_data_list.append((local_datas,marker))
					if O:
						marker = torch.nn.functional.one_hot(torch.LongTensor([3]),num_classes=4)
						for  d in O_d:
							local_datas = torch.FloatTensor(d[l_chan[1],:npts]).reshape([npts,nce])
							CLA_data_list.append((local_datas,marker))
					marker = torch.nn.functional.one_hot(torch.LongTensor([0]),num_classes=4)
					for  d in nothing_d:
						local_data = torch.FloatTensor(d[l_chan[1],:npts]).reshape([npts,nce])
						CLA_data_list.append((local_datas,marker))
					
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
			if torch.cuda.is_available():
				for i in range(len(Train)):
					Train[i] = (Train[i][0].cuda(),Train[i][1].cuda())
				for i in range(len(Val)):
					Val[i] = (Val[i][0].cuda(),Val[i][1].cuda())
			dataset_train=torch.utils.data.DataLoader(Train,batch_size=500,shuffle=True)
			dataset_val=torch.utils.data.DataLoader(Val,batch_size=50,shuffle=True)
			dataset_test=torch.utils.data.DataLoader(Val,batch_size=1,shuffle=True)
			
			CLA_data = {"Train" : dataset_train,"Validation":dataset_val,"Test":dataset_test}
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
