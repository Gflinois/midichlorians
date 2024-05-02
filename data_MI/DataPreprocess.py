import os,sys
import mne
import torch
from scipy.io import loadmat
import numpy as np
sys.path.insert(2,"./data_MI")
from translator import electrodes



def Processall(CLA=False, HaLT=False, fiveF=False, PathToFiles = '.'):
	l = os.listdir(PathToFiles+'/mat/')
	for nf in l :
		ProcessFile(cla=CLA,halt=HaLT,fivef=fiveF,PathToFiles=PathToFiles,nf=nf)
		
		fpath=str(PathToFiles+'/npy/'+nf)
	return True
		
		
		
		
def ProcessFile(cla=False, halt=False, fivef=False, PathToFiles = '.',nf = '', l_chan = (False,range(22))):
	fpath=str(PathToFiles+'/mat/'+nf)
	nfpath=str(PathToFiles+'/npy/'+nf[:-4])
	if cla & (nf[:3]=='CLA'):
		load = loadmat(fpath)
		load = load['o'][0][0]
		
		markers = load[4]
		datas = load[5] * 10**(-6)
		testname = nf[3:]
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
					start = idx[i]
				while idx[i]+1==idx[i+1] and i<len(idx)-2:
					i+=1
				end = idx[i]
				events.append([start, 0,j+1])
				i+=1
		events = sorted(events, key=lambda event: event[0])
		
		Ws = mne.time_frequency.morlet(sfreq, range(1,36))#generating the ws based on morlet for cwt for freqs of 1-35Hz

		#treated = mne.time_frequency.tfr.cwt(datas, Ws)
		#print(treated.shape)
		#datas = treated.get_data()
		raw = mne.io.RawArray(data = datas,info = info)
		
		
		raw.set_channel_types(ch_types)
		raw.add_events(events)
		
		try :
			event_dict = {"nothing": 1, "LH": 2, "RH": 3, "O": 4}
			epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=0.8, preload=True)
			epochs.equalize_event_counts(["LH", "RH", "O", "nothing"]) 
			
			freqs = np.linspace(1,35,35)
			n_cycles = freqs / 2.0
			
			epochs = epochs.compute_tfr(method = 'morlet',freqs =freqs, n_cycles=n_cycles ,picks = 'eeg',n_jobs=-1)
			O_epochs = epochs["O"]
			O_d = O_epochs.get_data(copy=True)
			np.save(str(nfpath+'_O.npy'), O_d, allow_pickle=True)
			O=True
		except:
			event_dict = {"nothing": 1, "LH": 2, "RH": 3}
			epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=0.8, preload=True)
			freqs = np.linspace(1,35,35)
			n_cycles = freqs / 2.0
			
			epochs = epochs.compute_tfr(method = 'morlet',freqs =freqs, n_cycles=n_cycles ,picks = 'eeg',n_jobs=-1)
			O=False
		LH_epochs = epochs["LH"]
		RH_epochs = epochs["RH"]
		nothing_epochs = epochs["nothing"]
		RH_d = RH_epochs.get_data()
		np.save(str(nfpath+'_RH.npy'), RH_d, allow_pickle=True)
		LH_d = LH_epochs.get_data()
		np.save(str(nfpath+'_LH.npy'), LH_d, allow_pickle=True)
		nothing_d = nothing_epochs.get_data()
		np.save(str(nfpath+'_nothing.npy'), nothing_d, allow_pickle=True)
		print("loaded : ",fpath)
		
		

	if halt & (nf[:4]=='HaLT'):
		print(nf)
	if fivef & (nf[:2]=='5F'):
		print(nf)

	return True

def PlotHeatmap(data,vmax = 10**(-8)):
	heatmap=data
	import matplotlib
	import matplotlib.pyplot as plt
	cmap=matplotlib.cm.get_cmap("Reds")
	plt.imshow(heatmap,cmap="Reds",vmax =vmax)
	cb = plt.colorbar()
	plt.show()
		



if __name__ == '__main__':
	Processall(CLA =True,PathToFiles = './data_MI')
	print("done!!!")
