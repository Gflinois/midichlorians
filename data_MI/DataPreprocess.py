import torch, pyedflib, mne, os, sys
from scipy.io import loadmat
import numpy as np
sys.path.insert(2,"../eeg-motor-movementimagery-dataset-1.0.0/files")
from translator import electrodes



def Processall(PathToFiles = '../eeg-motor-movementimagery-dataset-1.0.0/files'):
	ld = os.listdir(PathToFiles)
	ltask = ["03","04","07","08","11","12"]
	lfail = []
	for d in ld :
		l = os.listdir(PathToFiles+'/'+d)
		for nf in l:
			if nf[-1]=='f' and (nf[5:7] in ltask):
				fpath = PathToFiles+'/'+d+'/'+nf
				if not ProcessFile(fpath,nf):
					lfail.append(fpath)
	return lfail
		
		
		
		
def ProcessFile(fpath,nf):
	print("loading : ",fpath)
	try :
		fileID=nf[:-4]

		load = pyedflib.EdfReader(fpath)

		n_sig = load.signals_in_file
		markers = load.readAnnotations()
		datas = np.array([load.readSignal(i) for i in range(n_sig)]) * 10**(-3)
		ch_names = load.getSignalLabels()
		ch_names.append('STI 014')
		ch_types = {}
		for chn in ch_names:
			ch_types[chn] = "eeg"
		ch_types['STI 014'] = "stim"
		sfreq = 160
		info = mne.create_info(ch_names,sfreq)

		events = np.array(load.readAnnotations())
		datas = np.pad(datas, [(0, 1), (0, 0)], mode='constant')
		for i in range (len(events[0])):
			events[0,i] = round(float(events[0,i])*sfreq)
			events[2,i] = int((events[2,i])[-1])+1
			events[1,i] = '0'
		events = torch.tensor(events.astype('int32')).t()
		print( events)
		Ws = mne.time_frequency.morlet(sfreq, range(1,36))#generating the ws based on morlet for cwt for freqs of 1-35Hz

		raw = mne.io.RawArray(data = datas,info = info)


		raw.set_channel_types(ch_types)
		raw.add_events(events)


		event_dict = {"nothing": 1, "LH": 2, "RH": 3}
		epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=0.8, preload=True)
		epochs.equalize_event_counts(["LH", "RH", "nothing"]) 

		freqs = np.linspace(1,35,35)
		n_cycles = freqs / 2.0

		epochs = epochs.compute_tfr(method = 'morlet',freqs =freqs, n_cycles=n_cycles ,picks = 'eeg',n_jobs=-1)
		LH_epochs = epochs["LH"]
		RH_epochs = epochs["RH"]
		nothing_epochs = epochs["nothing"]
		RH_d = RH_epochs.get_data()
		np.save(str("./data_MI/npy/"+fileID+'_RH.npy'), RH_d, allow_pickle=True)
		LH_d = LH_epochs.get_data()
		np.save(str("./data_MI/npy/"+fileID+'_LH.npy'), LH_d, allow_pickle=True)
		nothing_d = nothing_epochs.get_data()
		np.save(str("./data_MI/npy/"+fileID+'_nothing.npy'), nothing_d, allow_pickle=True)
		print("loaded : ",fpath)
		return True
	except :
		print("Failed to load : ",fpath)
		return False

def PlotHeatmap(data,vmax = 1):
	heatmap=data
	import matplotlib
	import matplotlib.pyplot as plt
	cmap=matplotlib.cm.get_cmap("Reds")
	plt.imshow(heatmap,cmap="Reds",vmax = vmax)
	cb = plt.colorbar()
	plt.show()
		



if __name__ == '__main__':
	print("list fail: ",Processall())
	print("done!!!")
