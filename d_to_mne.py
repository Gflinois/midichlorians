import os,sys,torch
from scipy.io import loadmat
import numpy as np
import mne
import matplotlib.pyplot as plt
sys.path.insert(1,"./NN")
sys.path.insert(2,"./data_MI")
from translator import electrodes


PathToFiles = "./data_MI"
l = os.listdir(PathToFiles)
fpath = PathToFiles+"/"+l[0]
load = loadmat(fpath)
load = load['o'][0][0]

data = load[5]
data = data.reshape([22,data.shape[0]])
data = np.pad(data, [(0, 1), (0, 0)], mode='constant')




ch_names = list(electrodes.keys())[:22]
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
		start = idx[i]
		while idx[i]+1==idx[i+1] and i<len(idx)-2:
			i+=1
		end = idx[i]
		events.append([start, 0,j])
		i+=1
events = sorted(events, key=lambda event: event[0])



raw = mne.io.RawArray(data = data,info = info)
raw.set_channel_types(ch_types)
raw.add_events(events)

treated = raw.copy()
treated.load_data()

filt_passhaut = mne.filter.create_filter(data[:22], sfreq,0.1,None)

filt_coupebande = mne.filter.create_filter(data[:22], sfreq,70,40)





treated.compute_psd(fmax=50,picks="eeg").plot(picks="eeg",exclude="bads")



ica = mne.preprocessing.ICA(n_components=4, random_state=97, max_iter=800)
ica.fit(raw,picks="eeg")
ica.apply(treated)


event_dict = {
    "nothing": 0,
    "LH": 1,
    "RH": 2,
    "O": 3
}

fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw.info["sfreq"], first_samp=raw.first_samp)

reject_criteria= dict(eeg = 150e-6)

epochs = mne.Epochs(
    treated,
    events,
    event_id=event_dict,
    tmin=-0.2,
    tmax=0.8,
    #reject=reject_criteria,
    preload=True,
)

fig = epochs.plot()


