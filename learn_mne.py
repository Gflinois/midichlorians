import mne,sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(2,"./data_MI")
from DataInRam import DataInRam 

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = (
    sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
)
raw = mne.io.read_raw_fif(sample_data_raw_file)
"""
l_chan =(True,range(22))#for the test
raw = DataInRam(CLA=True,PathToFiles="./data_MI",datatype="Dataloader",MNE=True,l_chan=l_chan)#for the test
"""
treated = raw.copy()
treated.load_data()
#raw.info is amazing

#print(raw.info)


#raw.compute_psd(fmax=50).plot(picks="data", exclude="bads")
#raw.plot(duration=10, n_channels=30)
#print(treated.info)


ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw)
ica.exclude = [1, 2]  # details on how we picked these are omitted here
#ica.plot_properties(raw, picks=ica.exclude)
ica.apply(treated)


#chan_idxs = [raw.ch_names.index("EEG 00"+str(i+1)) for i in range(8)]
chan_idxs = raw.ch_names
#raw.plot(order=chan_idxs,start=12,duration=4)
#treated.plot(order = chan_idxs,start=12,duration=4)
treated.plot(start=12,duration=4)

event_dict = {
    "auditory/left": 1,
    "auditory/right": 2,
    "visual/left": 3,
    "visual/right": 4,
    "smiley": 5,
    "buttonpress": 32,
}


#event_dict = {"nothing": 1, "LH": 2, "RH": 3, "O": 4}#for the test


events = mne.find_events(treated,stim_channel = "STI 014") 

print(events)

fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw.info["sfreq"], first_samp=raw.first_samp)


reject_criteria= dict(eeg = 150e-6)

epochs = mne.Epochs(
    treated,
    events,
    event_id=event_dict,
    tmin=-0.2,
    tmax=0.5,
    reject=reject_criteria,
    preload=True,
)


epochs.equalize_event_counts(["auditory/left", "auditory/right", "visual/left", "visual/right"]) 
aud_epochs = epochs["auditory"]
vis_epochs = epochs["visual"]
del raw, epochs  # free up memory

"""
epochs.equalize_event_counts(event_dict.keys()) #for the test
aud_epochs = epochs["LH"]#for the test
vis_epochs = epochs["O"]#for the test
"""


#aud_epochs.plot_image(picks=["EEG 0"+str(i+1) for i in range(9,22)])


aud_evoked = aud_epochs.average()
vis_evoked = vis_epochs.average()

mne.viz.plot_compare_evokeds(
    dict(auditory=aud_evoked, visual=vis_evoked),
    legend="upper left",
    show_sensors="upper right",
)
aud_evoked.plot_joint(picks="eeg")
#aud_evoked.plot_topomap(times=[0.0, 0.08, 0.1, 0.12, 0.2,0.3], ch_type="eeg")


plt.show()
