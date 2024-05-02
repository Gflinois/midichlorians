import torch
import os
import pytorch_lightning
import numpy as np
import mne


class neur_net_struct(pytorch_lightning.LightningModule):
	def __init__(self,Batchsize=1, freqs=35,Cv_Cout=44,CV_Wf=3, N_NEURONE=600, LSTM_LAYER=2, DROPSIZE=0,Numb_Of_Class=4,nce=22):
		#def parametres
		self.nce = nce #number of sensors
		self.freqs = freqs
		self.LSTM_LAYER=LSTM_LAYER
		self.N_NEURONE=N_NEURONE
		self.noc = Numb_Of_Class
		
		
		#création d'un réseau 
		super().__init__()
		
		self.conv1 = torch.nn.Conv2d(nce,Cv_Cout, (freqs//5+1,CV_Wf)) #Hout = freqs-(freqs//5+1)+1 = 35-8+1 = 28
		self.conv2 = torch.nn.Conv2d(Cv_Cout,2*Cv_Cout, (freqs//5+1,CV_Wf)) #Hout = freqs-2*(freqs//5+1)+2 = 35-16+2 = 21
		self.conv3 = torch.nn.Conv2d(2*Cv_Cout,4*Cv_Cout, (freqs//5+1,CV_Wf)) #Hout = freqs-3*(freqs//5+1)+3 = 35-24+3 = 14
		self.conv4 = torch.nn.Conv2d(4*Cv_Cout,6*Cv_Cout, (freqs//5+1,CV_Wf)) #Hout = freqs-4*(freqs//5+1)+4 = 35-32+4 = 7
		self.conv5 = torch.nn.Conv2d(6*Cv_Cout,8*Cv_Cout, (freqs//5,CV_Wf)) #Hout = freqs-5*(freqs//5+1)+6 = 35-40+6 = 1; Wout = Win-5*Wf+5
		
		#shape(c5) = [batchsize,8*Cv_Cout,freqs-5*(freqs//5+1)+6,Win-5*Wf+5] = [batchsize,352,1,190]
		
		
		self.lstm = torch.nn.LSTM(8*Cv_Cout, N_NEURONE, LSTM_LAYER, batch_first=True)
		#inputsize = Co*Conv_size_out, hiddensize = nb features to extract at each time ste (we will use the last time step's feature to predict the class),num of lstms, 
		self.drop = torch.nn.Dropout(DROPSIZE)
		self.Big = torch.nn.Linear( N_NEURONE,  N_NEURONE//2)
		self.drop1 = torch.nn.Dropout(DROPSIZE)
		self.Inter = torch.nn.Linear( N_NEURONE//2, N_NEURONE//4)
		self.Fin = torch.nn.Linear(N_NEURONE//4, Numb_Of_Class)
		
		
		
		
		
	def forward(self,data):
		#freqs = np.linspace(1,35,35)
		#n_cycles = freqs / 2.0	
		#data = torch.FloatTensor(mne.time_frequency.tfr_array_morlet(data = data.cpu(),sfreq = 200,freqs = freqs,n_cycles=n_cycles ,n_jobs=-1)).cuda()
	
		batchsize = data.shape[0]  if len(data.shape)>=3 else 1
		nb_of_time = data.shape[1] if len(data.shape)>=3 else data.shape[0]
		#nb_of_time = data.shape[3] if len(data.shape)>=3 else data.shape[0]
		nce = self.nce
		data = torch.reshape(data,[batchsize,nce,self.freqs,nb_of_time]).cuda()
		c1 = self.conv1(data)
		c2 = self.conv2(c1)
		c3 = self.conv3(c2)
		c4 = self.conv4(c3)
		c5 = self.conv5(c4)
		c5 = torch.reshape(c5,[c5.shape[0],c5.shape[3],c5.shape[1]])
		l,mem = self.lstm(c5)
		s = l[:,-1,:]
		t = torch.reshape(s,[l.shape[0],1,l.shape[2]])
		m = torch.nn.functional.relu(t)
		b = self.Big(m)
		i = self.Inter(b)
		f = self.Fin(i)
		
		result = 2*torch.sigmoid(f)-1
		return result
		
	


	def configure_optimizers(self):
		optimizer=torch.optim.Adam(self.parameters())
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, factor=0.5, patience=5)
		return {
			'optimizer': optimizer,
			'lr_scheduler': scheduler,
			'monitor': 'train_loss'
			}


	
	def training_step(self,batch,batch_idx):
		x,y = batch
		y=torch.tensor(y,dtype=torch.float32).cuda()
		r = self(x)
		loss = torch.nn.functional.mse_loss(r, y)
		self.log('train_loss', loss)
		return loss
		



	def validation_step(self, batch,batch_idx):
		x,y = batch
		y=torch.tensor(y,dtype=torch.float32).cuda()
		r = self(x)
		loss = torch.nn.functional.mse_loss(r, y)
		
		self.log('val_loss', loss)
		self.log('hp_metric', loss)


	def test_loop(self, dataloader):
		size = len(dataloader.dataset)
		num_batches = len(dataloader)
		test_loss, correct = 0, 0
		self.pred_table=torch.zeros(self.noc,self.noc)
		
		with torch.no_grad():
			for X, y in dataloader:
				y=torch.tensor(y,dtype=torch.float32).cuda()
				pred = self(X)
				test_loss += torch.nn.functional.mse_loss(pred, y).item()
				correct += int(pred.argmax()== y.argmax())
				self.pred_table[y.argmax()][pred.argmax()]+=1
		for i in range(self.noc):
			ti=sum(self.pred_table[i])
			for j in range(self.noc):
				self.pred_table[i][j]=self.pred_table[i][j].item()*100//ti.item() if ti!=0 else  0 

		test_loss /= size
		print("Test Error: \n Accuracy: "+str(100*correct/size)+f"%, Avg loss: {test_loss:>8f} \n")
		return 100*correct/size
