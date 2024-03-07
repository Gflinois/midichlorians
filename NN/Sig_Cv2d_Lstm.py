import torch
import os
import pytorch_lightning



class neur_net_struct(pytorch_lightning.LightningModule):
	def __init__(self,Batchsize=1, Cv_Cin=1,Cv_Cout=12,CV_Wf=3, N_NEURONE=16, LSTM_LAYER=1, DROPSIZE=0,Numb_Of_Class=4):
		
		#def parametres
		nce = 1
		self.LSTM_LAYER=LSTM_LAYER
		self.N_NEURONE=N_NEURONE
		self.noc = Numb_Of_Class
		#self.h0c0=(torch.zeros ([LSTM_LAYER,Batchsize,N_NEURONE]),torch.zeros([LSTM_LAYER,Batchsize,N_NEURONE]))
		
		
		#création d'un réseau 
		super().__init__()
		self.conv = torch.nn.Conv2d(Cv_Cin,Cv_Cout, (nce,CV_Wf)) #Cin,Co,(Hf=nce,Wf)
		self.lstm = torch.nn.LSTM(Cv_Cout, N_NEURONE, LSTM_LAYER, batch_first=True)#inputsize = Co*Conv_size_out,hiddensize = nb features to extract at each time ste (we will use the last time step's feature to predict the class),num of lstms, 
		self.drop = torch.nn.Dropout(DROPSIZE)
		self.Big = torch.nn.Linear( N_NEURONE,  N_NEURONE//2)
		self.drop1 = torch.nn.Dropout(DROPSIZE)
		self.Inter = torch.nn.Linear( N_NEURONE//2, N_NEURONE//4)
		self.Fin = torch.nn.Linear(N_NEURONE//4, Numb_Of_Class)
		
		
		self.conv = torch.nn.Conv2d(Cv_Cin,Cv_Cout, (nce,4))
		self.testl = torch.nn.LSTM(Cv_Cout, N_NEURONE, LSTM_LAYER, batch_first=True)
		self.test1 = torch.nn.Linear(N_NEURONE, N_NEURONE//2)
		self.test2 = torch.nn.Linear(N_NEURONE//2, N_NEURONE//4)
		self.test3 = torch.nn.Linear(N_NEURONE//4, Numb_Of_Class)
		
		
		
		
	def forward(self,data):
		
		batchsize = data.shape[0]  if len(data.shape)>=3 else 1
		nb_of_time = data.shape[1] if len(data.shape)>=3 else data.shape[0]
		nce = data.shape[2]        if len(data.shape)>=3 else data.shape[1]
		data = torch.reshape(data,[batchsize,1,nce,nb_of_time])
		c = self.conv(data)
		r = torch.nn.functional.relu(c)
		r = torch.reshape(c,[batchsize,c.shape[3],c.shape[1]])
		l,mem = self.lstm(r)
		s = l[:,-1,:]
		t = torch.reshape(s,[batchsize,s.shape[1]])
		m = torch.nn.functional.relu(t)
		b = torch.nn.functional.relu(self.drop(self.Big(m)))
		i = torch.nn.functional.relu(self.drop1(self.Inter(b)))
		f = self.Fin(i)
		
		
		
		###
		data = torch.reshape(data[:,0,0,40],[batchsize,1])
		result1 = torch.zeros(data.shape[0],1,1,4).cuda()
		for j in range(len(data[:,0])):
			i = data[j,0]
			result1[j,0] = (torch.nn.functional.one_hot(torch.LongTensor([int(i)]),num_classes=4)).cuda()
			
		
		c = self.conv(result1)
		c = torch.reshape(c,[c.shape[0],1,c.shape[1]])
		fl,mem = self.testl(c)
		f1 = self.test1(fl)
		f2 = self.test2(f1)
		f = self.test3(f2)
		###
		
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
		y=torch.tensor(y,dtype=torch.float32)
		r = self(x)
		loss = torch.nn.functional.mse_loss(r, y)
		self.log('train_loss', loss)
		return loss
		



	def validation_step(self, batch,batch_idx):
		x,y = batch
		y=torch.tensor(y,dtype=torch.float32)
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
				result1 = torch.zeros(X.shape[0],1,4).cuda()
				for j in range(len(X[:,40,0])):
					i = X[j,40,0]
					result1[j] = (torch.nn.functional.one_hot(torch.LongTensor([int(i)]),num_classes=4)).cuda()
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
	
	
	"""
	def CalculateGradmap (self,data):
		pred=self.forward(data)
		self.pred=pred
		grads=torch.autograd.grad(pred[:, pred.argmax().item()],self.conv_layer_output)
		pooled_grads=grads[0].mean((0,2,3))
		self.conv_layer_output=torch.nn.functional.relu(self.conv_layer_output.squeeze().detach())
		self.conv_layer_output[:,:] *= pooled_grads
		#heatmap=torch.sigmoid(self.conv_layer_output)
		#heatmap=heatmap
		heatmap=self.conv_layer_output
				
		
		return heatmap


	def PlotHeatmap(self,data):
		heatmap=self.CalculateGradmap(data)
		import matplotlib
		import matplotlib.pyplot as plt
		cmap=matplotlib.cm.get_cmap("Reds")
		plt.imshow(heatmap,cmap="Reds")
		plt.show()
		
	"""
