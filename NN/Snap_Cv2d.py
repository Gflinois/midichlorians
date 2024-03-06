import torch
import os
import pytorch_lightning



class neur_net_struct(pytorch_lightning.LightningModule):
	def __init__(self,Batchsize=1, Cv_Cin=1,Cv_Cout=12,frame_W=200, DROPSIZE=0,Numb_Of_Class=4):
		
		#def parametres
		nce = 23
		CV_Wf=frame_W//5
		N_NEURONE = 8*Cv_Cout
		self.noc = Numb_Of_Class
		
		#création d'un réseau 
		super().__init__()
		self.conv1 = torch.nn.Conv2d(Cv_Cin,Cv_Cout, (nce,CV_Wf)) #Cin,Co,(Hf=nce,Wf)
		self.conv2 = torch.nn.Conv2d(Cv_Cout,2*Cv_Cout, (1,CV_Wf)) #Cin,Co,(Hf=nce,Wf)
		self.conv3 = torch.nn.Conv2d(2*Cv_Cout,4*Cv_Cout, (1,CV_Wf)) #Cin,Co,(Hf=nce,Wf)
		self.conv4 = torch.nn.Conv2d(4*Cv_Cout,6*Cv_Cout, (1,CV_Wf)) #Cin,Co,(Hf=nce,Wf)
		self.conv5 = torch.nn.Conv2d(6*Cv_Cout,8*Cv_Cout, (1,CV_Wf)) #Cin,Co,(Hf=nce,Wf)
		self.drop = torch.nn.Dropout(DROPSIZE)
		self.Big = torch.nn.Linear( N_NEURONE,  N_NEURONE//2)
		self.drop1 = torch.nn.Dropout(DROPSIZE)
		self.Inter = torch.nn.Linear( N_NEURONE//2, N_NEURONE//4)
		self.Fin = torch.nn.Linear(N_NEURONE//4, Numb_Of_Class)
		
		
		
		
		
	def forward(self,data):
		data = data[:,:-4,:] if len(data.size())>=3 else data[:-4,:]
		batchsize = data.size()[0]  if len(data.size())>=3 else 1
		nb_of_time = data.size()[1] if len(data.size())>=3 else data.size()[0]
		nce = data.size()[2]        if len(data.size())>=3 else data.size()[1]
		data = torch.reshape(data,[batchsize,1,nce,nb_of_time])
		c1 = self.conv1(data)
		r1 = torch.nn.functional.relu(c1)
		c2 = self.conv2(r1)
		r2 = torch.nn.functional.relu(c2)
		c3 = self.conv3(r2)
		r3 = torch.nn.functional.relu(c3)
		c4 = self.conv4(r3)
		r4 = torch.nn.functional.relu(c4)
		c5 = self.conv5(r4)
		r5 = torch.nn.functional.relu(c5)
		r5 = torch.reshape(r5,[batchsize,r5.size()[1]])
		b = torch.nn.functional.relu(self.drop(self.Big(r5)))
		i = torch.nn.functional.relu(self.drop1(self.Inter(b)))
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
