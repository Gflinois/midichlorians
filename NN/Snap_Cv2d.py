import torch
import os
import pytorch_lightning



class neur_net_struct(pytorch_lightning.LightningModule):
	def __init__(self, N_NEURONE, LSTM_LAYER, DROPSIZE,V):
		#création d'un réseau 
		super().__init__()
		self.conv = torch.nn.Conv2d(1, 1, (6,1))
		self.lstm = torch.nn.LSTM(14*(30-6+1), N_NEURONE, LSTM_LAYER, batch_first=True)
		self.drop = torch.nn.Dropout(DROPSIZE)
		self.Big = torch.nn.Linear( N_NEURONE,  round(N_NEURONE*(3/10)))
		self.drop1 = torch.nn.Dropout(DROPSIZE)
		self.Neck = torch.nn.Linear( round(N_NEURONE*(3/10)), N_NEURONE//4+4)
		self.small = torch.nn.Linear(N_NEURONE//4+4, 11)
		
		
		self.LSTM_LAYER=LSTM_LAYER
		self.N_NEURONE=N_NEURONE
		self.reset()
		
		
		
	def forward(self,data):
		#with torch.no_grad() :
			if len(data.size())==2:
				s=(1,)+data.size()
				data=torch.reshape(data,s)
			data=torch.reshape(data,[data.size()[0],1,30,14])
			c=torch.nn.functional.relu(self.conv(data))
			self.conv_layer_output=c
			c=torch.reshape(c,[c.size()[0],1,c.size()[-2]*c.size()[-1]])
			z,a = self.lstm(c)
			t = torch.nn.functional.relu(z[:,-1,:])
			t = torch.nn.functional.relu(self.drop(self.Big(t)))
			t = torch.nn.functional.relu(self.drop1(self.Neck(t)))
			t = self.small(t)
			t = 2*torch.sigmoid(t)-1
			#self.pred=t
			return t
		
		


	def reset(self):
		selfmemory = torch.zeros([self.LSTM_LAYER,14,self.N_NEURONE])


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
		x=torch.reshape(x,[x.size()[0],1,30,14])
		c=torch.nn.functional.relu(self.conv(x))
		c=torch.reshape(c,[c.size()[0],1,c.size()[-2]*c.size()[-1]])
		z,a = self.lstm(c)
		t = torch.nn.functional.relu(z[:,-1,:])
		t = torch.nn.functional.relu(self.drop(self.Big(t)))
		t = torch.nn.functional.relu(self.drop1(self.Neck(t)))
		t = self.small(t)
		t = 2*torch.sigmoid(t)-1
		loss = torch.nn.functional.mse_loss(t, y)
		self.log('train_loss', loss)
		return loss
		



	def validation_step(self, batch,batch_idx):
		x,y = batch
		y=torch.tensor(y,dtype=torch.float32)
		x=torch.reshape(x,[x.size()[0],1,30,14])
		c=torch.nn.functional.relu(self.conv(x))
		c=torch.reshape(c,[c.size()[0],1,c.size()[-2]*c.size()[-1]])
		z,a = self.lstm(c)
		t = torch.nn.functional.relu(z[:,-1,:])
		t = torch.nn.functional.relu(self.drop(self.Big(t)))
		t = torch.nn.functional.relu(self.drop1(self.Neck(t)))
		t = self.small(t)
		
		t = 2*torch.sigmoid(t)-1
		loss = torch.nn.functional.mse_loss(t, y)
		
		self.log('val_loss', loss)
		self.log('hp_metric', loss)


	def test_loop(self, dataloader):
		size = len(dataloader.dataset)
		num_batches = len(dataloader)
		test_loss, correct = 0, 0
		self.pred_table=torch.zeros(11,11)
		
		with torch.no_grad():
			for X, y in dataloader:
				
				pred = self(X)
				test_loss += torch.nn.functional.mse_loss(pred, y).item()
				correct += int(pred.argmax()== y.argmax())
				self.pred_table[y.argmax()][pred.argmax()]+=1
		for i in range(11):
			ti=sum(self.pred_table[i])
			for j in range(11):
				self.pred_table[i][j]=self.pred_table[i][j].item()*100//ti.item() if ti!=0 else  0 

		test_loss /= size
		print("Test Error: \n Accuracy: "+str(100*correct/size)+f"%, Avg loss: {test_loss:>8f} \n")
		return 100*correct/size
	
	
	
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
		

