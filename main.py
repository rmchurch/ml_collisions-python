# -*- coding: utf-8 -*-
"""full_both_sp.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hQ629OzxIuMUyDMCIiqXJirxvdDKdNKc

"""# set vars"""

from __future__ import print_function, division
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import timeit
import h5py
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from torch.nn import functional as F
from torch.autograd import Variable
from radam import RAdam, PlainRAdam, AdamW
from models import Unet,ReSeg,StackedRecurrentHourglass

batch_size = 128
lr = 1e-05
warmup_period = 10
momentum = 0.99
num_epochs = 100
percentage_train = 0.8
percentage_val = 0.1
lr_decay = 0.25
step_size = 15
# loss_weights = [1,1e0,1e21,1e15]
loss_weights = [1,0.05,0.05,0.05,0.05]
#loss_weights = [1,0,0,0,0]
nphi = 8
plot_rate = 500
output_rate = 500
val_rate = 1000
datapath = '/scratch/gpfs/marcoam/ml_collisions/data/xgc1/ti272_JET_heat_load/'
run_num = '00094/'
lim = 150000
use_vth = True #normalize momentum in check_properties to m*n*vth, instead of m*n*upar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""# choose network"""

#net = Unet().to(device)
net = ReSeg().to(device)
#net = StackedRecurrentHourglass().to(device)

#print(sum(p.numel() for p in net.parameters() if p.requires_grad))

"""# load data"""

def load_data_hdf(iphi):
  
  hf_f = h5py.File(datapath+run_num+'hdf_f.h5','r')
  hf_df = h5py.File(datapath+run_num+'hdf_df.h5','r')
  
  e_f = hf_f['e_f'][iphi]  
  i_f = hf_f['i_f'][iphi]
  e_df = hf_df['e_df'][iphi] 
  i_df = hf_df['i_df'][iphi]
 
  hf_f.close()
  hf_df.close()
 
  ind1,ind2,ind3 = i_f.shape
 #change lim back to ind2 if want full set 
  f = np.zeros([lim,2,ind1,ind1])
  df = np.zeros([lim,2,ind1,ind1])

  for n in range(lim):
    f[n,0,:,:-1] = e_f[:,n,:]
    f[n,1,:,:-1] = i_f[:,n,:]
    df[n,0,:,:-1] = e_df[:,n,:]
    df[n,1,:,:-1] = i_df[:,n,:]

    f[n,0,:,-1] = e_f[:,n,-1]
    f[n,1,:,-1] = i_f[:,n,-1]
    df[n,0,:,-1] = e_df[:,n,-1]
    df[n,1,:,-1] = i_df[:,n,-1]
    
  del i_f,e_f,i_df,e_df

  # find where f is negative and replace w/ zero
  neg_f_inds = np.where(f < 0)
  f[neg_f_inds] = 0
  
  # find where df is 0
  zero_df_inds = np.where(np.einsum('ijkl -> i',np.abs(df)) < 1)
  zero_df_inds = list(zero_df_inds[0]) 

  fid = open('bad_inds.txt','w')
  for ind in zero_df_inds:
    fid.write(str(ind)+'\n') 
  fid.close()   

  df+=f

  # instantiate variables for conservation properties and for normalization
  hf_cons = h5py.File(datapath+run_num+'hdf_cons_fullvol.h5','r')
  hf_vol = h5py.File(datapath+run_num+'hdf_vol.h5','r')
  cons = conservation_variables(hf_cons,hf_vol)
 
  hf_cons.close()
  hf_vol.close()
 
  hf_stats = h5py.File(datapath+run_num+'hdf_stats.h5','r')
  zvars = stats_variables(hf_stats)
  
  hf_stats.close()

  for n in range(lim):
    f[n] = (f[n]-zvars.mean_f)/zvars.std_f
#     df[n] = (df[n]-zvars.mean_df)/zvars.std_df
    df[n] = (df[n]-zvars.mean_fdf)/zvars.std_fdf
    
  zvars.mean_f = zvars.mean_f[np.newaxis]
  zvars.mean_df = zvars.mean_df[np.newaxis]
  zvars.mean_fdf = zvars.mean_fdf[np.newaxis]
  zvars.std_f = zvars.std_f[np.newaxis]
  zvars.std_df = zvars.std_df[np.newaxis]
  zvars.std_fdf = zvars.std_fdf[np.newaxis]
  
  for i in range(int(np.ceil(np.log(batch_size)/np.log(2)))):
    zvars.mean_f = np.concatenate((zvars.mean_f,zvars.mean_f),axis=0)
    zvars.mean_df = np.concatenate((zvars.mean_df,zvars.mean_df),axis=0)
    zvars.mean_fdf = np.concatenate((zvars.mean_fdf,zvars.mean_fdf),axis=0)
    zvars.std_f = np.concatenate((zvars.std_f,zvars.std_f),axis=0)
    zvars.std_df = np.concatenate((zvars.std_df,zvars.std_df),axis=0)  
    zvars.std_fdf = np.concatenate((zvars.std_fdf,zvars.std_fdf),axis=0)
  
  zvars.mean_f = torch.from_numpy(zvars.mean_f).to(device).double()
  zvars.mean_df = torch.from_numpy(zvars.mean_df).to(device).double()
  zvars.mean_fdf = torch.from_numpy(zvars.mean_fdf).to(device).double()
  zvars.std_f = torch.from_numpy(zvars.std_f).to(device).double()
  zvars.std_df = torch.from_numpy(zvars.std_df).to(device).double()
  zvars.std_fdf = torch.from_numpy(zvars.std_fdf).to(device).double()
       
  return f,df,lim,zero_df_inds,zvars,cons

class stats_variables():
  
  def __init__(self, hf_stats):
    self.std_f = hf_stats['std_f'][...]
    self.std_df = hf_stats['std_df'][...]
    self.std_fdf = hf_stats['std_fdf'][...]
    self.mean_f = hf_stats['mean_f'][...]
    self.mean_df = hf_stats['mean_df'][...]
    self.mean_fdf = hf_stats['mean_fdf'][...]

class conservation_variables():

  def __init__(self, hf_cons, hf_vol):
    self.f0_dsmu = hf_cons['f0_dsmu'][...]
    self.f0_dvp = hf_cons['f0_dvp'][...]
    self.f0_nvp = hf_cons['f0_nvp'][...]
    self.f0_nmu = hf_cons['f0_nmu'][...]
    self.ptl_mass = hf_cons['ptl_mass'][...]
    self.sml_ev2j = 1.6022e-19
    
    self.temp = hf_cons['f0_T_ev'][...]
    self.vol = np.zeros([self.temp.shape[0],self.f0_nmu+1,self.temp.shape[1]])
    self.vol[0] = hf_vol['vole'][0]
    self.vol[1] = hf_vol['voli'][0]

class DistFuncDataset(Dataset):

    def __init__(self, f_array, df_array, temp_array, vol_array):
        self.data = torch.from_numpy(f_array).double()
        self.target = torch.from_numpy(df_array).double()
        self.temp = torch.from_numpy(temp_array).double()
        self.vol = torch.from_numpy(vol_array).double()
        
    def __len__(self):
        return len(self.data)
      
    def __getitem__(self, index):
        a = self.data[index]
        b = self.target[index]
        b = b.view(-1,32,32)
        c = self.temp[index]
        d = self.vol[index]
            
        return a, b, c, d

"""# split data"""

def split_data(f,df,cons,num_nodes,bad_inds):
    
    inds = list(np.arange(num_nodes))
    for bad_ind in bad_inds:
      inds.remove(bad_ind)

    #np.random.seed(0)
    np.random.shuffle(inds) 
    
    num_train = int(np.floor(percentage_train*num_nodes))
    num_val = int(np.floor(percentage_val*num_nodes))
    
    train_inds = inds[:num_train]
    val_inds = inds[num_train:num_train+num_val]
    test_inds = inds[num_train+num_val:]
 
    f_train = f[train_inds]
    f_val = f[val_inds]
    f_test = f[test_inds]
    

    # write out indices for running validation and tests
    fid_inds = open('inds.txt','w')

    fid_inds.write('train\n')
    for tr in train_inds:
      fid_inds.write(str(tr)+'\n')
    fid_inds.write('val\n')
    for v in val_inds:
      fid_inds.write(str(v)+'\n')
    fid_inds.write('test\n')
    for te in test_inds:
      fid_inds.write(str(te)+'\n')
    fid_inds.close()
    
    del f  
      
    df_train = df[train_inds]
    df_val = df[val_inds]
    df_test = df[test_inds]
    
    del df
        
    # temperature and volume separate arrays here
    temp = np.einsum('ij -> ji', cons.temp)   
    temp_train = temp[train_inds]
    temp_val = temp[val_inds]
    temp_test = temp[test_inds]
    
    del temp
    
    vol = np.einsum('ijk -> kji', cons.vol)
    vol_train = vol[train_inds]
    vol_val = vol[val_inds]
    vol_test = vol[test_inds]
    
    
    trainset = DistFuncDataset(f_train, df_train, temp_train, vol_train)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, 
                             shuffle=True, pin_memory=True, num_workers=4)
    
    del f_train, df_train, temp_train, vol_train
    
    valset = DistFuncDataset(f_val, df_val, temp_val, vol_val)
    
    valloader = DataLoader(valset, batch_size=batch_size, 
                           shuffle=True, pin_memory=True, num_workers=4)
        
    return trainloader, valloader, f_test, df_test, temp_test, vol_test

"""# check props"""

# same procedure as col_f_convergence_eval
def check_properties_each(f_slice, cons, temp, vol, sp):
    
    f_slice = f_slice.double()
       
    if len(f_slice.shape) == 2:
      nperp, npar = f_slice.shape
      nbatch = 1
    elif len(f_slice.shape) == 3:  
      nbatch,nperp,npar = f_slice.shape
      
    vth = torch.sqrt(temp*cons.sml_ev2j/cons.ptl_mass[sp]) 
    
    vpar,vperp,vperp1 = create_vpa_vpe_grid(cons)    
    vpar = torch.tensor(vpar).double().to(device)
    vperp = torch.tensor(vperp).double().to(device)
    vperp1 = torch.tensor(vperp1).double().to(device)
        
    mass = cons.ptl_mass[sp]
    conv_factor_notemp = 1/np.sqrt((2*np.pi*cons.sml_ev2j/mass)**3)
    temp_factor = 1/torch.sqrt(temp)
    
    #smu_n = cons.f0_dsmu/3 # smu_n = f0_dsmu/f0_mu0_factor, f0_mu0_factor = 3
    f_slice_norm = torch.einsum('ijk,i,j -> ijk',f_slice,temp_factor,1./vperp1)*conv_factor_notemp
      
    ones_tensor = torch.ones(nbatch,nperp,npar).double().to(device)
      
    vol_tensor = torch.einsum('ijk,ij -> ijk',ones_tensor,vol)
    vperp_tensor = torch.einsum('ijk,i,j -> ijk',ones_tensor,vth,vperp)
    vpar_tensor = torch.einsum('ijk,i,k -> ijk',ones_tensor,vth,vpar)
    
    mass_tensor = vol_tensor
    mom_tensor = vpar_tensor*cons.ptl_mass[sp]*vol_tensor
    energy_tensor = (vpar_tensor**2 + vperp_tensor**2)*cons.ptl_mass[sp]*vol_tensor
        
    mass_tensor, mom_tensor, energy_tensor = \
    mass_tensor.to(device), mom_tensor.to(device), energy_tensor.to(device)                   
         
    mass = torch.sum(f_slice_norm*mass_tensor, dim = (1,2))
    momentum = torch.sum(f_slice_norm*mom_tensor, dim = (1,2))
    energy = torch.sum(f_slice_norm*energy_tensor, dim = (1,2))
                
    return mass, momentum, energy


def create_vpa_vpe_grid(cons):
    vpar = np.linspace(-cons.f0_nvp,cons.f0_nvp,2*cons.f0_nvp+1)*cons.f0_dvp
    vperp = np.linspace(0,cons.f0_nmu,cons.f0_nmu+1)*cons.f0_dsmu
    vperp1 = vperp.copy()
    vperp1[0] = vperp1[1]/3. #f0_mu0_factor
    return vpar,vperp,vperp1


def make_local_temp(f,cons,temp,sp):
    '''
    Calculates local temperature. Based on f0_moments in mom_module.F90 from XGC-Devel
    '''
    mass = cons.ptl_mass[sp]
    vth = torch.sqrt(temp*cons.sml_ev2j/mass) 

    vpar,vperp,vperp1 = create_vpa_vpe_grid(cons)
    vpar = torch.tensor(vpar).double().to(device)
    vperp = torch.tensor(vperp).double().to(device)
    vperp1 = torch.tensor(vperp1).double().to(device)

    volfac = torch.ones((vperp.size()[0],vpar.size()[0])).double().to(device)
    volfac[0,:] = 0.5 #mu_vol
    volfac[-1,:] = 0.5 #mu_vol


    den = torch.einsum('ijk,jk->i',f,volfac) #not true den w/out f0_grid_vol_vonly. But this normalizes out for Tlocal
    upar = torch.einsum('ijk,k,jk->i',f,vpar,volfac)/den*vth
    Tpar = torch.einsum('ijk,k,jk->i',f,vpar**2,volfac)/den*temp - mass/cons.sml_ev2j*upar**2
    Tperp = torch.einsum('ijk,j,jk->i',f,0.5*vperp**2,volfac)/den*temp
    Tlocal = (Tpar + 2*Tperp)/3
    return Tlocal


# makes calls to individual df/f property calculation
# computes more useful quantities as in col_f_core_m after the calls to col_f_convergence_eval 
def check_properties_main(f,df,temp,vol,cons,use_vth=False):
  
  masse = torch.from_numpy(np.array([cons.ptl_mass[0]])).to(device).double()
  massi = torch.from_numpy(np.array([cons.ptl_mass[1]])).to(device).double()
  
  #print('df')
  dne,dpe,dwe = check_properties_each(df[:,0],cons,temp[:,0],vol[:,:,0],0)
  dni,dpi,dwi = check_properties_each(df[:,1],cons,temp[:,1],vol[:,:,1],1)
  
  #print('f')  
  ne,mome,ene = check_properties_each(f[:,0],cons,temp[:,0],vol[:,:,0],0)
  ni,momi,eni = check_properties_each(f[:,1],cons,temp[:,1],vol[:,:,1],1) 

  dne_n = torch.abs(dne/ne)
  dni_n = torch.abs(dni/ni)
  
  if use_vth:     
    Te = make_local_temp(f[:,0],cons,temp[:,0],0)
    Ti = make_local_temp(f[:,1],cons,temp[:,1],1)
    vthe = torch.sqrt(Te*cons.sml_ev2j/masse) 
    vthi = torch.sqrt(Ti*cons.sml_ev2j/massi) 
    pvth = massi*ni*vthi + masse*ne*vthe
    min_pvth = massi*ni*torch.sqrt(10*cons.sml_ev2j/massi) + masse*ne*torch.sqrt(10*cons.sml_ev2j/masse)
    dp_p = torch.abs(dpi + dpe)/torch.max(pvth,min_pvth)
  else:
    dp_p = torch.abs(dpi + dpe)/torch.max(torch.abs(momi + mome),1e-3*torch.max(massi,masse)*ne)
  
  dw_w = torch.abs((dwi + dwe)/(eni + ene))
  
  return dne_n,dni_n,dp_p,dw_w

# dataiter = iter(trainloader)
# data, targets, temp, vol = dataiter.next()
# data, targets, temp, vol = data.to(device), targets.to(device), temp.to(device), vol.to(device)     

# outputs = net(data)
# outputs = outputs.to(device)

# nbatch = len(data)     

# data_unnorm = data*zvars.std_f[:nbatch] + zvars.mean_f[:nbatch]
# targets_unnorm = targets*zvars.std_fdf[:nbatch] + zvars.mean_fdf[:nbatch]
# outputs_unnorm = outputs[:nbatch,0]*zvars.std_fdf[:nbatch,1] + zvars.mean_fdf[:nbatch,1]

# targets_nof = targets_unnorm - data_unnorm
# outputs_nof = outputs_unnorm[:nbatch] - data_unnorm[:nbatch,1]

# outputs_nof_to_cat = outputs_nof[:nbatch].unsqueeze(1)
# targets_nof_to_cat = targets_nof[:nbatch,0].unsqueeze(1)

# outputs_nof = torch.cat((outputs_nof_to_cat,targets_nof_to_cat),1)
# check_properties_main(data_unnorm[:,:,:,:-1],outputs_nof[:,:,:,:-1],temp,vol,cons)

"""# train"""

def train(trainloader,valloader,sp_flag,epoch,end,zvars,cons):
  
    props_xgc = []
    props_ml = []
    train_loss_vector = []
    l2_loss_vector = []
    cons_loss_vector = []
    val_loss_vector = []
    
    running_loss = 0.0
    running_l2_loss = 0.0
    running_cons_loss = 0.0
    timestart = timeit.default_timer()
    for i, (data, targets, temp, vol) in enumerate(trainloader):
        timeend = timeit.default_timer()
        #print(timeend-timestart)
     
        data, targets, temp, vol = data.to(device), targets.to(device), temp.to(device), vol.to(device)     
      
        if sp_flag == 0:
            optimizer.zero_grad()
        else:
            optimizer_e.zero_grad()

        outputs = net(data.float()).double()
        outputs = outputs.to(device)
        
        nbatch = len(data)     
        
        data_unnorm = data*zvars.std_f[:nbatch] + zvars.mean_f[:nbatch]
        targets_unnorm = targets*zvars.std_fdf[:nbatch] + zvars.mean_fdf[:nbatch]
        outputs_unnorm = outputs[:,0]*zvars.std_fdf[:nbatch,1] + zvars.mean_fdf[:nbatch,1]
        
	# don't think I need some of these nbatch but unsure           
        targets_nof = targets_unnorm - data_unnorm
        outputs_nof = outputs_unnorm[:nbatch] - data_unnorm[:nbatch,1]
        
        outputs_nof_to_cat = outputs_nof[:nbatch].unsqueeze(1)
        targets_nof_to_cat = targets_nof[:nbatch,0].unsqueeze(1)        
        
        # concatenate with actual dfe
        outputs_nof = torch.cat((targets_nof_to_cat,outputs_nof_to_cat),1)

        masse_xgc,massi_xgc,mom_xgc,energy_xgc = check_properties_main(data_unnorm[:,:,:,:-1],\
                                                                       targets_nof[:,:,:,:-1],temp,vol,cons,
                                                                       use_vth=use_vth)
        masse_ml,massi_ml,mom_ml,energy_ml = check_properties_main(data_unnorm[:,:,:,:-1],\
                                                                   outputs_nof[:,:,:,:-1],temp,vol,cons,
                                                                   use_vth=use_vth)
        
	# only use ml properties for loss - keep track of xgc properties for comparison later 
        masse_loss = torch.sum(masse_ml)/nbatch
        massi_loss = torch.sum(massi_ml)/nbatch
        mom_loss = torch.sum(mom_ml)/nbatch
        energy_loss = torch.sum(energy_ml)/nbatch

        l2_loss = criterion(outputs[:,0],targets[:,1])      
        
        if i % 100 == 99:
            print('masse',masse_loss.item(),'massi',massi_loss.item(),'mom',mom_loss.item(),'en',energy_loss.item(),'l2',l2_loss.item())
              
        loss = l2_loss*loss_weights[0]\
             + masse_loss*loss_weights[1]\
             + massi_loss*loss_weights[2]\
             + mom_loss*loss_weights[3]\
             + energy_loss*loss_weights[4]
  
        cons_loss = masse_loss*loss_weights[1]\
                  + massi_loss*loss_weights[2]\
                  + mom_loss*loss_weights[3]\
                  + energy_loss*loss_weights[4]
  
        loss.backward()
        if sp_flag == 0:
            optimizer.step()
        else:
            optimizer_e.step()

        running_loss += loss.item()
        running_l2_loss += l2_loss.item()
        running_cons_loss += cons_loss.item()
       
        if i % output_rate == output_rate-1:
            print('   [%d, %5d] loss: %.6f' %
                  (epoch + 1, end + i + 1, running_loss / output_rate))
            print('      L2 loss: %.6f' % (running_l2_loss / output_rate))
            print('      conservation loss: %.6f' % (running_cons_loss / output_rate))
           
        if i % plot_rate == plot_rate-1:
            train_loss_vector.append(running_loss / output_rate)
            l2_loss_vector.append(running_l2_loss / output_rate)
            cons_loss_vector.append(running_cons_loss / output_rate)
            running_loss = 0.0
            running_l2_loss = 0.0
            running_cons_loss = 0.0
            #plot_df(targets_unnorm[0,0,:,:-1],outputs_unnorm[0,0,:,:-1],epoch)
            
            props_xgc.append([torch.sum((masse_xgc)/nbatch).item(),\
                             torch.sum((massi_xgc)/nbatch).item(),\
                             torch.sum((mom_xgc)/nbatch).item(),\
                             torch.sum((energy_xgc)/nbatch).item()])

            props_ml.append([torch.sum((masse_ml)/nbatch).item(),\
                             torch.sum((massi_ml)/nbatch).item(),\
                             torch.sum((mom_ml)/nbatch).item(),\
                             torch.sum((energy_ml)/nbatch).item()])

        if i % val_rate == val_rate-1:         
          val_loss = validate(valloader,cons,zvars)
          val_loss_vector.append(val_loss)
        
          is_best = False
          if val_loss < np.min(val_loss_vector): ## check this
            is_best = True 

          if i % val_rate == val_rate-1:
            save_checkpoint({
                             'epoch': epoch+1,
                             'state_dict': net.state_dict(),
                             'val_loss': val_loss,
                             'optimizer': optimizer.state_dict(),
                             }, is_best, lr)

        timestart = timeit.default_timer()  
    end += i + 1
    
    cons_array = np.concatenate((np.array(props_xgc),np.array(props_xgc)),axis=1)
 
    return train_loss_vector, l2_loss_vector, cons_loss_vector, val_loss_vector, cons_array, end

"""# validate"""

def validate(valloader,cons,zvars):
  
  print('      Running validation set')
  
  running_loss = 0.0
  
  with torch.no_grad():
    for i, (data, targets, temp, vol) in enumerate(valloader):
      
      data, targets, temp, vol = data.to(device), targets.to(device), temp.to(device), vol.to(device)
      
      outputs = net(data.float()).double()
      outputs = outputs.to(device)
            
      nbatch = len(data)     

      data_unnorm = data*zvars.std_f[:nbatch] + zvars.mean_f[:nbatch]
      targets_unnorm = targets*zvars.std_fdf[:nbatch] + zvars.mean_fdf[:nbatch]
      outputs_unnorm = outputs[:nbatch,0]*zvars.std_fdf[:nbatch,1] + zvars.mean_fdf[:nbatch,1]

      targets_nof = targets_unnorm - data_unnorm
      outputs_nof = outputs_unnorm[:nbatch] - data_unnorm[:nbatch,1]
      
      outputs_nof_to_cat = outputs_nof[:nbatch].unsqueeze(1)
      targets_nof_to_cat = targets_nof[:nbatch,0].unsqueeze(1)        
        
      # concatenate with actual dfe
      outputs_nof = torch.cat((targets_nof_to_cat,outputs_nof_to_cat),1)

      masse_ml,massi_ml,mom_ml,energy_ml = check_properties_main(data_unnorm[:,:,:,:-1],\
                                                                 outputs_nof[:,:,:,:-1],temp,vol,cons,
                                                                 use_vth=use_vth)  

      masse_loss = torch.sum(masse_ml)/nbatch
      massi_loss = torch.sum(massi_ml)/nbatch
      mom_loss = torch.sum(mom_ml)/nbatch
      energy_loss = torch.sum(energy_ml)/nbatch
                
      l2_loss = criterion(outputs[:,0],targets[:,1])  
                  
      loss = l2_loss*loss_weights[0]\
            + masse_loss*loss_weights[1]\
            + massi_loss*loss_weights[2]\
            + mom_loss*loss_weights[3]\
    	    + energy_loss*loss_weights[4]    
      
      running_loss += loss.item()
  
  #print(i+nbatch/batch_size)
  avg_loss = running_loss/(i+1)
  
  print('         Validation loss: %.3f' % (avg_loss))

  return avg_loss

"""# test"""

def test(f_test,df_test,temp_test,vol_test):
 
    testset = DistFuncDataset(f_test, df_test, temp_test, vol_test)
    
    testloader = DataLoader(testset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)
      
    props_test_xgc = []
    props_test_ml = []
    
    l2_error = []
    lt1 = 0
    gt1 = 0
    with torch.no_grad():
        for (data, targets, temp, vol) in testloader:

            data, targets, temp, vol = data.to(device), targets.to(device), temp.to(device), vol.to(device)
          
            outputs = net(data.float()).double()
            outputs = outputs.to(device)   
            
            nbatch = len(data)     

            data_unnorm = data*zvars.std_f[:nbatch] + zvars.mean_f[:nbatch]
            targets_unnorm = targets*zvars.std_fdf[:nbatch] + zvars.mean_fdf[:nbatch]
            outputs_unnorm = outputs[:nbatch,0]*zvars.std_fdf[:nbatch,1] + zvars.mean_fdf[:nbatch,1]

            targets_nof = targets_unnorm - data_unnorm
            outputs_nof = outputs_unnorm[:nbatch] - data_unnorm[:nbatch,1]

            outputs_nof_to_cat = outputs_nof[:nbatch].unsqueeze(1)
            targets_nof_to_cat = targets_nof[:nbatch,0].unsqueeze(1)        

            # concatenate with actual dfe
            outputs_nof = torch.cat((targets_nof_to_cat,outputs_nof_to_cat),1)
  
            props_test_xgc.append([torch.sum(each_prop).item()\
                                 for each_prop in check_properties_main(data_unnorm[:,:,:,:-1],\
                                                                   targets_nof[:,:,:,:-1],temp,vol,cons)],\
                                                                    use_vth=use_vth)         
            props_test_ml.append([torch.sum(each_prop).item()\
                                 for each_prop in check_properties_main(data_unnorm[:,:,:,:-1],\
                                                                   outputs_nof[:,:,:,:-1],temp,vol,cons)],\
                                                                   use_vth=use_vth)
                                
            l2_loss = criterion(outputs[:,0],targets[:,1])  
            l2_error.append(l2_loss.item()*100)
    
    cons_test_array = np.concatenate((np.array(props_test_xgc),np.array(props_test_ml)),axis=1)

    print('\nHighest L2: %.6f' % (max(l2_error)))
    print('Lowest L2: %.6f' % (min(l2_error)))

    print('\nConservation properties: \
              \nXGC:\n   mass_e: %.6f \n   mass_i: %.6f \n   momentum: %.6f \n   energy: %.6f  \
              \nML:\n   mass_e: %.6f \n   mass_i: %.6f \n   momentum: %.6f \n   energy: %.6f ' % ( \
              max(cons_test_array[:,0]),max(cons_test_array[:,1]),max(cons_test_array[:,2]),max(cons_test_array[:,3]), \
              max(cons_test_array[:,4]),max(cons_test_array[:,5]),max(cons_test_array[:,6]),max(cons_test_array[:,7])))
    
    return None

def save_checkpoint(state, is_best, lr, filename='checkpoint.pth.tar'): 
#   torch.save(state,'/content/checkpoints/'+str(lr)+'/'+filename)
  torch.save(state, filename)
  if is_best:
    shutil.copy(filename, 'model_best.pth.tar')

"""# plot"""

def plot_df(df_xgc,df_ml,epoch):
  
  df_xgc = df_xgc.cpu().detach().numpy()
  df_ml = df_ml.cpu().detach().numpy()
  
  df_min = df_xgc.min().item()
  df_max = df_xgc.max().item()
  
  cbarticks = np.linspace(df_min,df_max,10)
  
  fig = plt.figure()
  fig.set_figheight(10)
  fig.set_figwidth(10)

  v_aspect=32/31
  ax1 = fig.add_subplot(1,2,1,aspect=v_aspect)
  ax2 = fig.add_subplot(1,2,2,aspect=v_aspect)
  ctr = ax1.contourf(df_xgc, vmin=df_min, vmax=df_max)
  ax2.contourf(df_ml)
  
  ax1.set_title('Actual df')
  ax2.set_title('Predicted df')
  
  fig.subplots_adjust(right=0.8)
  cbar_ax = fig.add_axes([0.85, 0.3, 0.05, 0.4])
  fig.colorbar(ctr, cax=cbar_ax, ticks=cbarticks)
  
#   fig.savefig('figs/dfs_{}'.format(epoch+1))
  
  plt.show()

"""# main"""

start = timeit.default_timer()
criterion = nn.MSELoss()

optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

#optimizer = RAdam(net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=lr_decay)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

for epoch in range(num_epochs):

  if epoch < warmup_period:
    for group in optimizer.param_groups:
      group['lr'] = (epoch+1)*lr/warmup_period

  lr_epoch = [group['lr'] for group in optimizer.param_groups][0]

  print('Epoch: {} (lr = {})'.format(epoch+1,lr_epoch)) 
  
  epoch1 = timeit.default_timer() 
  end = 0
  for iphi in range(nphi):

    print('Beginning training iphi = {}'.format(iphi))
    print('   Loading data')
    load1 = timeit.default_timer()
    f,df,num_nodes,bad_inds,zvars,cons = load_data_hdf(iphi)
   
    load2 = timeit.default_timer()
    print('      Loading time: %.3fs' % (load2-load1))
    
    print('   Creating training set')
    trainloader,valloader,f_test,df_test,temp_test,vol_test = split_data(f,df,cons,num_nodes,bad_inds)
    del f,df
    
    train1 = timeit.default_timer()
    ### gather testing data
    if epoch == 0:
      if iphi == 0:
        f_all_test,df_all_test,temp_all_test,vol_all_test = f_test,df_test,temp_test,vol_test
        del f_test,df_test,temp_test,vol_test

        print('   Starting training')
        train_loss, l2_loss, cons_loss, val_loss, cons_array, end = \
                                   train(trainloader,valloader,0,epoch,end,zvars,cons)
      
      else:
        f_all_test = np.vstack((f_all_test,f_test))
        df_all_test = np.vstack((df_all_test,df_test))
        temp_all_test = np.vstack((temp_all_test,temp_test))
        vol_all_test = np.vstack((vol_all_test,vol_test))
        del f_test,df_test,temp_test,vol_test

        print('   Starting training')
        train_loss_to_app, l2_loss_to_app, cons_loss_to_app, val_loss_to_app, cons_to_cat, end = \
                                   train(trainloader,valloader,0,epoch,end,zvars,cons)

        for loss1 in train_loss_to_app:
          train_loss.append(loss1)
        for loss2 in l2_loss_to_app:
          l2_loss.append(loss2)
        for loss3 in cons_loss_to_app:
          cons_loss.append(loss3)          
        for loss4 in val_loss_to_app:
          val_loss.append(loss4)      
        cons_array = np.concatenate((cons_array, cons_to_cat), axis=0)
    
    else:
      del f_test,df_test,temp_test,vol_test
      print('   Starting training')
      train_loss_to_app, l2_loss_to_app, cons_loss_to_app, val_loss_to_app, cons_to_cat, end = \
                                 train(trainloader,valloader,0,epoch,end,zvars,cons)

      for loss1 in train_loss_to_app:
          train_loss.append(loss1)
      for loss2 in l2_loss_to_app:
          l2_loss.append(loss2)         
      for loss3 in cons_loss_to_app:
          cons_loss.append(loss3)          
      for loss4 in val_loss_to_app:
          val_loss.append(loss4)      
      cons_array = np.concatenate((cons_array, cons_to_cat), axis=0)
         
    train2 = timeit.default_timer()
    print('Finished tranining iphi = {}'.format(iphi))
    print('   Training time for iphi = %d: %.3fs' % (iphi,train2-train1))
  
  #train_iterations = np.linspace(1,len(train_loss),len(train_loss))
  #val_iterations = np.linspace(2,len(train_loss),len(val_loss))

  fid_loss1 = open('train_tmp.txt','w')
  fid_loss2 = open('val_tmp.txt','w')
  fid_loss3 = open('l2_tmp.txt','w')
  fid_loss4 = open('cons_tmp.txt','w')
  lr_command = 'w' if epoch == 0 else 'a'
  fid_lr = open('lr.txt',lr_command) 

  curr_iter = len(train_loss) 
  curr_val_iter = len(val_loss)

  for i in range(curr_iter):
    fid_loss1.write(str(train_loss[i])+'\n')
    fid_loss3.write(str(l2_loss[i])+'\n')
    fid_loss4.write(str(cons_loss[i])+' '+str(cons_array[i,4])+' '+str(cons_array[i,5])+' '+str(cons_array[i,6])+'\n')
  for j in range(curr_val_iter): 
    fid_loss2.write(str(val_loss[j])+'\n')
  fid_lr.write(str(lr_epoch)+'\n')
  
  fid_loss1.close()
  fid_loss2.close()
  fid_loss3.close()
  fid_loss4.close()
  fid_lr.close()
 
  #plt.plot(train_iterations,train_loss,'-o',color='blue')
  #plt.plot(val_iterations,val_loss,'-o',color='orange')
  #plt.plot(train_iterations,l2_loss,'-o',color='red')
  #plt.plot(train_iterations,cons_loss,'-o',color='green')
  #plt.legend(['total','validation','l2','cons'])
  #plt.yscale('log')
  #plt.show()
  
  epoch2 = timeit.default_timer()
  if epoch >= warmup_period:
    scheduler.step()
  #scheduler.step(val_loss[-1])
  print('Epoch time: {}s\n'.format(epoch2-epoch1))

print('Starting testing')
test(f_all_test,df_all_test,temp_all_test,vol_all_test)
print('Finished testing')

stop = timeit.default_timer()
print('Runtime: %.3fmins' % ((stop-start)/60))

## used to see differences between voli/e and f0_grid_vol 

# fid1 = h5py.File('/content/hdf5_data/hdf_vol.h5','r')
# fid2 = h5py.File('/content/hdf5_data/hdf_cons_fullvol.h5','r')

# vole1 = fid1['vole'][0]
# voli1 = fid1['vole'][0]

# f0_grid_vol = fid2['f0_grid_vol'][...]
# vole2 = f0_grid_vol[0]
# voli2 = f0_grid_vol[1]
