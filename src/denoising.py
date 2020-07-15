import numpy as np
import argparse
import h5py

import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data.dataset import random_split

import matplotlib.patches as patches
import math

from utils import params_to_tb, add_noise, interpolate
from tensorboardX import SummaryWriter
import datetime

## Import model
from model.DAE import ConvDAE
from model.DAE1d import DAE1d
from model.modelutils import BetaWarmup
from model.VAE import ConvVAE
from model.VAE1d import ConvVAE1D
from model.AFVAE import AFVAE

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def evaluate_latent(net, spectrum, y, device, batch_size, latent_dim, trainloader):
    #X_train, X_test, y_train, y_test = train_test_split(spectrum.data.numpy(), y, random_state=12, test_size = 0.1)
    #dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    #dataset_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

    #BATCH_SIZE = args.batch_size

    #trainloader=torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    #                                        pin_memory=True)
    #test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
     #                                       pin_memory=True)
    net.eval()                                            
    x, y1 = next(iter(trainloader))
    x1, z1 = net.reconstruct(add_noise(x, noise_level=args.noise_level).to(device))
    x, y2 = next(iter(trainloader))
    x2, z2= net.reconstruct(add_noise(x, noise_level=args.noise_level).to(device))
    x, y3 = next(iter(trainloader))
    x3, z3= net.reconstruct(add_noise(x, noise_level=args.noise_level).to(device))
    x, y4 = next(iter(trainloader))
    x4, z4= net.reconstruct(add_noise(x, noise_level=args.noise_level).to(device))
    x, y5 = next(iter(trainloader))
    x5, z5= net.reconstruct(add_noise(x, noise_level=args.noise_level).to(device))
    x, y6 = next(iter(trainloader))
    x6, z6= net.reconstruct(add_noise(x, noise_level=args.noise_level).to(device))
    x, y7 = next(iter(trainloader))
    x7, z7= net.reconstruct(add_noise(x, noise_level=args.noise_level).to(device))
    x, y8 = next(iter(trainloader))
    x8, z8= net.reconstruct(add_noise(x, noise_level=args.noise_level).to(device))
    x, y9 = next(iter(trainloader))
    x9, z9= net.reconstruct(add_noise(x, noise_level=args.noise_level).to(device))
    x, y10 = next(iter(trainloader))
    x10, z10= net.reconstruct(add_noise(x, noise_level=args.noise_level).to(device))
    x, y11 = next(iter(trainloader))
    x11, z11= net.reconstruct(add_noise(x, noise_level=args.noise_level).to(device))
    x, y12 = next(iter(trainloader))
    x12, z12= net.reconstruct(add_noise(x, noise_level=args.noise_level).to(device))
    x, y13 = next(iter(trainloader))
    x13, z13= net.reconstruct(add_noise(x, noise_level=args.noise_level).to(device))
    #x, y14 = next(iter(trainloader))
    #x14, z14= net.reconstruct(x.to(device)).detach()
    #x, y15 = next(iter(trainloader))
    #x15, z15,= net.reconstruct(x.to(device))
    #x, y16 = next(iter(trainloader))
    #x16, z16,= net.reconstruct(x.to(device))
    #x, y17 = next(iter(trainloader))
    #x17, z17,= net.reconstruct(x.to(device))
    #x, y18 = next(iter(trainloader))
    #x18, z18,= net.reconstruct(x.to(device))
    #x, y19 = next(iter(trainloader))
    #x19, z19,= net.reconstruct(x.to(device))
    #x, y20 = next(iter(trainloader))
    #x20, z20,= net.reconstruct(x.to(device))
    #x, y21 = next(iter(trainloader))
    #x21, z21,= net.reconstruct(x.to(device))
    #x, y22 = next(iter(trainloader))
    #x22, z22,= net.reconstruct(x.to(device))
    #x, y23 = next(iter(trainloader))
    #x23, z23,= net.reconstruct(x.to(device))
    #x, y24 = next(iter(trainloader))
    #x24, z24,= net.reconstruct(x.to(device))
    
    
    
    z = torch.cat([z1,z2,z3,z4,z5,z6,z7,z8,
               z9,z10,z11,z12,z13
               #,z14
               #,z15,z16,z17,z18,z19,z20,z21,z22,z23,z24
               ]).squeeze(1).cpu()
    labels = torch.cat([y1,y2,y3,y4,y5,y6,y7,y8,
               y9,y10,y11,y12,y13
               #,y14
               #,y15,y16,y17,y18,y19,y20,y21,y22,y23,y24
               ]).cpu()
	
    plt.rcParams['image.cmap']='coolwarm'
    plt.rcParams.update({'font.size': 14})
    data_subset = z.detach()
    x = StandardScaler().fit_transform(data_subset.numpy())
    pca = PCA(n_components=latent_dim**2)
    principalComponents = pca.fit_transform(x)
    plt.rcParams['figure.figsize'] = [25, 5]
    plt.subplot(1, 4, 1)
    plt.imshow(z1[0].detach().cpu().view((args.latentdim,args.latentdim)))
    plt.subplot(1, 4, 2)
    plt.imshow(z1[1].detach().cpu().view((args.latentdim,args.latentdim)))
    plt.subplot(1, 4, 3)
    plt.imshow(z1[2].detach().cpu().view((args.latentdim,args.latentdim)))
    plt.subplot(1, 4, 4)
    plt.imshow(z1[3].detach().cpu().view((args.latentdim,args.latentdim)))
    plt.savefig("Latentspace"+str(args.latentdim)+".pdf")
    plt.close()
    
    plt.subplot(1, 4, 1)
    plt.scatter(principalComponents[:,0],principalComponents[:,1], c = labels[:,0])
    plt.colorbar()
    plt.xlabel("Pca 1")
    plt.ylabel("Pca 2")
    plt.title("Teff") 
    plt.subplot(1, 4, 2)
    plt.scatter(principalComponents[:,0],principalComponents[:,1], c = labels[:,1])
    plt.colorbar()
    plt.xlabel("Pca 1")
    plt.ylabel("Pca 2")
    plt.title("Log(g)") 
    plt.subplot(1, 4, 3)
    plt.scatter(principalComponents[:,0],principalComponents[:,1], c = labels[:,2])
    plt.colorbar()
    plt.xlabel("Pca 1")
    plt.ylabel("Pca 2")
    plt.title("z") 
    plt.subplot(1, 4, 4)
    plt.scatter(principalComponents[:,0],principalComponents[:,1], c = labels[:,3])
    plt.colorbar()
    plt.xlabel("Pca 1")
    plt.ylabel("Pca 2")
    plt.title("Vsini") 
    plt.savefig("PCA"+str(args.latentdim)+".png")
    plt.close()
    
    tsne = TSNE()
    tsne_results = tsne.fit_transform(principalComponents)
    plt.subplot(1, 4, 1)
    plt.scatter(tsne_results[:,0],tsne_results[:,1], c = labels[:,0])
    plt.colorbar()
    plt.xlabel(" ")
    plt.ylabel(" ")
    plt.title("Teff") 
    plt.subplot(1, 4, 2)
    plt.scatter(tsne_results[:,0],tsne_results[:,1], c = labels[:,1])
    plt.colorbar()
    plt.xlabel(" ")
    plt.ylabel(" ")
    plt.title("Log(g)") 
    plt.subplot(1, 4, 3)
    plt.scatter(tsne_results[:,0],tsne_results[:,1], c = labels[:,2])
    plt.colorbar()
    plt.xlabel(" ")
    plt.ylabel(" ")
    plt.title("Z") 
    plt.subplot(1, 4, 4)
    plt.scatter(tsne_results[:,0],tsne_results[:,1], c = labels[:,3])
    plt.colorbar()
    plt.xlabel(" ")
    plt.ylabel(" ")
    plt.title("Vsini") 
    plt.savefig("TSNE"+str(args.latentdim)+".png")
    plt.close()


def evaluate_img(net, dl_test, losses, schedule, epoch, device, writer,args, vizualize = 0):
      net.eval()
      loss, nll, kl = losses
      #with torch.set_grad_enabled(False):
      #  data_batch = next(iter(dl_test))[0]
      #  noisy_data_batch = data_batch + torch.normal(mean = torch.zeros_like(data_batch), std = 0.05)
      #  recon, recon_z, _ = net(noisy_data_batch.to(device))
      #  recon = recon.detach().cpu()
      #  recon_z = recon_z.detach().cpu()

      batch_elbo, batch_nll, batch_kl = [], [], []
      with torch.set_grad_enabled(False):
        for i, (mini_batch_x, mini_batch_y) in enumerate(dl_test):
          noisy_signal = add_noise(mini_batch_x, noise_level=args.noise_level)
          outputs = net(noisy_signal.to(device))
          telbo, tnll, tkl = net.elbo(mini_batch_x.to(device), outputs, schedule[epoch])
          batch_elbo.append(telbo.item())
          batch_nll.append(tnll.item())
          batch_kl.append(tkl.item())
  
      print(f'Test: ELBO: {np.mean(batch_elbo)}, KL: {np.mean(batch_kl)}, MSE: {np.mean(batch_nll)}')
      writer.add_scalar("test_elbo", np.mean(batch_elbo), global_step=epoch)
      writer.add_scalar("test_nll", np.mean(batch_nll), global_step=epoch)
      writer.add_scalar("test_kl", np.mean(batch_kl), global_step=epoch)

      betas = [schedule[i] for i in range(len(loss))]
      if(vizualize):
            with torch.set_grad_enabled(False):
              for i, (mini_batch_x, mini_batch_y) in enumerate(dl_test):
                noisy_data_batch = add_noise(mini_batch_x, noise_level=args.noise_level)
                recon, recon_z = net.reconstruct(noisy_data_batch.to(device))
                recon_z = recon_z.view((-1,args.latentdim,args.latentdim))
            plt.figure(figsize=(16, 8))
            print(f'MSE_recon: {torch.sum((recon.cpu()-mini_batch_x.cpu())**2)}')
            print(f'MSE_noise: {torch.sum((noisy_data_batch.cpu()-mini_batch_x.cpu())**2)}')
            plt.subplot(2, 5, 1)
            plt.plot(range(len(nll)), (nll))
            plt.title('MSE')

            plt.subplot(2, 5, 2)
            plt.plot(range(len(kl)), kl)
            plt.title('KL-divergence')

            plt.subplot(2, 5, 3)
            plt.plot(range(len(loss)), loss)
            plt.title('ELBO')

            plt.subplot(2, 5, 4)
            plt.plot(range(len(betas)), betas)
            plt.title('Weight on KL-term (beta)')

            plt.subplot(2, 5, 5)
            plt.title('Original samples')
            plt.imshow(mini_batch_x[0,0,:].cpu().numpy(),aspect='auto', interpolation='none')

            plt.subplot(2, 5, 6)
            plt.title('Reconstruction')
            plt.imshow(recon[0,0,:].cpu().numpy(), aspect='auto', interpolation='none')

            plt.subplot(2, 5, 7)
            plt.title('Latent space')
            plt.imshow(recon_z[0].cpu().numpy(), aspect='auto', interpolation='none')
            plt.subplot(2, 5, 8)
            plt.title("Signal with noise")
            plt.plot(np.arange(0,8*4096), (mini_batch_x[0].cpu()-recon[0,0].cpu()).view(-1,8*4096).squeeze(0).numpy())

            plt.subplot(2, 5, 9)
            plt.title("Signal with noise")
            plt.plot(np.arange(0,8*4096), (mini_batch_x[0].cpu()-noisy_data_batch[0].cpu()).view(-1,8*4096).squeeze(0).numpy())

            plt.subplot(2, 5, 10)
            bins = np.linspace(-0.3, 0.3, 250)
            plt.hist((mini_batch_x[0].cpu()-recon[0,0].cpu()).view(-1,8*4096).squeeze(0).numpy(), bins, alpha=0.5, label='recon')
            plt.hist((mini_batch_x[0].cpu()-noisy_data_batch[0].cpu()).view(-1,8*4096).squeeze(0).numpy(), bins, alpha=0.5, label='noise')
            plt.legend(loc='upper right')
            plt.savefig(args.model_type + str(args.latentdim) + "results.pdf")


def main(args):
    writer = SummaryWriter("../results_DVAE/" + args.model_type + str(args.latentdim)+ "-" +str(datetime.datetime.now()))
    params_to_tb(writer, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    print("Load Data")
    with h5py.File("../data/noisefree_modelgrid.h5", "r") as hf :
        spectrum = hf["spectrum"][:]
        teff = hf["TEFF"][:]
        logg = hf["LOGG"][:]
        Z = hf["Z"][:]
        vsini = hf["VSINI"][:]
    y = np.vstack([teff, logg, Z, vsini])
    y = np.transpose(y)
    y = torch.from_numpy(y).float()
    print("Data Loaded")
    spectrum = torch.from_numpy(spectrum).float().unsqueeze(1)
    
    spectrum, y = interpolate(spectrum,y, number_of_inters = 2)
    
    X_train, X_test, y_train, y_test = train_test_split(spectrum.data.numpy(), y.data.numpy(), random_state=12, test_size = 0.1)
	
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    dataset_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

    BATCH_SIZE = args.batch_size

    train_loader=torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                            pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                            pin_memory=True)

    print("Pytorch DataLoader ready")


    print('Initializing network...')
    if args.model_type =='VAE1D':
      print("conv1d")
      net = ConvVAE1D(dataset[0][0].squeeze(0).shape, args.latentdim**2)
    elif args.model_type =='AFVAE':
      print("AFVAE")
      net = AFVAE(dataset[0][0].squeeze(0).shape, args.latentdim**2)
    elif args.model_type =='DAE':
      print("DAE")
      net = ConvDAE(dataset[0][0].shape, args.latentdim**2)
    elif args.model_type =='DAE1d':
      print("DAE1d")
      net = DAE1d(dataset[0][0].squeeze(0).shape, args.latentdim**2)
    else:
      net = ConvVAE(dataset[0][0].shape, args.latentdim**2)
    net.to(device)

    print('Moving network to GPU...')

    print('Running a forward pass...')
    with torch.no_grad():
      outputs = net(next(iter(train_loader))[0].to(device))
    print('Done!')

    dimensions = spectrum.shape[-1]
    #optimizer = optim.Adam(net.parameters(), lr = args.learningrate, weight_decay=10e-5)
    optimizer = optim.AdamW(net.parameters(), lr = args.learningrate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.num_epochs-10], gamma=0.1)
    schedule = BetaWarmup(args.num_epochs)
    train_elbo, valid_elbo = [], []
    train_nll, valid_nll = [], []
    train_kl, valid_kl = [], []

    old_batch = None
    old_label = None
    for epoch in range(args.num_epochs):

      net.train()
      batch_elbo, batch_nll, batch_kl = [], [], []

      for i, (mini_batch_x, mini_batch_y) in enumerate(train_loader):
        #noisy_signal = add_noise(mini_batch_x, noise_level=0.05)
        outputs = net(add_noise(mini_batch_x, noise_level=args.noise_level).to(device))
        elbo, nll, kl = net.elbo(mini_batch_x.to(device), outputs, schedule[epoch])

        optimizer.zero_grad()
        elbo.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
        
        batch_elbo.append(elbo.item())
        batch_nll.append(nll.item())
        batch_kl.append(kl.item())

      scheduler.step()
      train_elbo.append(np.mean(batch_elbo))
      train_nll.append(np.mean(batch_nll))
      train_kl.append(np.mean(batch_kl))
      
      writer.add_scalar("train_elbo", train_elbo[-1], global_step=epoch)
      writer.add_scalar("train_nll", train_nll[-1], global_step=epoch)
      writer.add_scalar("train_kl", train_kl[-1], global_step=epoch)
      

      print(f'Epoch {epoch+1}/{args.num_epochs}, ELBO: {train_elbo[-1]}, KL: {train_kl[-1]}, NLL: {train_nll[-1]}')

      if epoch % 5 == 0:
        evaluate_img(net, test_loader, (np.array(train_elbo), np.array(train_nll), np.array(train_kl)), schedule,epoch, device, writer, args)
    # final performance on test_set
    torch.save(net.state_dict(), "../savedmodels/"+ args.model_type + str(args.latentdim)+str(args.noise_level)+'.pth')
    
    evaluate_img(net, test_loader, (np.array(train_elbo), np.array(train_nll), np.array(train_kl)), schedule,epoch, device, writer, args, vizualize=1)
    plt.close()
    evaluate_latent(net, spectrum, y, device, BATCH_SIZE, args.latentdim, test_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()    

    parser.add_argument('--name', type=str, default="") # Name for Tensorboard
    
    parser.add_argument('--latentdim', type=int, default=50) # Name for Tensorboard

    parser.add_argument('--model_type', type=str, default="VAE2D")

    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--learningrate', type=float, default=3e-4)
    parser.add_argument('--grad_clip', type=float, default=1)
    
    parser.add_argument('--num_epochs', type=int, default=200)
    
    parser.add_argument('--noise_level', type=float, default=0.05)

    args = parser.parse_args()
    main(args)


# TODO: 
# update Pytorch -> nye baselines
