import argparse
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from dataloader import load_data, get_data
from utils import params_to_tb, add_noise, test_sun, interpolate


from model.resnet1d import ResidualNetworkD1
from model.attention_network import Attention_network
from model.bresnet1d import BayesianResidualNetworkD1
from model.VAE1d import ConvVAE1D
from model.AFVAE import AFVAE
from model.DAE1d import DAE1d

from tensorboardX import SummaryWriter
import datetime

def loss_func(y, y_hat):
    return F.mse_loss(y, y_hat, reduction='none').view(y.shape[0], -1).sum(1).mean()

def nll_loss(dist, target):
    # we must return a scalar as that what pytorch requires for backpropagation
    #print(dist.log_prob(target).shape)
    return -dist.log_prob(target).view(target.shape[0], -1).sum(1).mean()

def main(args):
    # Setup tensorboard stuff
    writer = SummaryWriter("../tensorboard_data/"+ args.model_type +"-"+ args.denoise+str(args.denoise_latent)+ str(args.noise_level) +str(datetime.datetime.now()))
    params_to_tb(writer, args)


    ## Load Data
    size = 2
    spectrum, y = load_data(args.channelwise)
    if args.benchmark:
        spectrum = spectrum[y[:,0] <= 6000]
        y = y[y[:,0] <= 6000]
        spectrum = spectrum[y[:,0] >= 4000]
        y = y[y[:,0] >= 4000]
        size = 4

    spectrum, y = interpolate(spectrum, y, number_of_inters = size)

    torch.manual_seed(0)
    #spectrum = add_noise(spectrum, args.noise_level)
    print(spectrum.shape)
	
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model_type == 'bayes' or args.model_type == 'bAttnVGG' or args.model_type == 'bAttn1d':
        Bayesian = True
    else:
        Bayesian = False

    X_train, X_test, y_train, y_test = train_test_split(spectrum.data.numpy(), y.data.numpy(), random_state=55, test_size = 0.1)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=55, test_size = 0.1)
    
    X_train= torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_val= torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).float()
    X_test= torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    
    print("Normalizing")
    train_means = torch.mean(y_train, dim = 0)
    train_std = torch.std(y_train, dim = 0)
    y_train = (y_train-train_means)/train_std

    y_val = (y_val-train_means)/train_std
    y_test = (y_test-train_means)/train_std
    print(train_std)
    print(train_means)
        
    print(spectrum.shape)
    print(y.shape)

    if args.model_type == 'conv1d':
        model = conv1D(in_size = spectrum.shape[-1], out_size = 4,
		            input_channels = spectrum.shape[1],
                    convolutions = args.convolutions,
                    kernel_size = args.kernel_size,
                    hiddenlayer= args.hiddenlayer,
                    maxpool = args.maxpool,
                    dropout = args.dropout)
    elif args.model_type == 'resnet':
        print("resnet")
        model = ResidualNetworkD1(in_size = spectrum.shape[-1], out_size = 4,
                    input_channels = spectrum.shape[2],
                    convolutions = args.convolutions,
                    kernel_size = args.kernel_size,
                    hiddenlayer= args.hiddenlayer,
                    maxpool = args.maxpool,
                    dropout = args.dropout)
    elif args.model_type == 'conv2d':
        print("resnet2d")
        model = ResidualNetworkD2(in_size = 8*4096, out_size = 4,
                    convolutions = args.convolutions,
                    kernel_size = args.kernel_size,
                    hiddenlayer= args.hiddenlayer,
                    maxpool = args.maxpool,
                    dropout = args.dropout)
    elif  args.model_type == 'bayes':
        print('Bayesian')
        model = BayesianResidualNetworkD1(in_size = spectrum.shape[-1], out_size = 4,
                    input_channels = spectrum.shape[2],
                    convolutions = args.convolutions,
                    kernel_size = args.kernel_size,
                    hiddenlayer= args.hiddenlayer,
                    maxpool = args.maxpool,
                    dropout = args.dropout)
    elif args.model_type == 'attention':
        print("spatialAttetion")
        model = SpatialAttentionNetwork(4)
    elif args.model_type == 'AttnVGG':
        print("AttnVGG")
        model = AttnVGG_after(im_size=4096, num_classes=4,
            attention=True, normalize_attn=True)
    elif args.model_type == 'bAttnVGG':
        print("bAttnVGG")
        model = bAttnVGG_after(im_size=4096, num_classes=4,
            attention=True, normalize_attn=args.norm_att)
    elif args.model_type == 'bAttn1d':
        print("batt1d")
        model = bAttnVGG_1d(im_size=4096, num_classes=4,
            attention=True, normalize_attn=True)
    else:
        model = conv2D(in_size = 8*4096, out_size = 4,
            convolutions = args.convolutions,
            kernel_size = args.kernel_size,
            hiddenlayer= args.hiddenlayer,
            maxpool = args.maxpool,
            dropout = args.dropout)

    model.to(device)

    if(args.l1):
        criterion = nn.L1Loss()
    else:    
        criterion = nn.MSELoss()


    if(args.SGD):
        optimizer = optim.AdamW(model.parameters(), lr=args.learningrate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learningrate, weight_decay = args.l2) 
   
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[(args.epochs-args.lr_decay_milestones)], gamma=args.lr_decay_factor)
    if (args.model_type == 'attention'):
        lr = 3e-4
        optim.Adam([
                {'params': model.networks.parameters(), 'lr':  lr, 'weight_decay': 10e-5},
                {'params': model.finals.parameters(), 'lr':  lr, 'weight_decay': 10e-5},
                {'params': model.stn.parameters(), 'lr': lr*10e-2, 'weight_decay': 10e-5}
        ])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[325,420], gamma=args.lr_decay_factor)

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataset_val = torch.utils.data.TensorDataset(X_val, y_val)
    dataset_test = torch.utils.data.TensorDataset(X_test, y_test)
    
    BATCH_SIZE = args.batch_size

    trainloader=torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                            pin_memory=True)
    valloader=torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                            pin_memory=True)
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                            pin_memory=True)

    if(args.denoise != " "):
        if(args.denoise == 'VAE1D'):
            denoiser = ConvVAE1D(dataset[0][0].squeeze(0).shape, args.denoise_latent**2)
        elif(args.denoise =='DAE'):
            denoiser = ConvDAE(dataset[0][0].shape,  args.denoise_latent**2)
        elif(args.denoise =='DAE1d'):
            print("DAE1d")
            denoiser = DAE1d(dataset[0][0].squeeze(0).shape,  args.denoise_latent**2)
        elif(args.denoise =='VAE2D'):
            denoiser = ConvVAE(dataset[0][0].shape,  args.denoise_latent**2)
        elif(args.denoise == 'AFVAE'):
            denoiser = AFVAE(dataset[0][0].shape,  args.denoise_latent**2)

        denoiser.load_state_dict(torch.load("../savedmodels/"+args.denoise +  str(args.denoise_latent) +str(args.noise_level)+ ".pth", map_location=torch.device(device)))
        denoiser.to(device)
        denoiser.eval()
        test_spectrum_clean = spectrum[0:15].to(device)
        test_spectrum = spectrum[0:15].to(device)
        denoised, _ = denoiser.reconstruct(test_spectrum.to(device))
        print(f'MSE_recon: {torch.sum((denoised.cpu()-test_spectrum_clean.cpu())**2)}')
        print(f'MSE_noise: {torch.sum((test_spectrum.cpu()-test_spectrum_clean.cpu())**2)}')
        del test_spectrum_clean
        del test_spectrum
        del denoised

    print("setup Complete")
    TB_counter = 0
    epochs = args.epochs
    start_epoch = 0
    if args.restore_checkpoint:
        checkpoint = torch.load("../savedmodels/checkpoint"+ args.model_type +"-"+ args.denoise+str(args.denoise_latent)+ str(args.noise_level))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        scheduler.load_state_dict(checkpoint['scheduler'])
        
    for epoch in range(start_epoch,epochs):
        train_loss = 0
        train_counter = 0
        model.train()
        for i, (mini_batch_x, mini_batch_y) in enumerate(trainloader):
                
            mini_batch_x = add_noise(mini_batch_x, args.noise_level)
            # If denoise run a denoising step
            if(args.denoise != " "):
                    mini_batch_x, _ = denoiser.reconstruct(mini_batch_x.to(device))

            optimizer.zero_grad()
            #### Forward Pass
            y_pred = model(mini_batch_x.to(device))

            #### Compute Loss
            if Bayesian:
                loss = nll_loss(y_pred, mini_batch_y.to(device))
                #print(loss.item())
                #print(y_pred.mean)
                #print(y_pred.stddev)
            else:
                loss = loss_func(y_pred, mini_batch_y.to(device))
            #loss = loss_func(y_pred.squeeze(), mini_batch_y.to(device))
            #### Backward pass
            loss.backward()
            optimizer.step()
            train_loss += loss.cpu().data.numpy()
            train_counter +=1
        
        scheduler.step()
        writer.add_scalar("train_loss", train_loss/train_counter, global_step=TB_counter)
        TB_counter +=1
        if ((epoch) % 10) == 0:
            val_loss = 0
            val_counter = 0
            with torch.set_grad_enabled(False):
                model.eval()
                for i, (val_batch_x, val_batch_y) in enumerate(valloader):
                    val_batch_x = add_noise(val_batch_x, args.noise_level)
                    if(args.denoise != " "):
                        val_batch_x, _ = denoiser.reconstruct(val_batch_x.to(device))
            
                    if Bayesian:
                        # just take the mean of the estimates
                        y_pred_test = model(val_batch_x.to(device)).mean
                    else:
                        y_pred_test = model(val_batch_x.to(device))
                    val_loss += loss_func(y_pred_test.squeeze(), val_batch_y.to(device))
                    val_counter += 1
        
            val_loss = (val_loss).cpu().data.numpy()/val_counter
            writer.add_scalar("validation_loss", val_loss, global_step=TB_counter)

        if ((epoch) % 10) == 0:
            print('Epoch {}: train_loss: {} Val loss: {}'.format(epoch, loss, val_loss))

        if ((epoch % 25)==0 and args.model_type == 'bAttnVGG'):
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'scheduler': scheduler.state_dict()
            }, "../savedmodels/checkpoint"+ args.model_type +"-"+ args.denoise+str(args.denoise_latent)+ str(args.noise_level))
        
    model.eval()
    old_batch = None
    old_label = None
    with torch.set_grad_enabled(False):
        final_val_loss = 0
        for i, (val_batch_x, val_batch_y) in enumerate(testloader):
            val_batch_x = add_noise(val_batch_x, args.noise_level)
			# If denoise run a denoising step
            if(args.denoise != " "):
                with torch.set_grad_enabled(False):
                    val_batch_x, _ = denoiser.reconstruct(val_batch_x.to(device))
            if  Bayesian:
                # just take the mean of the estimates
                y_pred_test = model(val_batch_x.to(device)).mean
                y_pred_test_std = model(val_batch_x.to(device)).stddev
            else:
                y_pred_test = model(val_batch_x.to(device))

            final_val_loss += loss_func(y_pred_test.squeeze(), val_batch_y.to(device)).cpu()
			
          
            y_pred = (y_pred_test.detach().cpu()*train_std)+train_means
            y = (val_batch_y.detach().cpu()*train_std)+train_means
				
            if i==0:
                residuals =  (y_pred-y).cpu().detach()
                if Bayesian:
                    residuals_stds = y_pred_test_std
            else:
                residuals = torch.cat([residuals, (y_pred-y).detach()], dim = 0)
                if Bayesian:
                    residuals_stds = torch.cat([residuals_stds, y_pred_test_std.detach()], dim = 0)

            if i < 3:
                with open('../residuals/data-'+ args.model_type +"-"+ args.denoise+str(args.denoise_latent)+ str(args.noise_level)+'.csv', 'a') as data:
                    np.savetxt(data, val_batch_x.view(val_batch_x.shape[0],-1).cpu().data.numpy(), delimiter=",")
                with open('../residuals/labels-'+ args.model_type +"-"+ args.denoise+str(args.denoise_latent)+ str(args.noise_level)+'.csv', 'a') as data:
                    np.savetxt(data, y.view(y.shape[0],-1).data.numpy(), delimiter=",")
                with open('../residuals/residuals-'+ args.model_type +"-"+ args.denoise+str(args.denoise_latent)+ str(args.noise_level)+'.csv','a') as res:
                    np.savetxt(res, (y_pred-y).detach(), delimiter=",")
                if Bayesian:
                    with open('../residuals/residuals-std-'+ args.model_type +"-"+ args.denoise+str(args.denoise_latent)+ str(args.noise_level)+'.csv','a') as res:
                        np.savetxt(res, (y_pred_test_std.detach().cpu()*train_std), delimiter=",")


    if args.model_type == 'bAttnVGG' or args.model_type == 'AttnVGG' or args.model_type == 'bAttn1d':
        model.visual_att(testloader, device, args)
    final_val_loss = final_val_loss
    final_test_loss = 0
    final_counter = 0
    with torch.set_grad_enabled(False):
        for i, (val_batch_x, val_batch_y) in enumerate(testloader):
            val_batch_x = add_noise(val_batch_x, args.noise_level)
            if(args.denoise != " "):
                val_batch_x, _ = denoiser.reconstruct(val_batch_x.to(device))
            if Bayesian:
                # just take the mean of the estimates
                y_pred_test = model(val_batch_x.to(device)).mean
            else:
                y_pred_test = model(val_batch_x.to(device))

            final_test_loss += loss_func(y_pred_test.squeeze(), val_batch_y.to(device)).cpu().data.numpy()
            final_counter += 1
    final_test_loss = final_test_loss/final_counter

    print("final validation loss: {}".format(final_val_loss))
    print("final std of residuals from validation set: {}".format(torch.std(residuals, dim = 0).cpu().data.numpy()))
    print("final mean squared error: {}".format(torch.mean(residuals**2, dim = 0).cpu().data.numpy()))
    print("final RMSE error: {}".format(torch.sqrt(torch.mean(residuals**2, dim = 0)).cpu().data.numpy()))
    print("final MAE error: {}".format(torch.mean(torch.abs(residuals), dim = 0).cpu().data.numpy()))
    if Bayesian:  
        print("final unnormed mean std from model: {}".format(torch.mean(y_pred_test_std.cpu()*train_std, dim = 0).cpu().data.numpy()))
    
    print("STARNET RMSE ")
    print("[51.2, 0.081, 0.040] ")
    print("STARNET MAE ")
    print("[31.2, 0.053, 0.025] ")

    print("final test loss: {}".format(final_test_loss))
    test_sun(model, train_means, train_std, device)
    print("Saving Residuals")
    if args.savemodel:
        torch.save(model.state_dict(),"../savedmodels/"+args.name)
	
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 1D or 2D
    parser.add_argument('--channelwise', type=eval, default=True)
    parser.add_argument('--savemodel', type=eval, default=False)
    parser.add_argument('--denoise', type=str, default=" ")
    parser.add_argument('--denoise_latent', type=int, default=0)

    parser.add_argument('--model_type', type=str, default=' ')
    parser.add_argument('--restore_checkpoint', type=eval, default=False)

    # attention
    parser.add_argument('--attention', type=eval, default=False)
    parser.add_argument('--norm_att', type=eval, default=True)    

    # Noise_levels
    parser.add_argument('--noise_level', type=float, default= 0)

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learningrate', type=float, default=3e-4)
    
    # Optimization args
    parser.add_argument('--SGD', type = bool, default = False)
    parser.add_argument('--lr_decay_milestones', type=int, default=50)
    parser.add_argument('--lr_decay_factor', type=float, default=0.1)

    # Regularization
    parser.add_argument('--l2', type=float, default=10e-5)
    parser.add_argument('--l1', type=eval, default=False)

    # Name for Tensorboard
    parser.add_argument('--name', type=str, default="")

    # Network params
    parser.add_argument('--convolutions', type=eval, default=[16,16])
    parser.add_argument('--kernel_size', type=eval, default=[7,7])
    parser.add_argument('--hiddenlayer', type=eval, default=[512,256])

    parser.add_argument('--maxpool', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    parser.add_argument('--benchmark', type=bool, default=False)
    args = parser.parse_args()
    main(args)
