from tensorboardX import SummaryWriter
from prettytable import PrettyTable
 
import torch

import h5py

# function to write params to TB
def params_to_tb(writer, args):
    t = PrettyTable(['Argument', 'Value'])
    param_dict = vars(args)
    for key, val in param_dict.items():
        t.add_row([key, val])
    writer.add_text("args", t.get_html_string(), global_step=0)


def add_noise(spectrum, noise_level):
    spectrum = spectrum + torch.normal(mean = torch.zeros_like(spectrum), std = noise_level)
    return spectrum


def my_normalize(x):
    mu = x.mean(dim=(1), keepdim=True)
    sigma = x.std(dim=(1), keepdim=True)
    x_scaled = (x - mu) / (sigma+1e-15)
    return x_scaled
    
    
def interpolate(spectrum, y, number_of_inters = 1, weight = 0.5):
    temp_spec = None
    temp_y = None
    for i in range(0, number_of_inters):
        permutation = torch.randperm(spectrum.size()[0])
        weight = torch.FloatTensor(spectrum.size()[0]).uniform_()*weight
        spectrum_inteporlate = torch.lerp(spectrum, spectrum[permutation],weight.unsqueeze(1).unsqueeze(2).unsqueeze(3))
        y_interpolate = torch.lerp(y, y[permutation],weight.unsqueeze(1))
        if (i == 0):
            temp_spec = spectrum_inteporlate
            temp_y = y_interpolate
        else:
            temp_spec = torch.cat((temp_spec, spectrum_inteporlate))
            temp_y = torch.cat((temp_y, y_interpolate))
    #spectrum = torch.cat((spectrum,spectrum_inteporlate))
    #spectrum = spectrum_inteporlate
    #y = y_interpolate
    
    return temp_spec, temp_y


def test_sun(model, train_means, train_std, device):
    with h5py.File("../data/sun_2020-06-17T13-27-26.219.h5", "r") as hf :
        spectrum = hf["spectrum"][:]

    sun = torch.from_numpy(spectrum).float().to(device)
    print(sun.shape)
    med1 = torch.median(sun[0,0][sun[0,0]!=0])
    med2 = torch.median(sun[0,1])
    med3 = torch.median(sun[0,2])
    med4 = torch.median(sun[0,3])
    med5 = torch.median(sun[0,4])
    med6 = torch.median(sun[0,5])
    med7 = torch.median(sun[0,6])
    med8 = torch.median(sun[0,7][sun[0,7]!=0])
    medians = torch.stack([med1,med2, med3, med4, med5, med6, med7, med8]).unsqueeze(0)
    medians = medians.view(8,1).repeat(1,4096).unsqueeze(0)
    sun = sun/medians
    sun = sun.unsqueeze(0)
    model.eval()
    results = model(sun)
    estimation = (results.mean.detach().cpu()*train_std)+train_means
    unceratainty = results.stddev.detach().cpu()*train_std
    print("PERFORMANCE ON SUN")
    print(estimation)
    print(unceratainty)
    print("---------")
