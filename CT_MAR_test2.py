
import argparse
import torch
import numpy as np
# import h5py
import os
import torch.optim as optim
import torchvision
from backbones.ncsnpp_generator_adagn import NCSNpp  # PROBLEM
from CT_MAR_dataset import CustomDataset
from torch.utils.data import DataLoader, random_split

import torch.nn.functional as F
import torchvision.transforms as transforms
# import backbones.generator_resnet 

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import json

import os
import torch
import sys
import platform
import gecatsim as xc
from gecatsim.pyfiles.CommonTools import *
# from gecatsim.reconstruction.pyfiles import recon

import argparse
import glob

import re

from PIL import Image 

def psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    psnr_value = compare_psnr(img1, img2, data_range=img2.max() - img2.min())
    return psnr_value

def ssim(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    ssim_value = compare_ssim(img1, img2, data_range=img2.max() - img2.min())
    return ssim_value
        
#%% Diffusion coefficients 
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small)  + eps_small
    return t.to(device)

def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3
   
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    
    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas

#%% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        
        _, _, self.betas = get_sigma_schedule(args, device=device)
        
        #we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
                                        )               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        
def sample_posterior(coefficients, x_0,x_t, t):
    
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped
  
    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        
        noise = torch.randn_like(x_t)
        
        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
            
    sample_x_pos = p_sample(x_0, x_t, t)
    
    return sample_x_pos

def sample_from_model(coefficients, generator, n_time, x_init, T, opt):
    x = x_init[:,[0],:] # noise
    source = x_init[:,[1],:]
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)#.to(x.device)
            
            x_0 = generator(torch.cat((x,source),axis=1), t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0[:,[0],:], x, t)
            x = x_new.detach()
        
    return x

def load_checkpoint(checkpoint_dir, netG, name_of_network, epoch,device = 'cuda:0'):
    checkpoint_file = checkpoint_dir.format(name_of_network, epoch)  

    checkpoint = torch.load(checkpoint_file, map_location=device)
    ckpt = checkpoint
    
    #print(name_of_network, ':', ckpt.keys())
    
    for key in list(ckpt.keys()):
         ckpt[key[7:]] = ckpt.pop(key)
    #print(name_of_network, ':', ckpt.keys())
    netG.load_state_dict(ckpt)
    netG.eval()
    print('get checkpoint!')

def AAPMRecon_init(inp_file, FOV):
    cfg = CFG()

    # Phantom
    cfg.phantom.callback = "Phantom_Voxelized"      # name of function that reads and models phantom
    cfg.phantom.projectorCallback = "C_Projector_Voxelized" # name of function that performs projection through phantom
    cfg.phantom.filename = 'CatSim_logo_1024.json'  # phantom filename, not actually used in AAPM Recon
    cfg.phantom.centerOffset = [0.0, 0.0, 0.0]      # offset of phantom center relative to origin (in mm)
    cfg.phantom.scale = 1                           # re-scale the size of phantom
    if platform.system() == "Linux":
        cfg.phantom.projectorNumThreads = 4
    elif platform.system() == "Windows":
        cfg.phantom.projectorNumThreads = 1
    else:
        cfg.phantom.projectorNumThreads = 1
    
    # physics
    cfg.physics.energyCount = 12                    # number of energy bins
    cfg.physics.monochromatic = -1                  # -1 for polychromatic (see protocol.cfg);
    cfg.physics.colSampleCount = 1                  # number of samples of detector cells in lateral direction
    cfg.physics.rowSampleCount = 1                  # number of samples of detector cells in longitudinal direction
    cfg.physics.srcXSampleCount = 2                 # number of samples of focal spot in lateral direction
    cfg.physics.srcYSampleCount = 2                 # number of samples of focal spot cells in longitudinal direction
    cfg.physics.viewSampleCount = 2                 # number of samples of each view angle range in rotational direction
    cfg.physics.recalcDet = 0                       # recalculate detector geometry
    cfg.physics.recalcSrc = 0                       # recalculate source geometry and relative intensity
    cfg.physics.recalcRayAngle = 0                  # recalculate source-to-detector-cell ray angles
    cfg.physics.recalcSpec = 0                      # recalculate spectrum
    cfg.physics.recalcFilt = 0                      # recalculate filters
    cfg.physics.recalcFlux = 0                      # recalculate flux
    cfg.physics.recalcPht = 0                       # recalculate phantom
    cfg.physics.enableQuantumNoise = 1              # enable quantum noise
    cfg.physics.enableElectronicNoise = 1           # enable electronic noise
    cfg.physics.rayAngleCallback = "Detector_RayAngles_2D" # name of function to calculate source-to-detector-cell ray angles
    cfg.physics.fluxCallback = "Detection_Flux"     # name of function to calculate flux
    cfg.physics.scatterCallback = "Scatter_ConvolutionModel"                # name of function to calculate scatter
    cfg.physics.scatterKernelCallback = ""          # name of function to calculate scatter kernel ("" for default kernel)
    cfg.physics.scatterScaleFactor = 1              # scale factor, 1 appropriate for 64-mm detector and 20-cm water
    cfg.physics.callback_pre_log = "Scatter_Correction"
    cfg.physics.prefilterCallback = "Detection_prefilter" # name of function to calculate detection pre-filter
    cfg.physics.crosstalkCallback = "CalcCrossTalk" # name of function to calculate X-ray crosstalk in the detector
    cfg.physics.col_crosstalk = 0.025
    cfg.physics.row_crosstalk = 0.02
    cfg.physics.opticalCrosstalkCallback = "CalcOptCrossTalk" # name of function to calculate X-ray crosstalk in the detector
    cfg.physics.col_crosstalk_opt = 0.04
    cfg.physics.row_crosstalk_opt = 0.045
    cfg.physics.lagCallback = ""                    # name of function to calculate detector lag
    cfg.physics.opticalCrosstalkCallback = ""       # name of function to calculate optical crosstalk in the detector
    cfg.physics.DASCallback = "Detection_DAS"       # name of function to calculate the detection process
    cfg.physics.outputCallback = "WriteRawView"     # name of function to produce the simulation output
    cfg.physics.callback_post_log = 'Prep_BHC_Accurate'
    cfg.physics.EffectiveMu = 0.2
    cfg.physics.BHC_poly_order = 5
    cfg.physics.BHC_max_length_mm = 300
    cfg.physics.BHC_length_step_mm = 10
    
    # protocol
    cfg.protocol.scanTypes = [1, 1, 1]              # flags for airscan, offset scan, phantom scan
    cfg.protocol.scanTrajectory = "Gantry_Helical"  # name of the function that defines the scanning trajectory and model
    cfg.protocol.viewsPerRotation = 1000            # total numbers of view per rotation
    cfg.protocol.viewCount = 1000                   # total number of views in scan
    cfg.protocol.startViewId = 0                    # index of the first view in the scan
    cfg.protocol.stopViewId = cfg.protocol.startViewId + cfg.protocol.viewCount - 1 # index of the last view in the scan
    cfg.protocol.airViewCount = 1                   # number of views averaged for air scan
    cfg.protocol.offsetViewCount = 1                # number of views averaged for offset scan
    cfg.protocol.rotationTime = 1.0                 # gantry rotation period (in seconds)
    cfg.protocol.rotationDirection = 1              # gantry rotation direction (1=CW, -1 CCW, seen from table foot-end)
    cfg.protocol.startAngle = 0                     # relative to vertical y-axis (n degrees)
    cfg.protocol.tableSpeed = 0                     # speed of table translation along positive z-axis (in mm/sec)
    cfg.protocol.startZ = 0                         # start z-position of table
    cfg.protocol.tiltAngle = 0                      # gantry tilt angle towards negative z-axis (in degrees)
    cfg.protocol.wobbleDistance = 0.0               # focalspot wobble distance
    cfg.protocol.focalspotOffset = [0, 0, 0]        # focalspot position offset
    cfg.protocol.mA = 500                           # tube current (in mA)
    cfg.protocol.spectrumCallback = "Spectrum"      # name of function that reads and models the X-ray spectrum
    cfg.protocol.spectrumFilename = "xcist_kVp120_tar7_bin1.dat" # name of the spectrum file
    cfg.protocol.spectrumUnit_mm = 1;               # Is the spectrum file in units of photons/sec/mm^2/<current>?
    cfg.protocol.spectrumUnit_mA = 1;               # Is the spectrum file in units of photons/sec/<area>/mA?
    cfg.protocol.spectrumScaling = 1                # scaling factor, works for both mono- and poly-chromatic spectra
    cfg.protocol.bowtie = "large.txt"               # name of the bowtie file (or [] for no bowtie)
    cfg.protocol.filterCallback = "Xray_Filter"     # name of function to compute additional filtration
    cfg.protocol.flatFilter = ['air', 0.001]        # additional filtration - materials and thicknesses (in mm)
    cfg.protocol.dutyRatio = 1.0                    # tube ON time fraction (for pulsed tubes)
    cfg.protocol.maxPrep = -1                       # set the upper limit of prep, non-positive will disable this feature
    
    # Scanner
    cfg.scanner.detectorCallback = "Detector_ThirdgenCurved" # name of function that defines the detector shape and model
    cfg.scanner.sid = 550.0                         # source-to-iso distance (in mm)
    cfg.scanner.sdd = 950.0                         # source-to-detector distance (in mm)
    cfg.scanner.detectorColsPerMod = 1              # number of detector columns per module
    cfg.scanner.detectorRowsPerMod = 1              # number of detector rows per module
    cfg.scanner.detectorColOffset = -1.25             # detector column offset relative to centered position (in detector columns)
    cfg.scanner.detectorRowOffset = 0.0             # detector row offset relative to centered position (in detector rows)
    cfg.scanner.detectorColSize = 1.0               # detector column pitch or size (in mm)
    cfg.scanner.detectorRowSize = 1.0               # detector row pitch or size (in mm)
    cfg.scanner.detectorColCount = 900              # total number of detector columns
    cfg.scanner.detectorRowCount = cfg.scanner.detectorRowsPerMod     # total number of detector rows
    cfg.scanner.detectorPrefilter = []              # detector filter 
    cfg.scanner.focalspotCallback = "SetFocalspot"  # name of function that defines the focal spot shape and model
    cfg.scanner.focalspotData = "vct_large_fs.npz"  # Parameterize the model
    cfg.scanner.targetAngle = 7.0                   # target angle relative to scanner XY-plane (in degrees)
    cfg.scanner.focalspotWidth = 1.0
    cfg.scanner.focalspotLength = 1.0
    cfg.scanner.focalspotWidthThreshold =0.5
    cfg.scanner.focalspotLengthThreshold =0.5
    
    # Detector
    cfg.scanner.detectorMaterial = "GOS"            # detector sensor material
    cfg.scanner.detectorDepth = 3.0                 # detector sensor depth (in mm)
    cfg.scanner.detectionCallback = "Detection_EI"  # name of function that defines the detection process (conversion from X-rays to detector signal)
    cfg.scanner.detectionGain = 0.1                 # factor to convert energy to electrons (electrons / keV)
    cfg.scanner.detectorColFillFraction = 0.9       # active fraction of each detector cell in the column direction
    cfg.scanner.detectorRowFillFraction = 0.9       # active fraction of each detector cell in the row direction
    cfg.scanner.eNoise = 25                         # standard deviation of Gaussian electronic noise (in electrons)
    
    # recon
    cfg.recon.fov = FOV                           # diameter of the reconstruction field-of-view (in mm)
    cfg.recon.imageSize = 512                       # number of columns and rows to be reconstructed (square)
    cfg.recon.sliceCount = 1                        # number of slices to reconstruct
    cfg.recon.sliceThickness = 0.579                 # reconstruction slice thickness AND inter-slice interval (in mm)
    cfg.recon.centerOffset = [0.0, 0.0, 0.0]        # reconstruction offset relative to center of rotation (in mm)
    cfg.recon.reconType = 'fdk_equiAngle'           # Name of the recon function to call
    cfg.recon.kernelType = 'standard'               # 'R-L' for the Ramachandran-Lakshminarayanan (R-L) filter, rectangular window function
    cfg.recon.startAngle = 0                        # in degrees; 0 is with the X-ray source at the top
    cfg.recon.unit = 'HU'                           # '/mm', '/cm', or 'HU'
    cfg.recon.mu = 0.02                             # in /mm; typically around 0.02/mm
    cfg.recon.huOffset = -1000                      # unit is HU, -1000 HU by definition but sometimes something else is preferable
    cfg.recon.printReconParameters = False          # Flag to print the recon parameters
    cfg.recon.saveImageVolume = True                # Flag to save recon results as one big file
    cfg.recon.saveSingleImages = False              # Flag to save recon results as individual imagesrecon.printReconParameters = False      # Flag to print the recon parameters
    cfg.recon.displayImagePictures = False          # Flag to display the recon results as .png images
    cfg.recon.saveImagePictureFiles = False         # Flag to save the recon results as .png images
    cfg.recon.displayImagePictureAxes = False       # Flag to display the axes on the .png images
    cfg.recon.displayImagePictureTitles = False     # Flag to display the titles on the .png images

    cfg.resultsName = os.path.splitext(inp_file)[0]

    if cfg.physics.monochromatic>0:
        cfg.recon.mu = xc.GetMu('water', cfg.physics.monochromatic)[0]/10

    cfg.do_Recon = 1
    cfg.waitForKeypress = 0

    return cfg

def feval(funcName, *args):
    try:
        md = __import__(funcName)
    except:
        md = __import__("gecatsim.pyfiles."+funcName, fromlist=[funcName])  # equal to: from gecatsim.foo import foo
    strip_leading_module = '.'.join(funcName.split('.')[1:])
    func_name_only = funcName.split('.')[-1]

    if len(strip_leading_module) > 0:
        eval_name = f"md.{strip_leading_module}.{func_name_only}"
    else:
        eval_name = f"md.{func_name_only}"
    return eval(eval_name)(*args)

def recon(cfg, inp_data):
    # cfg.recon.reconType is the recon function's name
    imageVolume3D = feval("gecatsim.reconstruction.pyfiles." + cfg.recon.reconType, cfg, inp_data)
    imageVolume3D = imageVolume3D*(1000/(cfg.recon.mu)) + cfg.recon.huOffset
    return imageVolume3D

def AAPMRecon_main(cfg, inp_data):
    recon_img = recon(cfg, inp_data)
    return recon_img

def rawwrite(fname, data):
    with open(fname, 'wb') as fout:
        fout.write(data)

def extract_number(file_name):
    match = re.search(r'_img(\d+)', file_name)
    if match:
        return int(match.group(1))
    else:
        return -1  # 숫자가 없는 경우 처리

def sorted_path(path):
    file_path = os.listdir(path)
    head_path = [d for d in file_path if 'head' in d]
    body_path = [d for d in file_path if 'body' in d]
    head_path = sorted(head_path, key=extract_number)
    body_path = sorted(body_path, key=extract_number)
    return body_path + head_path

#%%
def sample_and_test(args):
    torch.manual_seed(42)
    # device = 'cuda:0'
    torch.cuda.set_device(args.gpu_chose)
    device = torch.device('cuda:{}'.format(args.gpu_chose))
    epoch_chosen=args.which_epoch
    
    to_range_0_1 = lambda x: (x + 1.) / 2.

    image_size= args.image_size 
    input_path= args.input_path

    #loading dataset
    transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Resize((image_size, image_size))
    ])

    metal_dir = os.path.join(input_path, 'LI/sinogram')
    nometal_dir = os.path.join(input_path, 'Target/sinogram')
    img_dir = os.path.join(input_path, 'Target/recon_img')
    mask_dir = os.path.join(input_path, 'metal_trace_mask')
    dataset = CustomDataset(metal_dir, nometal_dir, mask_dir, img_dir, transform=transform, option='all')

    # n_val = int(len(dataset) * 0.1)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=8)
    
    #Initializing and loading network
    # gen_diffusive_1 = NCSNpp(args).to(device)
    gen_diffusive_2 = NCSNpp(args).to(device)
    
    exp = args.exp
    output_dir = args.output_path
    exp_path = os.path.join(output_dir,exp)
    # exp_path = 'D:/Syndiff'
    
    print('load checkpoint ...')
    checkpoint_file = exp_path + "/{}_{}.pth"
    # checkpoint_file = 'D:/code/Syndiff/gen_diffusive_85.pth'
    load_checkpoint(checkpoint_file, gen_diffusive_2,'gen_diffusive',epoch=str(epoch_chosen), device = device)

    T = get_time_schedule(args, device)
    
    pos_coeff = Posterior_Coefficients(args, device)
    #save_dir = exp_path + "/eval/epoch_{}".format(epoch_chosen)
    save_dir = exp_path + "/generated_samples_LI/epoch_{}".format(epoch_chosen)
    
    # crop = transforms.CenterCrop((256, 152))
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # metal_sino_path = os.listidir(metal_dir)
    # metal_sino_path = os.listidir(metal_dir)

    loss1 = np.zeros((1, len(data_loader))) # (1,90)
    loss2 = np.zeros((1, len(data_loader))) 

    loss1_sino = np.zeros((1, len(data_loader))) # (1,90)
    loss2_sino = np.zeros((1, len(data_loader))) 
    # syn_im1=np.zeros((image_size,image_size,len(data_loader))) # (128, 128, 90)
    # syn_im2=np.zeros((image_size,image_size,len(data_loader)))
    
    ## File name 만들기
    # path = args.file_name_path
    # with open(path, 'r') as f:
    #     files_name = json.load(f)
    # files_name = [f.split('_')[3] for f in files_name]
    # files = os.listdir(path)
    # files_name = [f[:-4] for f in files]

    for iteration, batch in enumerate(data_loader): 
        
        real_data = batch['nometal'].to(device, non_blocking=True)
        source_data = batch['metal'].to(device, non_blocking=True)
        real_img = batch['recon_img']#.to(device, non_blocking=True)
        label = batch['label']#.to(device, non_blocking=True)
        name = batch['name']#.to(device, non_blocking=True)
        name = ''.join(name).split('_')[3]

        # x2_t = torch.cat((torch.randn_like(real_data),source_data),axis=1)

        # #diffusion steps
        # fake_sample2 = sample_from_model(pos_coeff, gen_diffusive_2, args.num_timesteps, x2_t, T, args)
        # fake_sample2_ori = fake_sample2.cpu().numpy()
        # fake_sample2 = transforms.functional.resize(fake_sample2, (900,1000))

        # # if label == 0:
        # #     FOV = 400
        # # elif label == 1:
        # #     FOV = 220.16
        
        # # inp_data = rawread(inp_file, [1000, 1, 900], 'float')
        # fake_sample2 = fake_sample2.cpu().numpy().squeeze(0).transpose(2,0,1)
        # print(fake_sample2.shape)
        # cfg = AAPMRecon_init('training_body_metalart_sino3_900x1000.raw', FOV)
        # recon_img = AAPMRecon_main(cfg, fake_sample2)

        real_data = real_data.cpu().numpy()
        source_data = source_data.cpu().numpy()
        # fake_sample2_ori = (fake_sample2_ori - fake_sample2_ori.min()) / (fake_sample2_ori.max() - fake_sample2_ori.min()) * 255
        # fake_sample2 = (fake_sample2 - fake_sample2.min()) / (fake_sample2.max() - fake_sample2_ori.min()) * 255
        real_data = (real_data - real_data.min()) / (real_data.max() - real_data.min()) * 255
        source_data = (source_data - source_data.min()) / (source_data.max() - source_data.min()) * 255
        # recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min()) * 255
        # real_img = (real_img - real_img.min()) / (real_img.max() - real_img.min()) * 255

        # rawwrite(os.path.join(args.save_path, save), recon_img)
        # print(os.path.join(args.save_path, save))

        # print(fake_sample2.shape, recon_img.shape)
        # exit()

        # fake_sample2 = to_range_0_1(fake_sample2) ; fake_sample2 = fake_sample2/fake_sample2.max()
        # real_data = to_range_0_1(real_data) ; real_data = real_data/real_data.max()
        # source_data = to_range_0_1(source_data); source_data = source_data/source_data.max() 
        #fake_sample2 = crop(fake_sample2) 
        #real_data = crop(real_data)
        #source_data = crop(source_data
        # syn_im2[:,:,iteration]=np.squeeze(fake_sample2.cpu().numpy())
        
        # 이미지 저장 
        # print_list = torch.cat((source_data, fake_sample2, real_data),axis=-1)
        # torchvision.utils.save_image(recon_img, save_dir+'/epoch{}_recon_{}.jpg'.format(epoch_chosen, label), normalize=True)
        # print(type(recon_img))
        # print(type(recon_img.astype('uint8')))

        source_data = source_data.reshape(512,512)
        # fake_sample2 = fake_sample2.reshape(512,512)
        real_data = real_data.reshape(512,512)
        # recon_img = recon_img.reshape(512,512)
        # real_img = real_img.reshape(512,512)
        
        # print(real_img.shape, recon_img.shape)
        # exit()
        # real_img = real_img.cpu().numpy()
        # save = np.concatenate((real_data.astype('uint8'), fake_sample2_ori.astype('uint8'), recon_img.astype('uint8'), real_img.astype('uint8')), axis=1)
        # print(fake_sample2_ori.shape, real_data.shape)
        save = np.concatenate((real_data.astype('uint8'), source_data.astype('uint8')), axis=1)
        save_img = Image.fromarray(save, mode='L')
        save_img.save(save_dir+'/epoch{}_{}_{}.jpg'.format(epoch_chosen, name, int(label)))

        # loss1[0, iteration] = ssim(real_img, recon_img)
        # loss2[0, iteration] = psnr(real_img, recon_img)#.cpu().numpy()
        loss1_sino[0, iteration] = ssim(real_data, source_data)
        loss2_sino[0, iteration] = psnr(real_data, source_data)#.cpu().numpy()
        print('SSIM:', loss1_sino[0, iteration], 'PSNR:', loss2_sino[0, iteration])
        

    print('Final SSIM: ',np.nanmean(loss1_sino))
    np.save('{}/ssim.npy'.format(save_dir), loss1_sino)

    print('Final PSNR: ',np.nanmean(loss2_sino))
    np.save('{}/psnr.npy'.format(save_dir), loss2_sino)

    # f = h5py.File(save_dir + '/im_syn.mat',  "w")
    # f.create_dataset('images_'+args.contrast1+'syn', data=syn_im1)
    # f.create_dataset('images_'+args.contrast2+'syn', data=syn_im2)
    # f.close()
            
#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('syndiff parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                            help='whether or not compute FID')
    parser.add_argument('--epoch_id', type=int,default=1000)
    parser.add_argument('--num_channels', type=int, default=2,
                            help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--beta_min', type=float, default= 0.1,
                            help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                            help='beta_max for diffusion')
    
    
    parser.add_argument('--num_channels_dae', type=int, default=64,
                            help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                            help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                            help='channel multiplier')

    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
                            help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                            help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                            help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                            help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                            help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                            help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                            help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                            help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                            help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true',default=False)
    
    #geenrator and training
    parser.add_argument('--exp', default='trial1', help='name of experiment')
    parser.add_argument('--input_path', default='D:/CT_MAR_RAW/Test', help='path to input data')
    parser.add_argument('--output_path', default='', help='path to output saves')

    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=512,
                            help='size of image')

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)
    
    
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4, help='sample generating batch size')
    
    #optimizaer parameters    
    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--beta1', type=float, default=0.5,
                            help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                            help='beta2 for adam')
    parser.add_argument('--contrast1', type=str, default='Baseline',
                        help='contrast selection for model')
    parser.add_argument('--contrast2', type=str, default='Target',
                        help='contrast selection for model')
    parser.add_argument('--which_epoch', type=int, default=50)
    parser.add_argument('--gpu_chose', type=int, default=0)


    parser.add_argument('--source', type=str, default='T2',
                        help='source contrast'), 
    parser.add_argument('--add_path', type=str, default='val'),
    parser.add_argument('--file_name_path', type=str, default='/home/heeryung/MAR/Syndiff/head_val_set.json')
    args = parser.parse_args()
    
    sample_and_test(args)
    

