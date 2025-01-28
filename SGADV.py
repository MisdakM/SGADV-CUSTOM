# Targeted attack to face recognition system
import eagerpy as ep
import torch.nn as nn
import codecs
import time
import torch
from math import ceil
from torch import Tensor
from scipy.io import loadmat
from facenet_pytorch import InceptionResnetV1
from insightface import iresnet100
import foolbox.attacks as attacks
from foolbox.models import PyTorchModel
from foolbox.utils import samples, FMR, cos_similarity_score
import lpips
from torchvision.utils import save_image
from scipy.io import savemat
from skimage.metrics import structural_similarity as ssim
import numpy as np

def main() -> None:
    # Settings
    samplesize = int(10)
    subject = int(2)
    foldersize = int(samplesize*subject/2)
    source = "lfw" # lfw, CelebA-HQ
    target = "CelebA-HQ" # lfw, CelebA-HQ
    dfr_model = 'facenet' # facenet, insightface
    threshold = 0.7032619898135847 # facenet: 0.7032619898135847; insightface: 0.5854403972629942
    loss_type = 'ST' #'ST', 'C-BCE'
    epsilons = 0.03
    steps = 10
    step_size = 0.001
    convergence_threshold = 0.0001

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    totalsize = samplesize*subject
    batchsize = 20
    
    attack_model = attacks.LinfPGD # dummy attack model, for keeping logs intact

    # Initialize the ATN model
    input_size = 3 * 250 * 250  # Assuming input images are 250x250 with 3 channels
    atn_model = attacks.SimpleAutoencoder(hidden_size=10).to(device)
    optimizer = torch.optim.Adam(atn_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()  # Use MSE loss for simplicity
    
    
    # Log
    log_time = time.strftime("%Y%m%d%H%M%S",time.localtime())
    f = codecs.open(f'results/logs/{loss_type}_{source}_{target}_{dfr_model}_{attack_model.__module__}_{log_time}.txt','a','utf-8')
    f.write(f"samplesize = {samplesize}, subject = {subject}, source = {source}, target = {target}, dfr_model = {dfr_model}, threshold = {threshold}\n")
    f.write(f"attack_model = {attack_model}, loss_type = {loss_type}, epsilons = {epsilons}, steps = {steps}, step_size = {step_size}, convergence_threshold = {convergence_threshold}, batchsize = {batchsize}\n")
    
    # Target Model
    if dfr_model=='insightface':
        model = iresnet100(pretrained=True).eval()
    elif dfr_model=='facenet':
        model = InceptionResnetV1(pretrained='vggface2').eval()

    mean=[0.5]*3
    std=[0.5]*3
    preprocessing = dict(mean = mean, std = std, axis=-3)
    bounds=(0, 1)
    fmodel = PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)

    # Load data
    features_tmp = loadmat(f'mat/{target}_{dfr_model}_templates.mat')[f'{target}_{dfr_model}_templates']
    features = Tensor(features_tmp)
    source_images, _ = ep.astensors(*samples(fmodel, dataset=f"{source}_test", batchsize=subject*samplesize, model_type=dfr_model))

    # Input data
    attack_index = list(range(samplesize*subject))
    attack_images = source_images[attack_index]
    target_index = list(range(foldersize,foldersize*2))+list(range(samplesize,foldersize))+list(range(0,samplesize))
    target_features = features[target_index]
    del source_images
    
    # Run attack
    raw_advs = Tensor([]).to(device)
    advs_features = Tensor([]).to(device)
    time_cost = 0

    print(f"Total batches: {ceil(totalsize / batchsize)}")
    
    for i in range(ceil(totalsize / batchsize)):
        print(f"Batch: {i+1}")
        start = i * batchsize
        if i == ceil(totalsize / batchsize) - 1:
            batchsize = totalsize - batchsize * i

        batch_images = attack_images[start:start + batchsize].raw.to(device)
        batch_targets = target_features[start:start + batchsize].to(device)

        start_time = time.time()
        
        # Train the ATN
        atn_model.train()
        for _ in range(steps):  # Train for a few iterations
            optimizer.zero_grad()

            perturbations = atn_model(batch_images)  # Generate perturbations
            advs = batch_images + perturbations  # Add perturbations
            advs = torch.clamp(advs, 0, 1)  # Ensure valid image values

            # Calculate loss based on target features
            advs_features = fmodel(advs)  # No need to reshape as images are already in correct format
            loss = criterion(advs_features, batch_targets)

            # print(f'advs_features shape: {advs_features.shape}')
            # print(f'batch_targets shape: {batch_targets.shape}')
            
            print(f'loss: {loss.item()}')

            loss.backward()
            optimizer.step()
        
        end_time = time.time()
        time_cost += end_time - start_time

        # Evaluate adversarial examples
        atn_model.eval()
        with torch.no_grad():
            perturbations = atn_model(batch_images)
            advs = torch.clamp(batch_images + perturbations, 0, 1)
            advs_features_tmp = fmodel(advs)  # No need to reshape as images are already in correct format
            print(f"advs_features_tmp shape: {advs_features_tmp.shape}")

        raw_advs = torch.cat((raw_advs, advs), 0)
        advs_features = torch.cat((advs_features, advs_features_tmp), 0)

        print(f"advs_features shape after batch {i+1}: {advs_features.shape}")

        del advs_features_tmp

    
    del fmodel, model
    print(f"Attack costs {time_cost}s")
    f.write(f"Attack costs {time_cost}s\n")
    
    # Save advs template
    adv_template = advs_features.detach().cpu().numpy()
    savemat(f'mat/{loss_type}_{source}_{target}_{dfr_model}_templates.mat', mdict={f"{loss_type}_{source}_{target}_{dfr_model}_templates": adv_template})
    
    # Save advs
    save_image(raw_advs[10], f'results/images/{loss_type}_{source}_{target}_{dfr_model}_{log_time}_adv.jpg')
    noise = (raw_advs[10]-attack_images[10].raw+bounds[1]-bounds[0])/((bounds[1]-bounds[0])*2)
    save_image(noise, f'results/images/{loss_type}_{source}_{target}_{dfr_model}_{log_time}_noise.jpg')
    del noise
    
    # Compute SSIM
    attack_images_np = attack_images.raw.cpu().numpy().transpose(0, 2, 3, 1)
    
    print(f"attack_images_np shape: {attack_images_np.shape}")
    print(f"raw_advs shape: {raw_advs.cpu().numpy().shape}")

    raw_advs_np = raw_advs.cpu().numpy().transpose(0, 2, 3, 1)
    ssim_scores = [ssim(attack_images_np[i], raw_advs_np[i], win_size=7, channel_axis=-1, data_range=1.0) for i in range(attack_images_np.shape[0])]
    ssim_score = np.mean(ssim_scores)
    print(f"SSIM = {ssim_score}")
    f.write(f"SSIM = {ssim_score}\n")
    
    # Compute LPIPS
    loss_fn = lpips.LPIPS(net='alex')
    if bounds != (-1, 1):
        attack_images = attack_images.raw.cpu()*2-1
        raw_advs = raw_advs.cpu()*2-1
    lpips_score = loss_fn.forward(attack_images,raw_advs).mean()
    print(f"LPISP = {lpips_score}")
    f.write(f"LPISP = {lpips_score}\n")
    del attack_images, raw_advs, loss_fn
    
    # Compute dissimilarity - helps to tell how different the adversarial examples are from the target templates
    print(f"advs_features shape: {advs_features.shape}")
    print(f"target_features shape: {target_features.shape}")
    dissimilarity = 1-cos_similarity_score(advs_features,target_features).mean()
    print(f"Dissimilarity = {dissimilarity}")
    f.write(f"Dissimilarity = {dissimilarity}\n")
    
    # Compute FMR
    fmr_target, fmr_renew = FMR(advs_features, target_features, threshold, samplesize)
    print("Attack performance:")
    f.write("Attack performance:\n")
    print(f" advs vs targets: FMR = {fmr_target * 100:.2f}%")
    f.write(f" advs vs targets: FAR = {fmr_target * 100:.2f}%\n")
    print(f" advs vs renews: FMR = {fmr_renew * 100:.2f}%")
    f.write(f" advs vs renews: FAR = {fmr_renew * 100:.2f}%\n")

    f.close()

if __name__ == "__main__":
    main()
