import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
import pandas as pd
from utils.extract_mnist_images import max_old, min_old
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils.extract_mnist_images import max_old, min_old
from utils.extract_mnist_images import chunks_not_normalized


def sample(model, scheduler, train_config, model_config, diffusion_config):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    df_list = []
    xt = torch.randn((train_config['num_samples'],
                      model_config['im_channels'],
                      model_config['im_size'],
                      model_config['im_size'])).to(device)
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        
        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        
        # Save x0
        ims = torch.clamp(xt, max=255).detach().cpu()
        
        #ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=train_config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage()(grid)
        
        if not os.path.exists(os.path.join(train_config['task_name'], 'samples')):
            os.mkdir(os.path.join(train_config['task_name'], 'samples'))
        img.save(os.path.join(train_config['task_name'], 'samples', 'x0_{}.png'.format(i)))
        img.close()
        # Convert tensor to DataFrame and append to list
        df = pd.DataFrame(ims.numpy().reshape(-1, model_config['im_channels'] * model_config['im_size'] * model_config['im_size']))
    
    print('ultimo df', df)#print('ims',ims)
    # Concatenate all DataFrames in the list
    df_all = df
    #print('df_all',df_all)
    min_new=df_all.min().min()
    print('min new',min_new)
    max_new=df_all.max().max()
    print('max_new', max_new)
    def denormalize(vector, max_old, min_old, max_new, min_new):
        #vector=(vector/255*(max_old-min_old)+min_old)
        vector=max_old-(((max_new-vector)/(max_new-min_new))*(max_old-min_old))
        return vector
    i=0
    print('df_all', df_all)
    rows, cols = df_all.shape
    while i<cols:
        #print('antes',df_all[i])
        df_all[i]=[denormalize(value, max_old, min_old, max_new, min_new) for value in df_all[i]]
        #print('despues',df_all[i])
        i=i+1
    #print('df_all',df_all)
    df_all=pd.DataFrame(df_all)
    

    df_all.to_excel('data/samples.xlsx', index=False)
    plt.plot(df_all.iloc[0])
    plt.plot(df_all.iloc[1])
    plt.plot(df_all.iloc[2])
    plt.plot(df_all.iloc[3])
    plt.title('Generated Samples')
    plt.show()

    pca = PCA(n_components=2)
    pca_original = pca.fit_transform(chunks_not_normalized)
    
    pca_generated=pca.fit_transform(df_all)
    plt.figure(figsize=(6, 5))
    plt.scatter(pca_original[:, 0], pca_original[:, 1], label='Original')
    plt.scatter(pca_generated[:, 0], pca_generated[:, 1], label='Generated')
    plt.title('PCA plot')
    plt.legend()
    plt.show()

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(chunks_not_normalized)
    tsne_results_generated = tsne.fit_transform(df_all)
    plt.figure(figsize=(6, 5))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], label='Original')
    plt.scatter(tsne_results_generated[:, 0], tsne_results_generated[:, 1], label='Generated')
    plt.title('t-SNE plot')
    plt.legend()
    plt.show()



def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    #print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    # Load model with checkpoint
    model = Unet(model_config).to(device)
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  train_config['ckpt_name']), map_location=device))
    model.eval()
    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    with torch.no_grad():
        sample(model, scheduler, train_config, model_config, diffusion_config)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    infer(args)