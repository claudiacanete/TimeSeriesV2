r"""
File to extract csv images from csv files for mnist dataset.
"""

import os
import cv2
from tqdm import tqdm
import numpy as np
import _csv as csv
import yfinance as yf
import pandas as pd
import yaml
import argparse
import math
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='Arguments for ddpm training')
parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
args = parser.parse_args()
with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
model_config = config['model_params']           
horizon=model_config['time_horizon']

def extract_images(save_dir, csv_fname):
    assert os.path.exists(save_dir), "Directory {} to save images does not exist".format(save_dir)
    assert os.path.exists(csv_fname), "Csv file {} does not exist".format(csv_fname)
    with open(csv_fname) as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            im = np.zeros((horizon))
            im[:] = list(row[1:])
            
            size_image=int(math.sqrt(horizon))
            im = im.reshape((size_image,size_image))
            if not os.path.exists(os.path.join(save_dir, row[0])):
                os.mkdir(os.path.join(save_dir, row[0]))
            cv2.imwrite(os.path.join(save_dir, row[0], '{}.png'.format(idx)), im)
            if idx % 1000 == 0:
                print('Finished creating {} images in {}'.format(idx+1, save_dir))       
df = yf.download(tickers=['^GSPC'],start = "1928-01-01",end = "2024-01-01",interval="1d") 
print(df)  
# Select 'Close' column and convert it to a list
close_data = df['Close'].tolist()
#print(close_data)

max_old=max(close_data)
print('max_old',max_old)
min_old=min(close_data)
print('min_old',min_old)


# Split the list into chunks of 30
#we will have to divide by this when converting from image to time series again
chunks = [[1]+close_data[i:i + horizon] for i in range(0, len(close_data), int(horizon/10))]
chunks_not_normalized=pd.DataFrame(chunks)
chunks_not_normalized=chunks_not_normalized.dropna()
#print('chunks',chunks)
# Normalize the chunks
def normalize(chunk, max_old, min_old):
    i=1
    while i < len(chunk):
        chunk[i]=(chunk[i]-min_old)/(max_old-min_old)*255
        #chunk[i]=(chunk[i]-min_old)/(max_old-min_old)
        i=i+1
    return chunk
chunks = [normalize(chunk, max_old, min_old) for chunk in chunks]
#print('chunks',chunks)

# Convert chunks to a DataFrame
df_chunks = pd.DataFrame(chunks)
# Remove NaN values
df_chunks_nonan=df_chunks.dropna()

#df_str = df_chunks_nonan.apply(lambda row: ','.join(row.astype(str)), axis=1)
#print(df_str)
# Write DataFrame to Excel
#df_str.to_excel('data/timeseries.xlsx', index=False) 
# Randomly shuffle the index of nba.
random_indices = df_chunks_nonan.sample(frac=1, random_state=42).index
# Set a cutoff for how many items we want in the test set (in this case 1/3 of the items)
test_cutoff = math.floor(len(df_chunks_nonan)*0.3)
# Generate the test set by taking the first 1/3 of the randomly shuffled indices.
df_chunks_nonan_test = df_chunks_nonan.loc[random_indices[1:test_cutoff]]
# Generate the train set with the rest of the data.
df_chunks_nonan_train = df_chunks_nonan.loc[random_indices[test_cutoff:]]
df_chunks_nonan_train.to_excel('data/train.xlsx', index=False)  
df_chunks_nonan_test.to_excel('data/test.xlsx', index=False)     
read_file = pd.read_excel ("data/train.xlsx")

# Write the dataframe object 
# into csv file 
read_file.to_csv ("data/train.csv",  
                  index = None, 
                  header=True)    
import os
os.remove("data/train.xlsx")

read_file = pd.read_excel ("data/test.xlsx")
read_file.to_csv ("data/test.csv",  
                  index = None, 
                  header=True)    
os.remove("data/test.xlsx")
if __name__ == '__main__':
 extract_images('data/train/images', 'data/train.csv')
 extract_images('data/test/images', 'data/test.csv')