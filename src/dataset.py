import pandas as pd 
import os
#
import glob
import os
import random
import time
from threading import Thread

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, root, transform=None, unaligned=False, mode="train", A="No Finding", B="Pneumonia"):
        efficentMode = True # optional
        cwd = os.getcwd()
        db_list = f"{cwd}\\data\\Data_toy.csv"
        self.diseases_list = set()
        self.disease_count = {} # optional
        self.A = A
        self.B = B
        df = pd.read_csv(db_list)
        # print(df.head(10))
        if not efficentMode:
            for i, row in df.iterrows():
                    if '|' in row[1]:
                        diseases = row[1].split('|')
                        for d in diseases:
                            self.diseases_list.add(d)
                            if d not in df:
                                df[d] = 0
                                df.loc[i, d] = 1
                            else:
                                df.loc[i, d] = 1
                    else:
                        d = str(row[1])
                        self.diseases_list.add(d)
                        self.disease_count[d] = self.disease_count.get(d, 0) + 1
                        if d not in df:
                            df[d] = 0
                            df.loc[i, d] = 1
                        else:
                            df.loc[i, d] = 1
        else:
            A_df = df[df["Finding Labels"] == "No Finding"]
            B_df = df[df["Finding Labels"].str.match(B)]
        df.drop(columns=['Finding Labels'], inplace=True) # Optional. Disable if computation takes too long time
        # print(diseases_list)




        # A list of image_indices for every disease Hernia_List = [00000003_001.png, 00000003_002.png, 00000003_003.png....]


        # After getting one-hot encoding for each diseases, let's create two lists of image indices A and B. For now A will be No-Finding
        # and B will be a disease Pneumonia 
        self.images = {}

        # print(images)
        if not efficentMode:
            for i, d in enumerate(self.diseases_list):
                # print(f"--------------------{d}------------------------")
                self.images[d] = (df[df[d] == 1]["Image Index"].to_string(index=False).split("\n"))

        else:
            self.images[A] = A_df["Image Index"].to_string(index=False).split("\n")
            self.images[B] = B_df["Image Index"].to_string(index=False).split("\n")
        # print(self.images[B])
        # print(self.images)
###

        self.transform = transform
        self.unaligned = unaligned

        # self.files_A = sorted(glob.glob(os.path.join(root, f"{mode}/A") + "/*.*"))  # 
        # self.files_B = sorted(glob.glob(os.path.join(root, f"{mode}/B") + "/*.*")) # images

        


    def __getitem__(self, index):
        A_path = f"{os.getcwd()}\\data\\images\\"+self.images[self.A][index % len(self.images[self.A])]
        if not self.unaligned:
            B_path = f"{os.getcwd()}\\data\\images\\"+self.images[self.B][random.randint(0, len(self.images[self.B]) - 1)]
        else:
            B_path = f"{os.getcwd()}\\data\\images\\"+self.images[self.B][index % len(self.images[self.B])]
        
        print(A_path)
        print(B_path)       
        
        item_A = self.transform(Image.open(A_path))
        item_B = self.transform(Image.open(B_path))


        return {"A": item_A, "B": item_B}



    def __len__(self):
        return max(len(self.images[self.A]), len(self.images[self.B]))



    # pd.concat([s.str.split('|').apply(pd.Series, index=c.split('|')) 
    #                  for c, s in df.set_index('Image Index').iteritems()],
    #                  axis=1).reset_index()