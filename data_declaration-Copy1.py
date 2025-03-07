'''The following module declares the Dataset objects required by torch to iterate over the data.'''
from enum import Enum
import glob
import pathlib

import numpy as np
import nibabel as nib
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from skimage import transform
import monai
import scipy.io as sio
#from monai.transforms import AddChannel, Compose, RandAffine, RandRotate90, RandFlip, apply_transform

class Task(Enum):
    '''
        Enum class for the two classification tasks
    '''
    NC_v_AD = 1
    sMCI_v_pMCI = 2

def get_ptid(path):
    '''Gets the image id from the file path string'''
    fname = path.stem
    ptid_str = ""
    #the I that comes before the id needs to be removed hence [1:]
    ptid_str = fname
    return ptid_str

def get_ptid1(path):
    '''Gets the image id from the file path string'''
    fname = path.stem
    ptid_str = ""
    #the I that comes before the id needs to be removed hence [1:]
    ptid_str = fname[7:17]
    return ptid_str

def get_ptid2(path):
    '''Gets the image id from the file path string'''
    fname = path.stem.split('_')[0][8:]
    ptid_str = ""
    #the I that comes before the id needs to be removed hence [1:]
    ptid_str = fname[:3] + '_' + fname[3] + '_' + fname[4:]
    return ptid_str


# def get_acq_year(im_data_id, im_df):
#     '''Gets the acquisition year from a pandas dataframe by searching the image id'''
#     acq_date = im_df[im_df['Image Data ID'] == im_data_id]["Acq Date"].iloc[0]
#     acq_year_str = ""

#     slash_count = 0
#     for char in acq_date:
#         if char == "/":
#             slash_count += 1

#         if slash_count == 2:
#             acq_year_str += char

#     return acq_year_str[1:]

def get_label(path, labels):
    '''Gets label from the path'''
    label_str = path.parent.stem
    label = None

    if label_str == labels[0]:
        label = np.array([0], dtype=np.double)
    elif label_str == labels[1]:
        label = np.array([1], dtype=np.double)
    return label

def get_mri(path, training):
    '''Gets a numpy array representing the mri object from a file path'''
    #mri = nib.load(str(path)).get_fdata()
    # mri = np.load(str(path))
    mri = sio.loadmat(str(path))
    mri = mri['data']
    #mri = transform.resize(mri,(148, 148, 148))
    mri = np.expand_dims(mri, axis=0)
    # if training:
        # mri = monai.transforms.RandAffine(prob=0.5, rotate_range=(0, 0, np.pi/4), scale_range=(0.9, 1.1), padding_mode='zeros')(mri)
        # mri = monai.transforms.RandFlip(prob=0.5, spatial_axis=0)(mri)
        # mri = monai.transforms.RandFlip(prob=0.5, spatial_axis=1)(mri)
        # mri = monai.transforms.RandFlip(prob=0.5, spatial_axis=2)(mri)
        # mri = monai.transforms.RandRotate90(prob=0.5, spatial_axes=(0, 1))(mri)
        # mri = monai.transforms.RandRotate90(prob=0.5, spatial_axes=(0, 2))(mri)
        # mri = monai.transforms.RandRotate90(prob=0.5, spatial_axes=(1, 2))(mri)
    #print(mri.shape)
    # mri = np.asarray(mri)[1:, 4:133 ,1:]
    mri = np.asarray(mri)
    # print('before',mri.shape)
    # mri = mri[:, 1:, 1:, 1:]
    # print('after',mri.shape)
    return mri



def get_clinical(sub_id, clin_df):
    '''Gets clinical features vector by searching dataframe for image id'''
    clinical = np.zeros(14)
    if sub_id in clin_df["PTID"].values:
        row = clin_df.loc[clin_df["PTID"] == sub_id].iloc[0]

        # GENDER
        if row["PTGENDER"] == "Male":
            clinical[0] = 1
        else:
            clinical[0] = 0

        # AGE
        clinical[1] = row["AGE"]
        # Education
        clinical[2] = row["PTEDUCAT"]



        # clinical[4] = row["RAVLT_immediate_bl"]
        # clinical[5] = row["CDRSB_bl"]
        #if row["PTAU_bl"].empty:
        #    clinical[6] = 0
        #else:
        # clinical[4] = row["PTAU_bl"]
        # clinical[6] = row["missing_PTAU_bl"]
        # clinical[7] = row["ABETA_bl"]
        # clinical[8] = row["TAU_bl"]

        #if row["FDG_bl"].empty:
        #    clinical[7] = 0
        #else:
        clinical[3] = row["FDG_bl"]
        clinical[4] = row["FDG_bl_missing"]

        clinical[5] = row["TAU_bl"]
        clinical[6] = row["TAU_bl_missing"]
        clinical[7] = row["PTAU_bl"]
        clinical[8] = row["PTAU_bl_missing"]
        clinical[9] = row["ABETA_bl"]
        clinical[10] = row["ABETA_bl_missing"]


        # APOE4 if missing use 000
        apoe4_allele = row["APOE4"]
        if row["APOE4_missing"] == 1:
            clinical[11] = 0
            clinical[12] = 0
            clinical[13] = 0
        else:  
            if apoe4_allele == 0:
                clinical[11] = 1
                clinical[12] = 0
                clinical[13] = 0
            elif apoe4_allele == 1:
                clinical[11] = 0
                clinical[12] = 1
                clinical[13] = 0
            elif apoe4_allele == 2:
                clinical[11] = 0
                clinical[12] = 0
                clinical[13] = 1
        
    
    
    else:
        print(sub_id)
    return clinical


class MRIDataset(Dataset):
    '''Provides an object for the MRI data that can be iterated.'''
    def __init__(self, root_dir, labels, training, transform=None):

        self.root_dir = root_dir  # root_dir="../data/"
        self.transform = transform
        self.directories = []
        self.len = 0
        self.labels = labels
        self.training = training
        self.clin_data = pd.read_csv("/home/cyliu/dataset/adni/ADNIMERGE_preprocessed.csv")
    
        train_dirs = []

        for label in labels:
            train_dirs.append(root_dir + label)

        for train_dir in train_dirs:
            for path in glob.glob(train_dir + "/*.mat"):
                self.directories.append(pathlib.Path(path))

        self.len = len(self.directories)

    def __len__(self):

        return self.len

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        repeat = True

        while repeat:
            try:
                path = self.directories[idx]
                im_id = get_ptid(path)
                mri = get_mri(path, self.training)
                clinical = get_clinical(im_id, self.clin_data)
                # print(mri.shape)

                label = get_label(path, self.labels)
                # print(label)
                sample = {'mri': mri, 'clinical': clinical, 'label': label}
                #sample = {'mri': mri, 'label':label}
                if self.transform:
                    sample = self.transform(sample)

                return sample

            except IndexError as index_e:
                print(index_e)
                if idx < self.len:
                    idx += 1
                else:
                    idx = 0

        return sample


def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)    
    return (data - min)/(max-min)

class ToTensor():
    '''Convert ndarrays in sample to Tensors.'''
    def __call__(self, sample):
        image, clinical, label = sample['mri'],sample['clinical'], sample['label']
        #image, label = sample['mri'], sample['label']
        #mri_t = torch.from_numpy(minmaxscaler(image))
        mri_t = torch.from_numpy(image)
        # mri_t = torch.from_numpy(image).double()
        clin_t = torch.from_numpy(clinical)
        label = torch.from_numpy(label).double()
        return {'mri': mri_t,
                'clin_t': clin_t,
                'label': label}

        # image, label = sample['mri'], sample['label']
        # #mri_t = torch.from_numpy(minmaxscaler(image))
        # mri_t = torch.from_numpy(image)
        
        # label = torch.from_numpy(label).double()
        # return {'mri': mri_t,
        #         'label': label}
