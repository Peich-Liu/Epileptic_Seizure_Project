''' file with all parameters'''
import numpy as np
import pickle
# from sklearn.ensemble import RUSBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import torch
from torch.utils.data import Dataset, DataLoader
import os
from VariousFunctionsLib import  *
from datetime import timedelta
import random

class GeneralParamsCNN:
    patients=[]  #on which subjects to train and test
    PersCV_MinTrainHours=5 #minimum number of hours we need to start training in personal model
    PersCV_CVStepInHours=1 #how often we retrain and on how much next data we test
    GenCV_numFolds=5

##############################################################################################
#### PREPROCESSING PARAMETERS
class DatasetPreprocessParamsCNN: # mostly based on CHB-MIT dataset
    samplFreq = 256  # sampling frequency of data
    #Channels structure
    #Unipolar channels
    # channelNamesToKeep_Unipolar = ('T3')
    channelNamesToKeep_Unipolar = ('Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6')
    refElectrode ='Cz' #default 'Cz' or 'Avrg', or any of channels listed above
    #Bipolar channels
    # channelNamesToKeep_Bipolar = ('T3-Cz')
    channelNamesToKeep_Bipolar = ('Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fz-Cz', 'Cz-Pz', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2')
    # channelNamesToKeep_Bipolar = ('Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'Fz-Cz', 'Cz-Pz',
    #                    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fp2-F8', 'F8-T8', 'T8-P8', 'P8-O2') # TODO for old features
    # refElectrode='bipolar-dBanana' #if bipolar then ref electrode is not needed so put 'bipolar-dBanana'
    channelNamesToKeep=channelNamesToKeep_Unipolar

    # raw EEG data normalization
    eegDataNormalization='' # '' for none, 'NormWithPercentile', or 'QuantileNormalization'

##############################################################################################
#### FEATURES PARAMETERS

class winParamsCNN:
    #window size and step in which window is moved
    winLen= 4 #in seconds, window length on which to calculate features
    winStep= 1 #in seconds, step of moving window length

    #normalization of feature values or not
    featNorm = 'Norm' #'', 'Norm&Discr', 'Norm'

##############################################################################################
#### PERFORMANCE METRICS PARAMETERS

class PerformanceParams:

    #LABEL SMOOTHING PARAMETERS
    smoothingWinLen=5 #in seconds  - window for performing label voting
    votingPercentage=0.5 # e.g. if 50% of 1 in last seizureStableLenToTest values that needs to be 1 to finally keep label 1
    bayesProbThresh = 1.5  # smoothing with cummulative probabilities, threshold from Valentins paper

    #EVENT BASED PERFORMANCE METRIC
    toleranceStart = 30 #in seconds - tolerance before seizure not to count as FP
    toleranceEnd =60  #in seconds - tolerance after seizure not to count as FP
    minOveralp =0 #minimum relative overlap between predicted and true seizure to count as match
    maxEventDuration=300 #max length of sizure, automatically split events longer than a given duration
    minDurationBetweenEvents= 90 #mergint too close seizures, automatically merge events that are separated by less than the given duration

    predictionFreq=1/winParamsCNN.winStep #ultimate frequency of predictions



#SAVING SETUP once again to update if new info
with open('../PARAMETERS.pickle', 'wb') as f:
    pickle.dump([GeneralParamsCNN, DatasetPreprocessParamsCNN, winParamsCNN, PerformanceParams], f)


##############################################################################################
#### DEEP LEARNING PARAMETER
# class EEGDataset(Dataset):
#     def __init__(self, folderIn):
#         self.file_paths = np.sort(glob.glob(os.path.join(folderIn, '**/*.edf'), recursive=True))

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, index):
#         file_path = self.file_paths[index]
#         eegDataDF, samplFreq, fileStartTime = readEdfFile(file_path)  # Load data
#         print("eegDataDF=",eegDataDF)

#         eegDataArray = eegDataDF.to_numpy()

#         sample = torch.from_numpy(eegDataArray).float()

# The `EEGDataset` class is a custom dataset class for loading and processing EEG data from EDF files,
# with support for balancing the data based on seizure labels.
class EEGDataset(Dataset):
    def __init__(self, standDir, folderIn, seizure_info_df, samplFreq, winLen, winStep):
        self.folderIn = folderIn
        self.file_data = {} 
        self.sampling_rate = samplFreq
        self.window_size = winLen * samplFreq
        self.step_size =  winStep * samplFreq 
        # print("window_size=",self.window_size,"step_size=",self.step_size)
        self.edfFiles = []
        # print("folderIn=",folderIn)
        for folder in folderIn:
            # print(folder)
            edfFilesInFolder = glob.glob(os.path.join(folder, '**/*.edf'), recursive=True)
            self.edfFiles.extend(edfFilesInFolder)
        # self.edfFiles = np.sort(glob.glob(os.path.join(folderIn, '**/*.edf'), recursive=True))
        self.current_file_index = 0
        self.current_data, self.sampleFreq, self.fileStartTime = self.load_file(self.current_file_index)
        self.current_file_length = self.current_data.shape[0]
        self.index_within_file = 0
        # self.seizure_info_df = pd.read_csv(labelFile)
        self.seizure_info_df = seizure_info_df
        self.file_to_seizure = {}
        for _, row in self.seizure_info_df.iterrows():
            relative_path = row['filepath']
            absolute_path = os.path.abspath(os.path.join(standDir, relative_path))
            if os.path.exists(absolute_path):
                self.file_to_seizure[absolute_path] = (row['startTime'], row['endTime'], row['event'])
        self.filepaths = list(self.file_to_seizure.keys())
        # print("self.filepaths=",self.filepaths)
        # self.seizure_times = self.readSeizureTimes(labelFile)
        # unbalanced data modify
        self.window_indices = []
        # self.calculate_window_indices()
        # self.balance_data()
        
        for file_idx, file_path in enumerate(self.edfFiles):
            num_windows = (self.current_file_length - self.window_size) // self.step_size + 1
            for within_file_idx in range(num_windows):
                start = within_file_idx * self.step_size
                end = start + self.window_size
                if file_path in self.file_to_seizure:
                    seizureStart, seizureEnd, seizureType = self.file_to_seizure[file_path]
                    if seizureStart*self.sampleFreq < end and seizureEnd*self.sampleFreq > start and 'sz' in seizureType:
                        label = 1
                    else:
                        label = 0
                    
                else:
                    label = 0
                
                if end > self.current_file_length:
                    print("skip")
                    continue
                self.window_indices.append((file_idx, within_file_idx, label))

        label_0_indices = [idx for idx in self.window_indices if idx[2] == 0]
        label_1_indices = [idx for idx in self.window_indices if idx[2] == 1]

        sampled_label_0_indices = random.sample(label_0_indices, len(label_1_indices)*3)
        # print("label_0_indices=",len(sampled_label_0_indices),"label_1_indices=",len(label_1_indices))
        self.balanced_window_indices = sampled_label_0_indices + label_1_indices
        random.shuffle(self.balanced_window_indices)
        # print("self.balanced_window_indices=",self.balanced_window_indices)


    def __len__(self):
        return len(self.balanced_window_indices)
    
    def __getitem__(self, idx):
        # print("idx:", idx)
        # file_idx, within_file_idx, label = self.balanced_window_indices[idx]
        # self.current_data, self.sampleFreq, self.fileStartTime = self.load_file(file_idx)
        file_idx, within_file_idx, label = self.balanced_window_indices[idx]
        # if self.edfFiles[file_idx] not in self.file_data:
        #     self.load_file(file_idx)
        #     print("load")
        start = within_file_idx * self.step_size
        end = start + self.window_size
        # if end > len(self.current_data):
        #     raise IndexError("Index out of range")
        window = self.current_data[start:end].to_numpy()
        window_tensor = torch.tensor(window, dtype=torch.float)
        window_tensor = window_tensor.transpose(0, 1)
        # print("window_tensor=",window_tensor.shape,"file_idx=",file_idx)
        # add a error report
        # if(window_tensor.shape != torch.Size([18, 1024])):
        #     print("error=",window_tensor)
        return window_tensor, torch.tensor(label, dtype=torch.long)      
    
    # def load_file(self, file_index):
    #     filepath = self.edfFiles[file_index]
    #     if filepath in self.file_data:
    #         return self.file_data[filepath]  # 如果已加载，则返回存储的数据

    #     eegDataDF, samplFreq, fileStartTime = readEdfFile(filepath)
    #     self.file_data[filepath] = (eegDataDF, samplFreq, fileStartTime)  # 存储数据以供后续使用
    #     return eegDataDF, samplFreq, fileStartTime
    
    
    def load_file(self, file_index):
        # print("edfFiles=",self.edfFiles,"file_index=",file_index)
        filepath = self.edfFiles[file_index]
        eegDataDF, samplFreq, fileStartTime = readEdfFile(filepath)
        # print("eegDataDF=",eegDataDF)
        return eegDataDF, samplFreq, fileStartTime
    


class EEGDatasetTest(Dataset):
    def __init__(self, standDir, folderIn, seizure_info_df, samplFreq, winLen, winStep):
        self.folderIn = folderIn
        self.file_data = {} 
        self.sampling_rate = samplFreq
        self.window_size = winLen * samplFreq
        self.step_size =  winStep * samplFreq 
        # print("window_size=",self.window_size,"step_size=",self.step_size)
        self.edfFiles = []
        # print("folderIn=",folderIn)
        for folder in folderIn:
            # print(folder)
            edfFilesInFolder = glob.glob(os.path.join(folder, '**/*.edf'), recursive=True)
            self.edfFiles.extend(edfFilesInFolder)
        # self.edfFiles = np.sort(glob.glob(os.path.join(folderIn, '**/*.edf'), recursive=True))
        self.current_file_index = 0
        self.current_data, self.sampleFreq, self.fileStartTime = self.load_file(self.current_file_index)
        self.current_file_length = self.current_data.shape[0]
        self.index_within_file = 0
        # self.seizure_info_df = pd.read_csv(labelFile)
        self.seizure_info_df = seizure_info_df
        self.file_to_seizure = {}
        for _, row in self.seizure_info_df.iterrows():
            relative_path = row['filepath']
            absolute_path = os.path.abspath(os.path.join(standDir, relative_path))
            if os.path.exists(absolute_path):
                self.file_to_seizure[absolute_path] = (row['startTime'], row['endTime'], row['event'])
        self.filepaths = list(self.file_to_seizure.keys())
        # print("self.filepaths=",self.filepaths)
        # self.seizure_times = self.readSeizureTimes(labelFile)
        # unbalanced data modify
        self.window_indices = []
        # self.calculate_window_indices()
        # self.balance_data()
        
        for file_idx, file_path in enumerate(self.edfFiles):
            num_windows = (self.current_file_length - self.window_size) // self.step_size + 1
            for within_file_idx in range(num_windows):
                start = within_file_idx * self.step_size
                end = start + self.window_size
                if file_path in self.file_to_seizure:
                    seizureStart, seizureEnd, seizureType = self.file_to_seizure[file_path]
                    if seizureStart*self.sampleFreq < end and seizureEnd*self.sampleFreq > start and 'sz' in seizureType:
                        label = 1
                    else:
                        label = 0
                    
                else:
                    label = 0
                
                if end > self.current_file_length:
                    print("skip")
                    continue
                self.window_indices.append((file_idx, within_file_idx, label))

        label_0_indices = [idx for idx in self.window_indices if idx[2] == 0]
        label_1_indices = [idx for idx in self.window_indices if idx[2] == 1]
        
        self.test_window_indices = label_0_indices + label_1_indices
        # sampled_label_0_indices = random.sample(label_0_indices, len(label_1_indices)*3)
        # # print("label_0_indices=",len(sampled_label_0_indices),"label_1_indices=",len(label_1_indices))
        # self.balanced_window_indices = sampled_label_0_indices + label_1_indices
        # random.shuffle(self.balanced_window_indices)
        print("self.test_window_indices=",len(self.test_window_indices))


    def __len__(self):
        return len(self.test_window_indices)
    
    def __getitem__(self, idx):
        # print("idx:", idx)
        # file_idx, within_file_idx, label = self.balanced_window_indices[idx]
        # self.current_data, self.sampleFreq, self.fileStartTime = self.load_file(file_idx)
        file_idx, within_file_idx, label = self.test_window_indices[idx]
        # if self.edfFiles[file_idx] not in self.file_data:
        #     self.load_file(file_idx)
        #     print("load")
        start = within_file_idx * self.step_size
        end = start + self.window_size
        # if end > len(self.current_data):
        #     raise IndexError("Index out of range")
        window = self.current_data[start:end].to_numpy()
        window_tensor = torch.tensor(window, dtype=torch.float)
        window_tensor = window_tensor.transpose(0, 1)
        # print("window_tensor=",window_tensor.shape,"file_idx=",file_idx)
        # add a error report
        # if(window_tensor.shape != torch.Size([18, 1024])):
        #     print("error=",window_tensor)
        return window_tensor, torch.tensor(label, dtype=torch.long)      
    
    # def load_file(self, file_index):
    #     filepath = self.edfFiles[file_index]
    #     if filepath in self.file_data:
    #         return self.file_data[filepath]  # 如果已加载，则返回存储的数据

    #     eegDataDF, samplFreq, fileStartTime = readEdfFile(filepath)
    #     self.file_data[filepath] = (eegDataDF, samplFreq, fileStartTime)  # 存储数据以供后续使用
    #     return eegDataDF, samplFreq, fileStartTime
    
    
    def load_file(self, file_index):
        # print("edfFiles=",self.edfFiles,"file_index=",file_index)
        filepath = self.edfFiles[file_index]
        eegDataDF, samplFreq, fileStartTime = readEdfFile(filepath)
        # print("eegDataDF=",eegDataDF)
        return eegDataDF, samplFreq, fileStartTime
    
    
    
    
    
    
    
        # file_idx, within_file_idx, label = self.balanced_window_indices[idx]
        # self.current_file_index = file_idx
        # self.index_within_file = within_file_idx * self.window_size
        
        # # file range
        # if self.current_file_index >= len(self.edfFiles):
        #     raise IndexError("Index out of range")
        
        # filepath = self.edfFiles[self.current_file_index]

        # if filepath in self.file_to_seizure:
        #     seizureStart, seizureEnd, seizureType = self.file_to_seizure[filepath]
        # else:
        #     seizureStart, seizureEnd, seizureType = None, None, None
        
        # start = self.index_within_file
        # end = start + self.window_size

        # window = self.current_data[start:end].to_numpy()
        # self.index_within_file += self.step_size
        # window_tensor = torch.tensor(window, dtype=torch.float)
        # window_tensor = window_tensor.transpose(0, 1) 
        # # print("window_tensor",window_tensor.shape)
        # return window_tensor, torch.tensor(label, dtype=torch.long)
        # # return torch.tensor(window, dtype=torch.float), torch.tensor(label, dtype=torch.long)
        

    
    # def generate_label(self, window_start, window_end, seizure_start, seizure_end, szType):
    #     # print("window_start=",window_start,"window_end",window_end,"seizure_start",seizure_start,"seizure_end",seizure_end)
    #     # if 'sz' not in szType:
    #     #     return 0
    #     # if seizure_start is None or seizure_end is None:
    #     #     return 0
    #     # label = 1 if (seizure_start < window_end and seizure_end > window_start and 'sz' in szType) else 0
    #     if (seizure_start < window_end and seizure_end > window_start and 'sz' in szType):
    #         label = 1
    #         # print("szType=",szType,"seizure_start=",seizure_start)
    #     else:
    #         label = 0
    #     return label
    
    # def calculate_window_indices(self):
    #     self.window_indices = []
        
    #     for file_idx, file_path in enumerate(self.edfFiles):
    #         self.current_data, self.sampleFreq, self.fileStartTime = self.load_file(file_idx)
    #         self.current_file_length = self.current_data.shape[0]

    #         num_windows = (self.current_file_length - self.window_size) // self.step_size + 1
    #         print("num_windows=",num_windows)
    #         for within_file_idx in range(num_windows):
    #             start = within_file_idx * self.step_size
    #             end = start + self.window_size

    #             if file_path in self.file_to_seizure:
    #                 seizureStart, seizureEnd, seizureType = self.file_to_seizure[file_path]
    #                 label = 1 if seizureStart < end and seizureEnd > start and 'sz' in seizureType else 0
    #             else:
    #                 label = 0

    #             self.window_indices.append((file_idx, within_file_idx, label))
    #             # print("self.window_indices=",self.window_indices)

    # def balance_data(self):
    #     label_0_indices = [idx for idx in self.window_indices if idx[2] == 0]
    #     label_1_indices = [idx for idx in self.window_indices if idx[2] == 1]

    #     num_samples_to_select = min(len(label_0_indices), len(label_1_indices) * 3)
    #     sampled_label_0_indices = random.sample(label_0_indices, num_samples_to_select)

    #     self.balanced_window_indices = sampled_label_0_indices + label_1_indices
    #     print("sampled_label_0_indices=",sampled_label_0_indices,"label_1_indices",label_1_indices)
    #     random.shuffle(self.balanced_window_indices)