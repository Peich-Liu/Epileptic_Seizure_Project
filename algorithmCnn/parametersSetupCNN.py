''' file with all parameters'''
import numpy as np
import pickle
# from sklearn.ensemble import RUSBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import torch
from torch.utils.data import Dataset, DataLoader
import os
from VariousFunctionsLib import  *
from datetime import timedelta, datetime
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
#### STAND PARAMETERS
class StandardParamsCNN:
    #window size and step in which window is moved
    winLen= 4 #in seconds, window length on which to calculate features
    winStep= 1 #in seconds, step of moving window length

    #normalization of feature values or not
    featNorm = 'Norm' #'', 'Norm&Discr', 'Norm'

    #features extracted from data
    featSetNames = np.array( [])
    # featSetNames = np.array( ['MeanAmpl', 'LineLength', 'Frequency', 'ZeroCross'])
    featNames = np.array([])
    allFeatName = '-'.join(featSetNames)

    # #individual features within each set of features
    # indivFeatNames_MeanAmpl=['meanAmpl']
    # indivFeatNames_LL=['lineLenth']
    # indivFeatNames_Freq = ['p_dc_rel', 'p_mov_rel', 'p_delta_rel', 'p_theta_rel', 'p_alfa_rel', 'p_middle_rel', 'p_beta_rel', 'p_gamma_rel', 'p_dc', 'p_mov', 'p_delta', 'p_theta', 'p_alfa', 'p_middle', 'p_beta', 'p_gamma', 'p_tot']
    # indivFeatNames_SD = ['StandardDeviation']
    # indivFeatNames_DMe = ['DMe']
    # indivFeatNames_Skew = ['SKewnesss']
    # indivFeatNames_SO = ['SecondOrder']
    # indivFeatNames_KatzFD = ['KatzFD']
    # indivFeatNames_NW = ['MeanDeg','MeanBetw','MeanClose']


    #ZC features params
    ZC_thresh_type='rel' #'abs' or 'rel'
    ZC_thresh_arr_rel=[ 0.25, 0.50, 0.75, 1, 1.5]
    ZC_thresh_arr = [16, 32, 64, 128, 256]

#compile list of all features
StandardParamsCNN.allFeatNames = None

##############################################################################################
#### DEEP LEARNING PARAMETER
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
        self.standDir = standDir
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
        self.edfFiles.sort()
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

        # unbalanced data modify
        self.window_indices = []

        # print("self.edfFiles",self.edfFiles)
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
        
        # self.test_window_indices = label_0_indices + label_1_indices
        # # sampled_label_0_indices = random.sample(label_0_indices, len(label_1_indices)*3)
        # # # print("label_0_indices=",len(sampled_label_0_indices),"label_1_indices=",len(label_1_indices))
        # # self.balanced_window_indices = sampled_label_0_indices + label_1_indices
        # # random.shuffle(self.balanced_window_indices)
        # print("self.test_window_indices=",len(self.test_window_indices))


    def __len__(self):
        # return len(self.test_window_indices)
        return len(self.window_indices)

    
    def __getitem__(self, idx):
        # print("idx:", idx)
        # file_idx, within_file_idx, label = self.balanced_window_indices[idx]
        # self.current_data, self.sampleFreq, self.fileStartTime = self.load_file(file_idx)
        # file_idx, within_file_idx, label = self.test_window_indices[idx]
        file_idx, within_file_idx, label = self.window_indices[idx]
        # if self.edfFiles[file_idx] not in self.file_data:
        #     self.load_file(file_idx)
        #     print("load")
        
        start = within_file_idx * self.step_size
        end = start + self.window_size
        filepath = self.edfFiles[file_idx]
        rel_filepath = os.path.relpath(filepath, start=self.standDir)
        seizure_record = self.seizure_info_df[self.seizure_info_df['filepath'] == rel_filepath]

        # print("rel_filepath=",rel_filepath)
        # seizure_record = self.seizure_info_df[self.seizure_info_df['filepath'] == rel_filepath]
        window_start_time_seconds = within_file_idx * self.step_size / self.sampling_rate
        
        file_start_datetime = datetime.strptime(seizure_record['dateTime'].iloc[0], '%Y-%m-%d %H:%M:%S')
        window_start_datetime = file_start_datetime + timedelta(seconds=window_start_time_seconds)
        # print("window_start_datetime",window_start_datetime)
        
        additional_info = {
        'subject': seizure_record['subject'].iloc[0] if not seizure_record.empty else None,
        'session': seizure_record['session'].iloc[0] if not seizure_record.empty else None,
        'recording': seizure_record['recording'].iloc[0] if not seizure_record.empty else None,
        'dateTime': seizure_record['dateTime'].iloc[0] if not seizure_record.empty else None,
        'duration': seizure_record['duration'].iloc[0] if not seizure_record.empty else None,
        'event': seizure_record['event'].iloc[0] if not seizure_record.empty else None,
        'startTime': window_start_datetime,
        'endTime': end,
        'confidence': seizure_record['confidence'].iloc[0] if not seizure_record.empty else None,
        'channels': seizure_record['channels'].iloc[0] if not seizure_record.empty else None,
        'filepath': rel_filepath,
    }
        # print("seizure_record",seizure_record)
        # print("additional_info",additional_info)
        # if end > len(self.current_data):
        #     raise IndexError("Index out of range")
        window = self.current_data[start:end].to_numpy()
        window_tensor = torch.tensor(window, dtype=torch.float)
        window_tensor = window_tensor.transpose(0, 1)
        # print("self.seizure_info_df",self.seizure_info_df)
        # print("window_tensor=",window_tensor.shape,"file_idx=",file_idx)
        # add a error report
        # if(window_tensor.shape != torch.Size([18, 1024])):
        #     print("error=",window_tensor)
        return window_tensor, torch.tensor(label, dtype=torch.long), additional_info    
    
    
    def load_file(self, file_index):
        # print("edfFiles=",self.edfFiles,"file_index=",file_index)
        filepath = self.edfFiles[file_index]
        print("filepath",filepath)
        eegDataDF, samplFreq, fileStartTime = readEdfFile(filepath)
        # print("eegDataDF=",eegDataDF)
        return eegDataDF, samplFreq, fileStartTime