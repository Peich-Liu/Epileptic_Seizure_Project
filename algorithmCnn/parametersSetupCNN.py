''' file with all parameters'''
import sys
import os
sys.path.append(r'../../Epileptic_Seizure_Project')
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
    dataset = 'CHBMIT'
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
    def updateDatasetPreprocessParams(params):
        DatasetPreprocessParamsCNN.channelNamesToKeep_Unipolar = params.get('Unipolar',DatasetPreprocessParamsCNN.channelNamesToKeep_Unipolar)
        DatasetPreprocessParamsCNN.channelNamesToKeep_Bipolar = params.get('Bipolar',DatasetPreprocessParamsCNN.channelNamesToKeep_Bipolar)
        DatasetPreprocessParamsCNN.samplFreq = params.get('samplFreq',DatasetPreprocessParamsCNN.samplFreq)
        DatasetPreprocessParamsCNN.eegDataNormalization = params.get('eegDataNormalization',DatasetPreprocessParamsCNN.eegDataNormalization)
        DatasetPreprocessParamsCNN.dataset = params.get('dataset',DatasetPreprocessParamsCNN.dataset)
        print("uni",DatasetPreprocessParamsCNN.channelNamesToKeep_Unipolar)

##############################################################################################
#### FEATURES PARAMETERS

class winParamsCNN:
    #window size and step in which window is moved
    winLen= 4 #in seconds, window length on which to calculate features
    winStep= 1 #in seconds, step of moving window length
    winStepTest = winLen
    #normalization of feature values or not
    featNorm = 'Norm' #'', 'Norm&Discr', 'Norm'
    def updateWinParamsCNN(params):
        winParamsCNN.winLen = params.get('winLen',winParamsCNN.winLen)
        winParamsCNN.winStep = params.get('winStep',winParamsCNN.winStep)
        winParamsCNN.featNorm = params.get('featNorm',winParamsCNN.featNorm)
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

    predictionFreq=1/winParamsCNN.winStepTest #ultimate frequency of predictions



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
    
    #ZC features params
    ZC_thresh_type='rel' #'abs' or 'rel'
    ZC_thresh_arr_rel=[ 0.25, 0.50, 0.75, 1, 1.5]
    ZC_thresh_arr = [16, 32, 64, 128, 256]

#compile list of all features
StandardParamsCNN.allFeatNames = None

##############################################################################################
#### DEEP LEARNING PARAMETER
class EEGDataset(Dataset):
    def __init__(self, standDir, folderIn, seizure_info_df, samplFreq, winLen, winStep, Norm):        
        self.dataNorm = Norm
        self.folderIn = folderIn
        self.standDir = standDir
        self.file_data = {} 
        self.sampling_rate = samplFreq
        self.window_size = winLen * samplFreq
        self.step_size =  winStep * samplFreq
        self.file_cache = {}

        self.edfFiles = []

        for folder in folderIn:
            print("folder",folder)
            edfFilesInFolder = glob.glob(os.path.join(folder, '**/*.edf'), recursive=True)
            self.edfFiles.extend(edfFilesInFolder)
        self.edfFiles.sort()

        self.seizure_info_df = seizure_info_df
        print("seizure_info_df",seizure_info_df)
        self.file_to_seizure = {}
        for _, row in self.seizure_info_df.iterrows():
            relative_path = row['filepath']
            absolute_path = os.path.abspath(os.path.join(standDir, relative_path))
            if os.path.exists(absolute_path):
                self.file_to_seizure[absolute_path] = (row['startTime'], row['endTime'], row['event'])
        print("file_to_seizure",self.file_to_seizure)
        
        self.filepaths = list(self.file_to_seizure.keys())

        # unbalanced data modify
        self.window_indices = []
        print("edfFiles",self.edfFiles)
        for file_idx, file_path in enumerate(self.edfFiles):
            self.current_file_index = file_idx
            self.current_data, self.sampleFreq, self.fileStartTime = self.load_file(self.current_file_index)
            self.current_file_length = self.current_data.shape[0]
            self.index_within_file = 0
            # print("file_path",file_path,"current_data",self.current_data,"current_data.shape",self.current_data.shape[0])
            num_windows = (self.current_file_length - self.window_size) // self.step_size + 1
            print("num_windows",num_windows)
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
        print("trainwindow_indices",len(self.window_indices))
        label_0_indices = [idx for idx in self.window_indices if idx[2] == 0]
        label_1_indices = [idx for idx in self.window_indices if idx[2] == 1]

        # # if len(label_0_indices) > 100000:
        # #     sampled_label_0_indices = random.sample(label_0_indices, 100000)
        # sampled_label_0_indices = random.sample(label_0_indices, len(label_1_indices)*3)
        # print("label_0_indices=",len(sampled_label_0_indices),"label_1_indices=",len(label_1_indices))
        # self.balanced_window_indices = sampled_label_0_indices+ label_1_indices
        # # self.balanced_window_indices = label_0_indices + label_1_indices
        # random.shuffle(self.balanced_window_indices)
        # print("self.balanced_window_indices=",len(self.balanced_window_indices))


    def __len__(self):
        # return len(self.balanced_window_indices)
        return len(self.window_indices)

    
    
    def __getitem__(self, idx):
        # print("idx",idx)
        file_idx, within_file_idx, label = self.window_indices[idx]
        # print("file_idx",file_idx)
        start = within_file_idx * self.step_size
        end = start + self.window_size
        filepath = self.edfFiles[file_idx]
        # print("getitem",filepath)
        current_data, sampleFreq, _ = self.load_file(file_idx)
        rel_filepath = os.path.relpath(filepath, start=self.standDir)
        seizure_record = self.seizure_info_df[self.seizure_info_df['filepath'] == rel_filepath]
        
        window_start_time_seconds = within_file_idx * self.step_size / self.sampling_rate
        try:
            file_start_datetime = datetime.strptime(seizure_record['dateTime'].iloc[0], '%Y-%m-%d %H:%M:%S')
        except:
            file_start_datetime = datetime.strptime(seizure_record['dateTime'].iloc[0], '%Y-%m-%d')
            
        window_start_datetime = file_start_datetime + timedelta(seconds=window_start_time_seconds)
        # print("window_start_datetime",window_start_datetime)

        window = current_data[start:end].to_numpy()
        # print("window",window.shape)
        window_tensor = torch.tensor(window, dtype=torch.float)
        window_tensor = window_tensor.transpose(0, 1)
        # print("additional_info",additional_info)
        # print("start",start)
        return window_tensor, torch.tensor(label, dtype=torch.long)
    
    def load_file(self, file_index):
        filepath = self.edfFiles[file_index]

        if filepath not in self.file_cache:

            eegDataDF, samplFreq, fileStartTime = readEdfFile(filepath)
            if self.dataNorm == 'NormWithPercentile':
                eegDataDF = normalizeWithPercentile(eegDataDF, 0.99)
            elif self.dataNorm == 'QuantileNormalization':
                eegDataDF = QuantileNormalization(eegDataDF)
            elif self.dataNorm == 'Z-Score':
                eegDataDF = z_score_normalize(eegDataDF)

            self.file_cache[filepath] = (eegDataDF, samplFreq, fileStartTime)
        return self.file_cache[filepath]
        # print("filepath",filepath)
        # eegDataDF, samplFreq, fileStartTime = readEdfFile(filepath)
    def clear_cache(self):
        self.file_cache.clear()    
    # def __getitem__(self, idx):
    #     file_idx, within_file_idx, label = self.balanced_window_indices[idx]
    #     start = within_file_idx * self.step_size
    #     end = start + self.window_size

    #     window = self.current_data[start:end].to_numpy()
    #     window_tensor = torch.tensor(window, dtype=torch.float)
    #     window_tensor = window_tensor.transpose(0, 1)

    #     return window_tensor, torch.tensor(label, dtype=torch.long)      
    
    
    # def load_file(self, file_index):
    #     filepath = self.edfFiles[file_index]
    #     eegDataDF, samplFreq, fileStartTime = readEdfFile(filepath)
    #     if (self.dataNorm=='NormWithPercentile'):
    #         eegDataDF= normalizeWithPercentile(eegDataDF, 0.99)
    #         # plotDensityPlotOfData(eegDataDF, folderOut + '/EEGDensity_'+dataNorm+'.png')
    #     elif (self.dataNorm=='QuantileNormalization'):
    #         eegDataDF= QuantileNormalization(eegDataDF)
    #         # plotDensityPlotOfData(eegDataDF, folderOut + '/EEGDensity_'+dataNorm+'.png')
    #     elif(self.dataNorm=='Z-Score'):
    #         eegDataDF= z_score_normalize(eegDataDF)
    #     return eegDataDF, samplFreq, fileStartTime
    
























class EEGDatasetTest(Dataset):
    def __init__(self, standDir, folderIn, seizure_info_df, samplFreq, winLen, winStep,Norm):
        self.dataNorm = Norm
        self.folderIn = folderIn
        self.standDir = standDir
        self.file_data = {} 
        self.sampling_rate = samplFreq
        self.window_size = winLen * samplFreq
        self.step_size =  winStep * samplFreq
        self.file_cache = {}


        self.edfFiles = []

        for folder in folderIn:
            print("folder",folder)
            edfFilesInFolder = glob.glob(os.path.join(folder, '**/*.edf'), recursive=True)
            self.edfFiles.extend(edfFilesInFolder)
        self.edfFiles.sort()

        self.seizure_info_df = seizure_info_df
        print("seizure_info_df",seizure_info_df)
        self.file_to_seizure = {}
        for _, row in self.seizure_info_df.iterrows():
            relative_path = row['filepath']
            absolute_path = os.path.abspath(os.path.join(standDir, relative_path))
            if os.path.exists(absolute_path):
                self.file_to_seizure[absolute_path] = (row['startTime'], row['endTime'], row['event'])
        print("file_to_seizure",self.file_to_seizure)
        
        self.filepaths = list(self.file_to_seizure.keys())

        # unbalanced data modify
        self.window_indices = []
        print("edfFiles",self.edfFiles)
        for file_idx, file_path in enumerate(self.edfFiles):
            self.current_file_index = file_idx
            self.current_data, self.sampleFreq, self.fileStartTime = self.load_file(self.current_file_index)
            self.current_file_length = self.current_data.shape[0]
            self.index_within_file = 0
            # print("file_path",file_path,"current_data",self.current_data,"current_data.shape",self.current_data.shape[0])
            num_windows = (self.current_file_length - self.window_size) // self.step_size + 1
            print("num_windows",num_windows)
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
                # print("file_idx",file_idx,"within_file_idx",within_file_idx,"label",label)
        print("self.window_indices",len(self.window_indices))
        print("window1",self.window_indices[1])


    def __len__(self):
        # return len(self.test_window_indices)
        return len(self.window_indices)

    
    def __getitem__(self, idx):
        # print("idx",idx)
        file_idx, within_file_idx, label = self.window_indices[idx]
        # print("file_idx",file_idx)
        start = within_file_idx * self.step_size
        end = start + self.window_size
        filepath = self.edfFiles[file_idx]
        # print("getitem",filepath)
        current_data, sampleFreq, _ = self.load_file(file_idx)
        rel_filepath = os.path.relpath(filepath, start=self.standDir)
        # print("rel_filepath",rel_filepath)
        seizure_record = self.seizure_info_df[self.seizure_info_df['filepath'] == rel_filepath]
        
        window_start_time_seconds = within_file_idx * self.step_size / self.sampling_rate
        try:
            file_start_datetime = datetime.strptime(seizure_record['dateTime'].iloc[0], '%Y-%m-%d %H:%M:%S')
        except:
            file_start_datetime = datetime.strptime(seizure_record['dateTime'].iloc[0], '%Y-%m-%d')
            
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

        window = current_data[start:end].to_numpy()
        # print("window",window.shape)
        window_tensor = torch.tensor(window, dtype=torch.float)
        window_tensor = window_tensor.transpose(0, 1)
        # print("additional_info",additional_info)
        # print("start",start)
        return window_tensor, torch.tensor(label, dtype=torch.long), additional_info
    
    def load_file(self, file_index):
        filepath = self.edfFiles[file_index]
        if filepath not in self.file_cache:
            eegDataDF, samplFreq, fileStartTime = readEdfFile(filepath)
            if self.dataNorm == 'NormWithPercentile':
                eegDataDF = normalizeWithPercentile(eegDataDF, 0.99)
            elif self.dataNorm == 'QuantileNormalization':
                eegDataDF = QuantileNormalization(eegDataDF)
            elif self.dataNorm == 'Z-Score':
                eegDataDF = z_score_normalize(eegDataDF)

            self.file_cache[filepath] = (eegDataDF, samplFreq, fileStartTime)
        return self.file_cache[filepath]
        # print("filepath",filepath)
        # eegDataDF, samplFreq, fileStartTime = readEdfFile(filepath)
    def clear_cache(self):
        self.file_cache.clear()
        
        
        # if (self.dataNorm=='NormWithPercentile'):
        #     eegDataDF= normalizeWithPercentile(eegDataDF, 0.99)
        #     # plotDensityPlotOfData(eegDataDF, folderOut + '/EEGDensity_'+dataNorm+'.png')
        # elif (self.dataNorm=='QuantileNormalization'):
        #     eegDataDF= QuantileNormalization(eegDataDF)
        #     # plotDensityPlotOfData(eegDataDF, folderOut + '/EEGDensity_'+dataNorm+'.png')
        # elif(self.dataNorm=='Z-Score'):
        #     eegDataDF= z_score_normalize(eegDataDF)
        # return eegDataDF, samplFreq, fileStartTime
