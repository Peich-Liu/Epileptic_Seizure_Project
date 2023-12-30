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
from multiprocessing import Pool, cpu_count
from tsaug import Reverse, Drift
class GeneralParamsTF:
    patients=[]  #on which subjects to train and test
    PersCV_MinTrainHours=5 #minimum number of hours we need to start training in personal model
    PersCV_CVStepInHours=1 #how often we retrain and on how much next data we test
    GenCV_numFolds=5

##############################################################################################
#### PREPROCESSING PARAMETERS
class DatasetPreprocessParamsTF: # mostly based on CHB-MIT dataset
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
    # eegDataNormalization='' # '' for none, 'NormWithPercentile', or 'QuantileNormalization' 'Z-Score'
    eegDataNormalization='' # '' for none, 'NormWithPercentile', or 'QuantileNormalization' 'Z-Score'

##############################################################################################
#### FEATURES PARAMETERS

class winParamsTF:
    #window size and step in which window is moved
    winLen= 4 #in seconds, window length on which to calculate features
    winStep= 1 #in seconds, step of moving window length

    #normalization of feature values or not
    featNorm = 'Norm' #'', 'Norm&Discr', 'Norm'

##############################################################################################
#### PERFORMANCE METRICS PARAMETERS

class PerformanceParamsTF:

    #LABEL SMOOTHING PARAMETERS
    smoothingWinLen=5 #in seconds  - window for performing label voting
    votingPercentage=0.5 # e.g. if 50% of 1 in last seizureStableLenToTest values that needs to be 1 to finally keep label 1
    bayesProbThresh = 1.5  # smoothing with cummulative probabilities, threshold from Valentins paper

    #EVENT BASED PERFORMANCE METRIC
    toleranceStart = 50 #in seconds - tolerance before seizure not to count as FP
    toleranceEnd =60  #in seconds - tolerance after seizure not to count as FP
    minOveralp =0 #minimum relative overlap between predicted and true seizure to count as match
    maxEventDuration=300 #max length of sizure, automatically split events longer than a given duration
    minDurationBetweenEvents= 90 #mergint too close seizures, automatically merge events that are separated by less than the given duration

    predictionFreq=1/winParamsTF.winStep #ultimate frequency of predictions



#SAVING SETUP once again to update if new info
with open('../PARAMETERS.pickle', 'wb') as f:
    pickle.dump([GeneralParamsTF, DatasetPreprocessParamsTF, winParamsTF, PerformanceParamsTF], f)


##############################################################################################
#### DEEP LEARNING PARAMETER
class EEGDataset(Dataset):
#Train dataset
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

        self.balanced_window_indices = label_0_indices
        if len(self.balanced_window_indices) > 30000:
            self.balanced_window_indices = random.sample(self.balanced_window_indices, 30000)

        # sampled_label_0_indices = random.sample(label_0_indices, len(label_1_indices)*3)
        # print("label_0_indices=",len(sampled_label_0_indices),"label_1_indices=",len(label_1_indices))
        # # self.balanced_window_indices = sampled_label_0_indices + label_1_indices
        # self.balanced_window_indices = sampled_label_0_indices
        # random.shuffle(self.balanced_window_indices)
        print("self.balanced_window_indices=",len(self.balanced_window_indices))
    def __len__(self):
        return len(self.balanced_window_indices)
    
    def __getitem__(self, idx):
        file_idx, within_file_idx, label = self.balanced_window_indices[idx]
        start = within_file_idx * self.step_size
        end = start + self.window_size
        window = self.current_data[start:end].to_numpy()
        window_tensor = torch.tensor(window, dtype=torch.float)

        masking_ratio = 0.15  
        mean_mask_length = 3 
        mask = noise_mask(window, masking_ratio, lm=mean_mask_length, mode='separate', distribution='geometric')

        IDs = within_file_idx
        # IDs = list(IDs)
        # IDs = [within_file_idx]  
        return window_tensor, torch.from_numpy(mask), IDs

    
    def load_file(self, file_index):
        # print("edfFiles=",self.edfFiles,"file_index=",file_index)
        filepath = self.edfFiles[file_index]
        eegDataDF, samplFreq, fileStartTime = readEdfFile(filepath)
        return eegDataDF, samplFreq, fileStartTime
# ###################################   
# #test data class 
class EEGDataForDL(Dataset):
    def __init__(self, standDir, folderIn, seizure_info_df, samplFreq, winLen, winStep):
        self.folderIn = folderIn
        self.file_data = {} 
        self.sampling_rate = samplFreq
        self.window_size = winLen * samplFreq
        self.step_size =  winStep * samplFreq 

        self.edfFiles = []
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
        self.window_indices = []

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
        self.test_window_indices = self.window_indices
        self.label_0_indices = [idx for idx in self.window_indices if idx[2] == 0]
        self.label_1_indices = [idx for idx in self.window_indices if idx[2] == 1]
    
class EEGDatasetTest(EEGDataForDL):
    def __init__(self):
        self.balanced_window_indices = self.label_0_indices
        if len(self.balanced_window_indices) > 30000:
            self.balanced_window_indices = random.sample(self.balanced_window_indices, 30000)

    def __len__(self):
        return len(self.test_window_indices)
    
    def __getitem__(self, idx):
        file_idx, within_file_idx, label = self.test_window_indices[idx]
        start = within_file_idx * self.step_size
        end = start + self.window_size
        window = self.current_data[start:end].to_numpy()
        window_tensor = torch.tensor(window, dtype=torch.float)

        masking_ratio = 0.15  
        mean_mask_length = 3 
        mask = noise_mask(window, masking_ratio, lm=mean_mask_length, mode='separate', distribution='geometric')

        IDs = within_file_idx
        return window_tensor, torch.tensor(label, dtype=torch.long), IDs

    
    def load_file(self, file_index):
        # print("edfFiles=",self.edfFiles,"file_index=",file_index)
        filepath = self.edfFiles[file_index]
        eegDataDF, samplFreq, fileStartTime = readEdfFile(filepath)
        return eegDataDF, samplFreq, fileStartTime


def noise_mask(X, masking_ratio, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
            should be masked concurrently ('concurrent')
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)

    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])

    return mask
def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked

    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
def compensate_masking(X, mask):
    """
    Compensate feature vectors after masking values, in a way that the matrix product W @ X would not be affected on average.
    If p is the proportion of unmasked (active) elements, X' = X / p = X * feat_dim/num_active
    Args:
        X: (batch_size, seq_length, feat_dim) torch tensor
        mask: (batch_size, seq_length, feat_dim) torch tensor: 0s means mask and predict, 1s: unaffected (active) input
    Returns:
        (batch_size, seq_length, feat_dim) compensated features
    """

    # number of unmasked elements of feature vector for each time step
    num_active = torch.sum(mask, dim=-1).unsqueeze(-1)  # (batch_size, seq_length, 1)
    # to avoid division by 0, set the minimum to 1
    num_active = torch.max(num_active, torch.ones(num_active.shape, dtype=torch.int16))  # (batch_size, seq_length, 1)
    return X.shape[-1] * X / num_active
def collate_unsuperv(data, max_len=None, mask_compensation=False, task=None, oversample=False):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, mask).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - mask: boolean torch tensor of shape (seq_length, feat_dim); variable seq_length.
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 ignore (padding)
    """

    batch_size = len(data)
    features, masks, IDs = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    target_masks = torch.zeros_like(X,
                                    dtype=torch.bool)  # (batch_size, padded_length, feat_dim) masks related to objective
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        target_masks[i, :end, :] = masks[i][:end, :]

    targets = X.clone()
    X = X * target_masks  # mask input
    if mask_compensation:
        X = compensate_masking(X, target_masks)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep
    target_masks = ~target_masks  # inverse logic: 0 now means ignore, 1 means predict
    # print("x",X.shape,"target", targets.shape,"target_masks", target_masks.shape,"padding_masks", padding_masks.shape,"IDs", len(IDs))
    return X, targets, target_masks, padding_masks, IDs

class EEGDatasetTestTF(Dataset):
    def __init__(self, standDir, folderIn, seizure_info_df, samplFreq, winLen, winStep):
        self.folderIn = folderIn
        self.standDir = standDir
        self.file_data = {} 
        self.sampling_rate = samplFreq
        self.window_size = winLen * samplFreq
        self.step_size =  winStep * samplFreq 
        self.file_cache = {}

        self.edfFiles = []

        for folder in folderIn:
            # print(folder)
            edfFilesInFolder = glob.glob(os.path.join(folder, '**/*.edf'), recursive=True)
            self.edfFiles.extend(edfFilesInFolder)
        self.edfFiles.sort()

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
            self.current_file_index = file_idx
            self.current_data, self.sampleFreq, self.fileStartTime = self.load_file(self.current_file_index)
            self.current_file_length = self.current_data.shape[0]
            self.index_within_file = 0
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


    def __len__(self):
        # return len(self.test_window_indices)
        return len(self.window_indices)

    
    def __getitem__(self, idx):
        # print("idx",idx)
        file_idx, within_file_idx, label = self.window_indices[idx]
        # if self.edfFiles[file_idx] not in self.file_data:
        #     self.load_file(file_idx)
        #     print("load")
        
        # start = within_file_idx * self.step_size
        # end = start + self.window_size
        # filepath = self.edfFiles[file_idx]
        # rel_filepath = os.path.relpath(filepath, start=self.standDir)
        start = within_file_idx * self.step_size
        end = start + self.window_size
        filepath = self.edfFiles[file_idx]
        # print("getitem",filepath)
        current_data, sampleFreq, _ = self.load_file(file_idx)
        rel_filepath = os.path.relpath(filepath, start=self.standDir)
        # print("rel_filepath",rel_filepath)
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
        window = current_data[start:end].to_numpy()
        window_tensor = torch.tensor(window, dtype=torch.float)
        # window_tensor = window_tensor.transpose(0, 1)
        # print("self.seizure_info_df",self.seizure_info_df)
        # print("window_tensor=",window_tensor.shape,"file_idx=",file_idx)
        # add a error report
        # if(window_tensor.shape != torch.Size([18, 1024])):
        #     print("error=",window_tensor)
        IDs = within_file_idx
        return window_tensor, torch.tensor(label, dtype=torch.long), additional_info, IDs 
    def load_file(self, file_index):
        # print("edfFiles=",self.edfFiles,"file_index=",file_index)
        filepath = self.edfFiles[file_index]
        if filepath not in self.file_cache:
            eegDataDF, samplFreq, fileStartTime = readEdfFile(filepath)
            self.file_cache[filepath] = (eegDataDF, samplFreq, fileStartTime)
        return self.file_cache[filepath]
        # print("filepath",filepath)
        # eegDataDF, samplFreq, fileStartTime = readEdfFile(filepath)
        # # print("eegDataDF=",eegDataDF)
        # # return eegDataDF, samplFreq, fileStartTime
        # self.file_cache[filepath] = (eegDataDF, samplFreq, fileStartTime)
        # return self.file_cache[filepath]
    def clear_cache(self):
        self.file_cache.clear()
    

def collate_superv(data, max_len=None, task=None, oversample=False):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    features, labels, _,IDs = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

    if task == "classification" and oversample:
        unique_labels, label_counts = np.unique(targets.numpy(), return_counts=True)
        if len(unique_labels) > 1:
            smallest_class_label = unique_labels[np.argmin(label_counts)]
            largest_class_label = unique_labels[np.argmax(label_counts)]
            #print("Smallest class label is {} with count {}".format(smallest_class_label, np.min(label_counts)))
            #print("Largest class label is {} with count {}".format(largest_class_label, np.max(label_counts)))
            replicate_factor = int(np.max(label_counts) / np.min(label_counts))
            if replicate_factor > 1:
                print("Oversampling small class")
                smallest_class_train_indices = [idx for idx in range(len(targets)) if float(targets[idx]) == smallest_class_label]
                #print("Initial count for smallest class is {}".format(len(smallest_class_train_indices)))
                # Oversample the smallest class
                smallest_class_X = np.tile(X.numpy()[smallest_class_train_indices], (replicate_factor, 1, 1))
                #print("Oversampled count for smallest class is {}".format(len(smallest_class_X)))
                # Augment the smallest class
                my_augmenter = (Reverse()
                                + Drift(max_drift=(0.1, 0.5)) @ 0.8)  # with 80% probability, random drift the signal up to 10% - 50%
                smallest_class_X_aug = my_augmenter.augment(smallest_class_X)
                #print("Augmented count for smallest class is {}".format(len(smallest_class_X_aug)))
                X = torch.cat([X, torch.Tensor(smallest_class_X_aug)])
                targets = torch.cat([targets, targets[smallest_class_train_indices].repeat((replicate_factor, 1))])
                padding_masks = torch.cat([padding_masks, padding_masks[smallest_class_train_indices].repeat((replicate_factor, 1))])
                IDs += tuple(np.arange(len(targets), len(targets) + len(smallest_class_X_aug)))
                #print(X.shape, targets.shape, padding_masks.shape, len(IDs))
    # print("size=",targets.shape)
    return X, targets, padding_masks, IDs
