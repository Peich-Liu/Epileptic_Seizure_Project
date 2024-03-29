''' file with all parameters'''
import numpy as np
import pickle
# from sklearn.ensemble import RUSBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import torch
from torch.utils.data import Dataset, DataLoader
import os
from VariousFunctionsLib import  *


class GeneralParams:
    patients=[]  #on which subjects to train and test
    PersCV_MinTrainHours=5 #minimum number of hours we need to start training in personal model
    PersCV_CVStepInHours=1 #how often we retrain and on how much next data we test
    GenCV_numFolds=5

##############################################################################################
#### PREPROCESSING PARAMETERS
class DatasetPreprocessParams: # mostly based on CHB-MIT dataset
    dataset = 'CHBMIT'
    samplFreq = 256  # sampling frequency of data
    #Channels structure
    #Unipolar channels
    channelNamesToKeep_Unipolar = ('Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6')
    refElectrode ='Cz' #default 'Cz' or 'Avrg', or any of channels listed above
    #Bipolar channels
    channelNamesToKeep_Bipolar = ('Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fz-Cz', 'Cz-Pz', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2')
    
    # channelNamesToKeep_Bipolar = ('Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'Fz-Cz', 'Cz-Pz',
    #                    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fp2-F8', 'F8-T8', 'T8-P8', 'P8-O2') # TODO for old features
    # refElectrode='bipolar-dBanana' #if bipolar then ref electrode is not needed so put 'bipolar-dBanana'
    channelNamesToKeep=channelNamesToKeep_Unipolar

    # raw EEG data normalization
    eegDataNormalization='' # '' for none, 'NormWithPercentile', or 'QuantileNormalization'
    
    def updateDatasetPreprocessParams(params):
        DatasetPreprocessParams.channelNamesToKeep_Unipolar = params.get('Unipolar',DatasetPreprocessParams.channelNamesToKeep_Unipolar)
        DatasetPreprocessParams.channelNamesToKeep_Bipolar = params.get('Bipolar',DatasetPreprocessParams.channelNamesToKeep_Bipolar)
        # DatasetPreprocessParams.refElectrode = params.get('refElectrode',DatasetPreprocessParams.refElectrode)
        DatasetPreprocessParams.samplFreq = params.get('samplFreq',DatasetPreprocessParams.samplFreq)
        DatasetPreprocessParams.eegDataNormalization = params.get('eegDataNormalization',DatasetPreprocessParams.eegDataNormalization)
        DatasetPreprocessParams.dataset = params.get('dataset',DatasetPreprocessParams.dataset)
        print("uni",DatasetPreprocessParams.channelNamesToKeep_Unipolar)

##############################################################################################
#### FEATURES PARAMETERS

def constructAllfeatNames(FeaturesParams ):
    allFeatNames=[]
    for fIndx, fName in enumerate(FeaturesParams.featSetNames):
        if (fName=='MeanAmpl'):
            allFeatNames.extend(FeaturesParams.indivFeatNames_MeanAmpl)
        elif (fName=='LineLength'):
            allFeatNames.extend(FeaturesParams.indivFeatNames_LL)
        elif (fName=='Frequency'):
            allFeatNames.extend(FeaturesParams.indivFeatNames_Freq)
        elif (fName=='StandardDeviation'):
            allFeatNames.extend(FeaturesParams.indivFeatNames_SD)
        elif (fName=='DMe'):
            allFeatNames.extend(FeaturesParams.indivFeatNames_DMe)
        elif (fName=='SKewnesss'):
            allFeatNames.extend(FeaturesParams.indivFeatNames_Skew)
        elif (fName=='SecondOrder'):
            allFeatNames.extend(FeaturesParams.indivFeatNames_SO)
        elif (fName=='KatzFD'):
            allFeatNames.extend(FeaturesParams.indivFeatNames_KatzFD)
        elif (fName=='MeanDeg'):
            allFeatNames.extend(FeaturesParams.indivFeatNames_MD)
        elif (fName=='MeanBetw'):
            allFeatNames.extend(FeaturesParams.indivFeatNames_MB)
        elif (fName=='MeanClose'):
            allFeatNames.extend(FeaturesParams.indivFeatNames_MC)
        elif (fName=='Network'):
            allFeatNames.extend(FeaturesParams.indivFeatNames_NW)
        elif (fName=='ZeroCrossAbs'):
            for i in range(len(FeaturesParams.ZC_thresh_arr)):
                allFeatNames.extend(['ZCThr'+ str(FeaturesParams.ZC_thresh_arr[i])])
        elif (fName=='ZeroCrossRel' or fName=='ZeroCross'):
            for i in range(len(FeaturesParams.ZC_thresh_arr_rel)):
                allFeatNames.extend(['ZCThr'+ str(FeaturesParams.ZC_thresh_arr_rel[i])])
    return(allFeatNames)

class FeaturesParams:
    #window size and step in which window is moved
    winLen= 0.391 #in seconds, window length on which to calculate features
    winStep= 0.391 #in seconds, step of moving window length

    #normalization of feature values or not
    featNorm = 'Norm' #'', 'Norm&Discr', 'Norm'

    #features extracted from data
    featSetNames = np.array( ['MeanAmpl', 'LineLength', 'Frequency', 'ZeroCross','StandardDeviation','DMe','SKewnesss','SecondOrder','KatzFD','SRP','RPcla'])
    # featSetNames = np.array( ['MeanAmpl', 'LineLength', 'Frequency', 'ZeroCross'])
    featNames = np.array(['MeanAmpl', 'LineLength', 'Frequency'])
    allFeatName = '-'.join(featSetNames)

    #individual features within each set of features
    indivFeatNames_MeanAmpl=['meanAmpl']
    indivFeatNames_LL=['lineLenth']
    indivFeatNames_Freq = ['p_dc_rel', 'p_mov_rel', 'p_delta_rel', 'p_theta_rel', 'p_alfa_rel', 'p_middle_rel', 'p_beta_rel', 'p_gamma_rel', 'p_dc', 'p_mov', 'p_delta', 'p_theta', 'p_alfa', 'p_middle', 'p_beta', 'p_gamma', 'p_tot']
    indivFeatNames_SD = ['StandardDeviation']
    indivFeatNames_DMe = ['DMe']
    indivFeatNames_Skew = ['SKewnesss']
    indivFeatNames_SO = ['SecondOrder']
    indivFeatNames_KatzFD = ['KatzFD']
    indivFeatNames_NW = ['MeanDeg','MeanBetw','MeanClose']


    #ZC features params
    ZC_thresh_type='rel' #'abs' or 'rel'
    ZC_thresh_arr_rel=[ 0.25, 0.50, 0.75, 1, 1.5]
    ZC_thresh_arr = [16, 32, 64, 128, 256]
    
    def updateFeaturesParams(params):
        FeaturesParams.winLen = params.get('winLen',FeaturesParams.winLen)
        FeaturesParams.winStep = params.get('winStep',FeaturesParams.winStep)
        FeaturesParams.featNorm = params.get('featNorm',FeaturesParams.featNorm)
        print("uni",DatasetPreprocessParams.channelNamesToKeep_Unipolar)
        

#compile list of all features
FeaturesParams.allFeatNames=constructAllfeatNames(FeaturesParams )

##############################################################################################
#### MACHINE LEARNING MODELS PARAMETERS

class StandardMLParams:

    modelType='RUSboost' #'KNN', 'SVM', 'DT', 'RF','BaggingClassifier','AdaBoost'

    # Data under/over sampling
    trainingDataResampling='NoResampling' #'NoResampling','ROS','SMOTE', 'RUS','TomekLinks', 'SMOTEtomek', 'SMOTEENN
    traininDataResamplingRatio='auto' #'auto', 0.2, 0.5
    # samplingStrategy='default' # depends on resampling, but if 'default' then default for each resampling type, otherwise now implemented only for RUS if not default

    #KNN parameters
    KNN_n_neighbors=5
    KNN_metric='euclidean' #'euclidean', 'manhattan'
    #SVM parameters
    SVM_kernel = 'linear'  # 'linear', 'rbf','poly'
    SVM_C = 1  # 1,100,1000
    SVM_gamma = 'auto' # 0  # 0,10,100
    #DecisionTree and random forest parameters
    DecisionTree_criterion = 'gini'  # 'gini', 'entropy'
    DecisionTree_splitter = 'best'  # 'best','random'
    DecisionTree_max_depth = 0  # 0, 2, 5,10,20
    RandomForest_n_estimators = 100 #10,50, 100,250
    #Bagging, boosting classifier parameters
    Bagging_base_estimator='SVM' #'SVM','KNN', 'DecisionTree'
    Bagging_n_estimators = 100  # 10,50, 100,250
    #RUSboost parameter
    RUS_estimator=DecisionTreeClassifier(max_leaf_nodes=20)
    RUS_n_estimators=30
    RUS_learning_rate=0.1
    RUS_algorithm='SAMME.R'
    RUS_sampling_strategy='auto'
    RUS_replacement=False
    RUS_random_state=None
    RUS_base_estimator='deprecated'


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

    predictionFreq=1/FeaturesParams.winStep #ultimate frequency of predictions



#SAVING SETUP once again to update if new info
with open('../PARAMETERS.pickle', 'wb') as f:
    pickle.dump([GeneralParams, DatasetPreprocessParams, FeaturesParams, StandardMLParams, PerformanceParams], f)