from loadEeg.loadEdf import *
from parametersSetup import *
from VariousFunctionsLib import  *
from evaluate.evaluate import *
import os

#####################################################
# # SIENA DATASET
# dataset='SIENA'
# rootDir=  '../../../../../scratch/dan/physionet.org/files/siena-scalp-eeg/1.0.0' #when running from putty
# rootDir=  '../../../../../shares/eslfiler1/scratch/dan/physionet.org/files/siena-scalp-eeg/1.0.0' #when running from remote desktop
# DatasetPreprocessParams.channelNamesToKeep=DatasetPreprocessParams.channelNamesToKeep_Unipolar

# # SEIZIT DATASET
# dataset='SeizIT1'
# rootDir=  '../../../../../databases/medical/ku-leuven/SeizeIT1/v1_0' #when running from putty
# # rootDir=  '../../../../../shares/eslfiler1/databases/medical/ku-leuven/SeizeIT1/v1_0' #when running from remote desktop
# DatasetPreprocessParams.channelNamesToKeep=DatasetPreprocessParams.channelNamesToKeep_Unipolar

# CHBMIT DATASET
dataset='CHBMIT'
rootDir=  '../../../../../scratch/dan/physionet.org/files/chbmit/1.0.0' #when running from putty
rootDir=  '../../../../../shares/eslfiler1/scratch/dan/physionet.org/files/chbmit/1.0.0' #when running from remote desktop
DatasetPreprocessParams.channelNamesToKeep=DatasetPreprocessParams.channelNamesToKeep_Bipolar

#####################################################
# SET DIFFERENT PARAMETERS
# Set features to use (it will be in the ouput folder name)
# FeaturesParams.featNames = np.array( ['ZeroCross'])
# FeaturesParams.featNames = np.array( ['StandardDeviation','DMe','SKewnesss','SecondOrder'])
# FeaturesParams.featNames = np.array( ['MeanAmpl', 'LineLength'])
# FeaturesParams.featNames = np.array( ['MeanAmpl', 'LineLength','Frequency'])
# FeaturesParams.featNames = np.array( ['MeanAmpl', 'LineLength','Frequency','ZeroCross'])
# FeaturesParams.featNames = np.array( ['MeanDeg'])
FeaturesParams.featNames = np.array( ['StandardDeviation','DMe','SKewnesss','SecondOrder','KatzFD','Network'])
FeaturesParams.featSetNames= FeaturesParams.featNames

#####################################################
# CREATE FOLDER NAMES
appendix='_NewNormalization' #if needed
# Output folder for standardized dataset
outDir= '/home/pliu/git_repo/10_datasets/'+ dataset+ '_Standardized'
os.makedirs(os.path.dirname(outDir), exist_ok=True)
# Output folder with calculated features and  ML model predictions
if (DatasetPreprocessParams.eegDataNormalization==''):
    outDirFeatures = '/home/pliu/git_repo/10_datasets/' + dataset + '_multi_Features_/'
    outPredictionsFolder = '/home/pliu/git_repo/10_datasets/' + dataset + '_multi_TrainingResults' +'_'+StandardMLParams.trainingDataResampling +'_'+ str(StandardMLParams.traininDataResamplingRatio)+'/01_GeneralKfold_' + StandardMLParams.modelType + '_WinStep[' + str(
        FeaturesParams.winLen) + ',' + str(FeaturesParams.winStep) + ']_' + '-'.join(
        FeaturesParams.featNames) + appendix+ '/'
else:
    outDirFeatures= '/home/pliu/git_repo/10_datasets/'+ dataset+ '_multi_Features_'+DatasetPreprocessParams.eegDataNormalization+'/'
    outPredictionsFolder = '/home/pliu/git_repo/10_datasets/' + dataset + '_multi_TrainingResults_' + DatasetPreprocessParams.eegDataNormalization +'_'+StandardMLParams.trainingDataResampling+'_'+ str(StandardMLParams.traininDataResamplingRatio)+ '/01_GeneralKfold_' + StandardMLParams.modelType + '_WinStep[' + str(
        FeaturesParams.winLen) + ',' + str(FeaturesParams.winStep) + ']_' + '-'.join(
        FeaturesParams.featNames) + appendix+ '/'
os.makedirs(os.path.dirname(outDirFeatures), exist_ok=True)
os.makedirs(os.path.dirname(outPredictionsFolder), exist_ok=True)

# testing that folders are correct
# print(os.path.exists(rootDir))
## print(os.listdir('../../../../../'))

#####################################################
# STANDARTIZE DATASET - Only has to be done once
# print('STANDARDIZING DATASET')
# # .edf as output
# if (dataset=='CHBMIT'):
#     # standardizeDataset(rootDir, outDir, origMontage='bipolar-dBanana')  # for CHBMIT
#     standardizeDataset(rootDir, outDir, electrodes= DatasetPreprocessParams.channelNamesToKeep_Bipolar,  inputMontage=Montage.BIPOLAR,ref='bipolar-dBanana' )  # for CHBMIT
# else:
#     standardizeDataset(rootDir, outDir, ref=DatasetPreprocessParams.refElectrode) #for all datasets that are unipolar (SeizIT and Siena)

# # if we want to change output format
# # standardizeDataset(rootDir, outDir, outFormat='csv')
# # standardizeDataset(rootDir, outDir, outFormat='parquet.gzip')

# #####################################################
# EXTRACT ANNOTATIONS - Only has to be done once
if (dataset=='CHBMIT'):
    from loadAnnotations.CHBMITAnnotationConverter import *
elif (dataset == 'SIENA'):
    from loadAnnotations.sienaAnnotationConverter import *
elif (dataset=='SeizIT1'):
    from loadAnnotations.seizeitAnnotationConverter import *

TrueAnnotationsFile = outDir + '/' + dataset + 'AnnotationsTrue.csv'
os.makedirs(os.path.dirname(TrueAnnotationsFile), exist_ok=True)
annotationsTrue= convertAllAnnotations(rootDir, TrueAnnotationsFile )
# annotationsTrue=annotationsTrue.sort_values(by=['subject', 'session'])
# check if all files in annotationsTrue actually exist in standardized dataset
# (if there were problems with files they might have been excluded, so exclude those files)
# TrueAnnotationsFile = outDir + '/' + dataset + 'AnnotationsTrue.csv'
# annotationsTrue=pd.read_csv(TrueAnnotationsFile)
annotationsTrue= checkIfRawDataExists(annotationsTrue, outDir)
annotationsTrue.to_csv(TrueAnnotationsFile, index=False)
TrueAnnotationsFile = outDir + '/' + dataset + 'AnnotationsTrue.csv'
annotationsTrue=pd.read_csv(TrueAnnotationsFile)

#load annotations - if we are not extracting them above
TrueAnnotationsFile = outDir + '/' + dataset + 'AnnotationsTrue.csv'
annotationsTrue=pd.read_csv(TrueAnnotationsFile)

# #####################################################
# EXTRACT FEATURES AND SAVE TO FILES - Only has to be done once
calculateFeaturesForAllFiles('/home/pliu/temp2/14', '/home/pliu/temp2/14', DatasetPreprocessParams, FeaturesParams, DatasetPreprocessParams.eegDataNormalization, outFormat ='parquet.gzip' )
# #
# # # CALCULATE KL DIVERGENCE OF FEATURES
# GeneralParams.patients = [ f.name for f in os.scandir(outDir) if f.is_dir() ]
# GeneralParams.patients.sort() #Sorting them
# FeaturesParams.allFeatNames = constructAllfeatNames(FeaturesParams)
# calculateKLDivergenceForFeatures(dataset, GeneralParams.patients , outDirFeatures, TrueAnnotationsFile, FeaturesParams)

# ####################################################
# # # TRAIN GENERALIZED MODEL
# #
# ## LOAD ALL DATA OF ALL SUBJECTS
# print('LOADING ALL DATA')
# # Create list of all subjects
# GeneralParams.patients = [ f.name for f in os.scandir(outDir) if f.is_dir() ]
# GeneralParams.patients.sort() #Sorting them
# # GeneralParams.patients=GeneralParams.patients[0:3]

# dataAllSubj= loadAllSubjData(dataset, outDirFeatures, GeneralParams.patients, FeaturesParams.featNames,DatasetPreprocessParams.channelNamesToKeep, TrueAnnotationsFile)

##################################
print('TRAINING') # run leave-one-subject-out CV

