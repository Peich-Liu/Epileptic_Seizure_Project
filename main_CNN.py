from loadEeg.loadEdf import *
from parametersSetup import *
from VariousFunctionsLib import  *
from evaluate.evaluate import *
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
from architecture import *
from trainer import *
# # #####################################################
# # SIENA DATASET
# dataset='SIENA'
# rootDir=  '../../../../../scratch/dan/physionet.org/files/siena-scalp-eeg/1.0.0' #when running from putty
# rootDir=  '../../../../../shares/eslfiler1/scratch/dan/physionet.org/files/siena-scalp-eeg/1.0.0' #when running from remote desktop
# DatasetPreprocessParams.channelNamesToKeep=DatasetPreprocessParams.channelNamesToKeep_Unipolar

# # # # SEIZIT DATASET
# # # dataset='SeizIT1'
# # # rootDir=  '../../../../../databases/medical/ku-leuven/SeizeIT1/v1_0' #when running from putty
# # # rootDir=  '../../../../../shares/eslfiler1/databases/medical/ku-leuven/SeizeIT1/v1_0' #when running from remote desktop
# # # DatasetPreprocessParams.channelNamesToKeep=DatasetPreprocessParams.channelNamesToKeep_Unipolar

# # # # CHBMIT DATASET
# # # dataset='CHBMIT'
# # # rootDir=  '../../../../../scratch/dan/physionet.org/files/chbmit/1.0.0' #when running from putty
# # # rootDir=  '../../../../../shares/eslfiler1/scratch/dan/physionet.org/files/chbmit/1.0.0' #when running from remote desktop
# # # DatasetPreprocessParams.channelNamesToKeep=DatasetPreprocessParams.channelNamesToKeep_Bipolar
# # #####################################################
# # CREATE FOLDER NAMES
# appendix='_NewNormalization' #if needed
# # Output folder for standardized dataset
# outDir= '/home/pliu/git_repo/10_datasets/'+ dataset+ '_Standardized'
# os.makedirs(os.path.dirname(outDir), exist_ok=True)
# # Output folder with calculated features and  ML model predictions
# if (DatasetPreprocessParams.eegDataNormalization==''):
#     outDirFeatures = '/home/pliu/git_repo/10_datasets/' + dataset + '_Features/'
#     outPredictionsFolder = '/home/pliu/git_repo/10_datasets/' + dataset + '_TrainingResults' +'_'+StandardMLParams.trainingDataResampling +'_'+ str(StandardMLParams.traininDataResamplingRatio)+'/01_General_' + StandardMLParams.modelType + '_WinStep[' + str(
#         FeaturesParams.winLen) + ',' + str(FeaturesParams.winStep) + ']_' + '-'.join(
#         FeaturesParams.featNames) + appendix+ '/'
# else:
#     outDirFeatures= '/home/pliu/git_repo/10_datasets/'+ dataset+ '_Features_'+DatasetPreprocessParams.eegDataNormalization+'/'
#     outPredictionsFolder = '/home/pliu/git_repo/10_datasets/' + dataset + '_TrainingResults_' + DatasetPreprocessParams.eegDataNormalization +'_'+StandardMLParams.trainingDataResampling+'_'+ str(StandardMLParams.traininDataResamplingRatio)+ '/01_General_' + StandardMLParams.modelType + '_WinStep[' + str(
#         FeaturesParams.winLen) + ',' + str(FeaturesParams.winStep) + ']_' + '-'.join(
#         FeaturesParams.featNames) + appendix+ '/'
# os.makedirs(os.path.dirname(outDirFeatures), exist_ok=True)
# os.makedirs(os.path.dirname(outPredictionsFolder), exist_ok=True)

# testing that folders are correct
# print(os.path.exists(rootDir))
## print(os.listdir('../../../../../'))

# #####################################################
# # # STANDARTIZE DATASET - Only has to be done once
# # print('STANDARDIZING DATASET')
# # # .edf as output
# # if (dataset=='CHBMIT'):
# #     # standardizeDataset(rootDir, outDir, origMontage='bipolar-dBanana')  # for CHBMIT
# #     standardizeDataset(rootDir, outDir, electrodes= DatasetPreprocessParams.channelNamesToKeep_Bipolar,  inputMontage=Montage.BIPOLAR,ref='bipolar-dBanana' )  # for CHBMIT
# # else:
# #     standardizeDataset(rootDir, outDir, ref=DatasetPreprocessParams.refElectrode) #for all datasets that are unipolar (SeizIT and Siena)

# #if we want to change output format
# # standardizeDataset(rootDir, outDir, outFormat='csv')
# # standardizeDataset(rootDir, outDir, outFormat='parquet.gzip')

# # #####################################################
# # EXTRACT ANNOTATIONS - Only has to be done once
# if (dataset=='CHBMIT'):
#     from loadAnnotations.CHBMITAnnotationConverter import *
# elif (dataset == 'SIENA'):
#     from loadAnnotations.sienaAnnotationConverter import *
# elif (dataset=='SeizIT1'):
#     from loadAnnotations.seizeitAnnotationConverter import *

# TrueAnnotationsFile = outDir + '/' + dataset + 'AnnotationsTrue.csv'
# os.makedirs(os.path.dirname(TrueAnnotationsFile), exist_ok=True)
# annotationsTrue= convertAllAnnotations(rootDir, TrueAnnotationsFile )
# # annotationsTrue=annotationsTrue.sort_values(by=['subject', 'session'])
# # check if all files in annotationsTrue actually exist in standardized dataset
# # (if there were problems with files they might have been excluded, so exclude those files)
# # TrueAnnotationsFile = outDir + '/' + dataset + 'AnnotationsTrue.csv'
# # annotationsTrue=pd.read_csv(TrueAnnotationsFile)
# annotationsTrue= checkIfRawDataExists(annotationsTrue, outDir)
# annotationsTrue.to_csv(TrueAnnotationsFile, index=False)
# TrueAnnotationsFile = outDir + '/' + dataset + 'AnnotationsTrue.csv'
# annotationsTrue=pd.read_csv(TrueAnnotationsFile)

# #load annotations - if we are not extracting them above
# TrueAnnotationsFile = outDir + '/' + dataset + 'AnnotationsTrue.csv'
# annotationsTrue=pd.read_csv(TrueAnnotationsFile)

# #####################################################
# splitDataIntoWindows('/home/pliu/git_repo/10_datasets/SIENA_Standardized', '/home/pliu/git_repo/test/',DatasetPreprocessParams, FeaturesParams,DatasetPreprocessParams.eegDataNormalization, outFormat ='parquet.gzip')
folderIn = '/home/pliu/git_repo/10_datasets/SIENA_Standardized/PN00'
dataset = EEGDataset(folderIn=folderIn)
print(dataset)
# dataset = EEGWindowsDataset(folder_path='/home/pliu/git_repo/test2')
# data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
# # print(dataset)

# # ####################################################
# # # TRAIN GENERALIZED MODEL
# #
# # ## LOAD ALL DATA OF ALL SUBJECTS
# print('LOADING ALL DATA')
# # Create list of all subjects
# GeneralParams.patients = [ f.name for f in os.scandir(outDir) if f.is_dir() ]
# GeneralParams.patients.sort() #Sorting them
# # GeneralParams.patients=GeneralParams.patients[0:3]

# dataAllSubj= loadAllSubjData(dataset, outDirFeatures, GeneralParams.patients, FeaturesParams.featNames,DatasetPreprocessParams.channelNamesToKeep, TrueAnnotationsFile)
# #
# ##################################
print('TRAINING') # run leave-one-subject-out CV
# edfFiles = np.sort(glob.glob(os.path.join(folderIn, '**/*.edf'), recursive=True))
# eegDataDF, samplFreq , fileStartTime= readEdfFile(edfFiles)  # Load data

full_dataset = EEGDataset(folderIn='/home/pliu/testForCNN/PN00')

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(train_loader)

n_classes = 2  
n_chans = 19  
model = Net(n_chans, n_classes)

# 准备训练集和验证集的特征和标签
X_train, y_train = next(iter(train_loader))
X_val, y_val = next(iter(val_loader))

# 初始化trainer
trainer_instance = trainer(model, (X_train, y_train), (X_val, y_val), n_classes)

# 编译模型
trainer_instance.compile(learning_rate=0.001)

# 训练模型
tracker = trainer_instance.train(epochs=50, batch_size=32, patience=10, directory='model.pt')

# 保存训练和验证损失
train_loss = tracker['train_tracker']
val_loss = tracker['val_tracker']
print(train_loss,",",val_loss)