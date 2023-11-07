from loadEeg.loadEdf import *
from parametersSetupCNN import *
from VariousFunctionsLib import  *
from evaluate.evaluate import *
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import Subset
from torch import nn
from architecture import *
from trainer import *
# # #####################################################
# SIENA DATASET
dataset='SIENA'
rootDir=  '../../../../../scratch/dan/physionet.org/files/siena-scalp-eeg/1.0.0' #when running from putty
rootDir=  '../../../../../shares/eslfiler1/scratch/dan/physionet.org/files/siena-scalp-eeg/1.0.0' #when running from remote desktop
DatasetPreprocessParamsCNN.channelNamesToKeep=DatasetPreprocessParamsCNN.channelNamesToKeep_Unipolar

# # # SEIZIT DATASET
# # dataset='SeizIT1'
# # rootDir=  '../../../../../databases/medical/ku-leuven/SeizeIT1/v1_0' #when running from putty
# # rootDir=  '../../../../../shares/eslfiler1/databases/medical/ku-leuven/SeizeIT1/v1_0' #when running from remote desktop
# # DatasetPreprocessParams.channelNamesToKeep=DatasetPreprocessParams.channelNamesToKeep_Unipolar

# # # CHBMIT DATASET
# # dataset='CHBMIT'
# # rootDir=  '../../../../../scratch/dan/physionet.org/files/chbmit/1.0.0' #when running from putty
# # rootDir=  '../../../../../shares/eslfiler1/scratch/dan/physionet.org/files/chbmit/1.0.0' #when running from remote desktop
# # DatasetPreprocessParams.channelNamesToKeep=DatasetPreprocessParams.channelNamesToKeep_Bipolar
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
# folderIn = '/home/pliu/git_repo/10_datasets/SIENA_Standardized/PN14'
# dataset = EEGDataset(folderIn=folderIn)
# print(dataset)
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
folderIn = '/home/pliu/testForCNN/PN00'
labelFile = '/home/pliu/git_repo/10_datasets/SIENA_Standardized/SIENAAnnotationsTrue.csv'
# setLabelforCNN('/home/pliu/git_repo/10_datasets/SIENA_Standardized/SIENAAnnotationsTrue.csv')



# folderIn = 'path_to_your_edf_files'  # Replace with the actual path to your EDF files
dataset = EEGDataset(folderIn,labelFile)

# 定义训练集和验证集的切割点
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# 顺序分割数据集
train_indices = list(range(0, train_size))
val_indices = list(range(train_size, len(dataset)))


train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# print(train_loader)

# all_data = []
# all_labels = []
# for data, labels in train_loader:
#     all_data.append(data)
#     all_labels.append(labels)

# X_train = torch.cat(all_data)
# y_train = torch.cat(all_labels)

# n_chans = 19
# n_classes = 2
# model = Net(n_chans, n_classes)

# val_data = []
# val_labels = []
# for data, labels in val_loader:
#     val_data.append(data)
#     val_labels.append(labels)

# # 将所有数据和标签堆叠成一个大的批次
# X_val = torch.cat(val_data)
# y_val = torch.cat(val_labels)
# # print(y_train)
# n_chans = 19
# n_classes = 2
# model = Net(n_chans, n_classes)
# Model = model
# Train_set=(X_train,y_train)
# Val_set=(X_val,y_val)
# # 初始化 trainer
# n_classes = 2  # 根据您的任务设置类别数
# Trainer = trainer(model, Train_set, Val_set, 2)

# # 编译 trainer
# learning_rate = 0.001
# Trainer.compile(learning_rate=learning_rate)

# # 训练模型
# epochs = 10
# Trainer.train(epochs)