from loadEeg.loadEdf import *
from parametersSetupCNN import *
from VariousFunctionsLib import  *
from evaluate.evaluate import *
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import Subset
from torch import nn
from architecture import *
from sklearn.metrics import precision_score, recall_score
from trainer import *
# # #####################################################
# # SIENA DATASET
# dataset='SIENA'
# rootDir=  '../../../../../scratch/dan/physionet.org/files/siena-scalp-eeg/1.0.0' #when running from putty
# rootDir=  '../../../../../shares/eslfiler1/scratch/dan/physionet.org/files/siena-scalp-eeg/1.0.0' #when running from remote desktop
# DatasetPreprocessParamsCNN.channelNamesToKeep=DatasetPreprocessParamsCNN.channelNamesToKeep_Unipolar

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
# folderIn = '/home/pliu/git_repo/10_datasets/SIENA_Standardized'
# labelFile = '/home/pliu/git_repo/10_datasets/SIENA_Standardized/SIENAAnnotationsTrue.csv'
# folerInchb = '/home/pliu/git_repo/CNN/CHBMIT_Standardized'
# labelFilechb = '/home/pliu/git_repo/CNN/CHBMIT_Standardized/CHBMITAnnotationsTrue.csv'
folerInchb = '/home/pliu/testForCNN/CHBCNNtemp'
labelFilechb = '/home/pliu/testForCNN/CHBCNNtemp/CHBMITAnnotationsTrue.csv'
foldtest = '/home/pliu/testForCNN/CNNtesttemp'
labelFiletest = '/home/pliu/testForCNN/CNNtesttemp/CHBMITAnnotationstest.csv'
# setLabelforCNN('/home/pliu/git_repo/10_datasets/SIENA_Standardized/SIENAAnnotationsTrue.csv')
# split the data
# pat0 = EEGDataset()
# ###############################
# ##SIN
# dataset = EEGDataset(folderIn,labelFile)
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# ###############################
# ##CHB
# chbdataset = EEGDataset(folerInchb,labelFilechb)
# print(chbdataset)
testchbdataset=EEGDataset(foldtest,labelFiletest)
chbdataset = EEGDataset(folerInchb,labelFilechb)
print(chbdataset)
train_size_chb = int(0.8 * len(chbdataset))
val_size_chb = len(chbdataset) - train_size_chb
train_dataset_chb, val_dataset_chb = random_split(chbdataset, [train_size_chb, val_size_chb])
# train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
################################
####PARAMETER SETUP
n_classes = 2
batch_size = 512
# ###############################
# ###SIN
# n_chans = 19
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# whole_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
# ###############################
# ###CHB
n_chan_chb = 18
train_loader_chb = DataLoader(train_dataset_chb, batch_size=batch_size, shuffle=True)
val_loader_chb = DataLoader(val_dataset_chb, batch_size=batch_size, shuffle=False)
test_loader_chb = DataLoader(chbdataset, batch_size=batch_size, shuffle=False)
# ####################################
# ### DATASET SIN
# all_data = []
# all_labels = []
# for data, labels in train_loader:
#     all_data.append(data)
#     all_labels.append(labels)
# X_train = torch.cat(all_data)
# y_train = torch.cat(all_labels)
# # print("X_train.shape=",X_train.shape,"y_train.shape=",y_train.shape)
# val_data = []
# val_labels = []
# for data, labels in val_loader:
#     val_data.append(data)
#     val_labels.append(labels)
# X_val = torch.cat(val_data)
# y_val = torch.cat(val_labels)
# # print("X_val.shape=",X_val.shape,"y_val.shape=",y_val.shape)
# ####################################
# # DATASET CHB
all_data_chb = []
all_labels_chb = []
for data, labels in train_loader_chb:
    all_data_chb.append(data)
    all_labels_chb.append(labels)
X_train_chb = torch.cat(all_data_chb)
y_train_chb = torch.cat(all_labels_chb)
# print("X_train.shape=",X_train.shape,"y_train.shape=",y_train.shape)
val_data_chb = []
val_labels_chb = []
for data, labels in val_loader_chb:
    val_data_chb.append(data)
    val_labels_chb.append(labels)
X_val_chb = torch.cat(val_data_chb)
y_val_chb = torch.cat(val_labels_chb)
# print("X_val.shape=",X_val.shape,"y_val.shape=",y_val.shape)
########################################### 
###TRAINING
###########################################
# # # SIN
# model = Net(n_chans, n_classes)
# Model = model
# Train_set=(X_train,y_train)
# print("Train_set=",Train_set[0].shape)
# Val_set=(X_val,y_val)

# Trainer = trainer(model, Train_set, Val_set, 2)
# learning_rate = 0.001
# Trainer.compile(learning_rate=learning_rate)
# epochs = 10
# Tracker = Trainer.train(epochs=epochs, batch_size=64, patience=10, directory='sin.pt')

# # # #########################################
# # # # CHB
model_chb = Net(n_chan_chb,n_classes)
Train_set_chb=(X_train_chb,y_train_chb)
val_dataset_chb=(X_val_chb,y_val_chb)
print("Train_set_chb=",Train_set_chb[0].shape)

Trainer_chb = trainer(model_chb, Train_set_chb, val_dataset_chb, 2)
learning_rate = 0.001
Trainer_chb.compile(learning_rate=learning_rate)
epochs = 10
Tracker = Trainer_chb.train(epochs=epochs, batch_size=64, patience=10, directory='chb2.pt')

print(Tracker)
##################################################################
# # EVLUATE(temp)
# # #####################################
# # #SINEA
# model = Net(n_chans, n_classes)
# model.load_state_dict(torch.load('chb.pt'))
# model.eval()

# all_predictions = []
# all_labels = []
# with torch.no_grad(): 
#     for data, labels in whole_loader:
#         if torch.cuda.is_available():
#             data = data.cuda()
#             model = model.cuda()
            
#         predictions = model(data)
        
#         _, predicted_classes = predictions.max(1)
        
#         all_predictions.extend(predicted_classes.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())
# threshold = 0.8
# with torch.no_grad():
#     for data, labels in whole_loader:
#         if torch.cuda.is_available():
#             data = data.cuda()
#             model = model.cuda()
            
#         outputs = model(data)
#         probabilities = torch.softmax(outputs, dim=1)[:, 1]  
#         predicted_classes = (probabilities > threshold).long()

#         all_predictions.extend(predicted_classes.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())

# accuracy = accuracy_score(all_labels, all_predictions)
# print(f'Accuracy: {accuracy}')
# precision = precision_score(all_labels, all_predictions)
# sensitivity = recall_score(all_labels, all_predictions)
# F1 = (2*precision*sensitivity) / (precision+sensitivity)

# print(f'Precision: {precision}')
# print(f'Sensitivity: {sensitivity}')
# print(f'F1:{F1}')
# # #####################################
# # #CHB-MIT
model_chb = Net(n_chan_chb,n_classes)
model_chb.load_state_dict(torch.load('chb2.pt'))
model_chb.eval()
all_predictions = []
all_labels = []
with torch.no_grad(): 
    for data, labels in test_loader_chb:
        if torch.cuda.is_available():
            data = data.cuda()
            model_chb = model_chb.cuda()
            
        predictions = model_chb(data)
        
        _, predicted_classes = predictions.max(1)
        
        all_predictions.extend(predicted_classes.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
threshold = 0.8
with torch.no_grad():
    for data, labels in test_loader_chb:
        if torch.cuda.is_available():
            data = data.cuda()
            model_chb = model_chb.cuda()
            
        outputs = model_chb(data)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]  
        predicted_classes = (probabilities > threshold).long()

        all_predictions.extend(predicted_classes.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_predictions)
print(f'Accuracy: {accuracy}')
precision = precision_score(all_labels, all_predictions)
sensitivity = recall_score(all_labels, all_predictions)
F1 = (2*precision*sensitivity) / (precision+sensitivity)

print(f'Precision: {precision}')
print(f'Sensitivity: {sensitivity}')
print(f'F1:{F1}')
# with torch.no_grad():
#     for data, labels in whole_loader:
#         if torch.cuda.is_available():
#             data = data.cuda()
#             model = model.cuda()
            
#         outputs = model(data)
#         probabilities = torch.softmax(outputs, dim=1)[:, 1]  # 假设正类是第二列
#         predicted_classes = (probabilities > 0.5).long()
        
#         actual_positives = labels == 1
#         if actual_positives.any():  
#             print("Probabilities of actual positives:", probabilities[actual_positives])
#             print("Predicted classes of actual positives:", predicted_classes[actual_positives])
#             print("Actual labels of positives:", labels[actual_positives])