import sys
sys.path.append(r'../../Epileptic_Seizure_Project')
from loadEeg.loadEdf import *
from parametersSetupTF import *
from VariousFunctionsLib import  *
from evaluate.evaluate import *
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import Subset
from torch import nn
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import KFold
from ts_transformer import model_factory, TSTransformerEncoder
from running import UnsupervisedRunner, AnomalyRunner, validate
from dataset import *
# from parametersSetupTF import PandasTSData
from loss import MaskedMSELoss, NoFussCrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
# # # #####################################################
# # # SIENA DATASET
dataset='SIENA'
rootDir=  '../../../../../scratch/dan/physionet.org/files/siena-scalp-eeg/1.0.0' #when running from putty
rootDir=  '../../../../../shares/eslfiler1/scratch/dan/physionet.org/files/siena-scalp-eeg/1.0.0' #when running from remote desktop
DatasetPreprocessParamsTF.channelNamesToKeep=DatasetPreprocessParamsTF.channelNamesToKeep_Unipolar

# # DatasetPreprocessParamsCNN.channelNamesToKeep=DatasetPreprocessParamsCNN.channelNamesToKeep_Unipolar
# # # # SEIZIT DATASET
# # # dataset='SeizIT1'
# # # rootDir=  '../../../../../databases/medical/ku-leuven/SeizeIT1/v1_0' #when running from putty
# # # rootDir=  '../../../../../shares/eslfiler1/databases/medical/ku-leuven/SeizeIT1/v1_0' #when running from remote desktop
# # # DatasetPreprocessParams.channelNamesToKeep=DatasetPreprocessParams.channelNamesToKeep_Unipolar

# CHBMIT DATASET
# dataset='CHBMIT'
# rootDir=  '../../../../../scratch/dan/physionet.org/files/chbmit/1.0.0' #when running from putty
# rootDir=  '../../../../../shares/eslfiler1/scratch/dan/physionet.org/files/chbmit/1.0.0' #when running from remote desktop
# DatasetPreprocessParamsTF.channelNamesToKeep=DatasetPreprocessParamsTF.channelNamesToKeep_Bipolar
# # # #####################################################
# # # CREATE FOLDER NAMES
# appendix='_NewNormalization' #if needed
# Output folder for standardized dataset
outDir= '/home/pliu/git_repo/10_datasets/'+ dataset+ '_Standardized'
os.makedirs(os.path.dirname(outDir), exist_ok=True)
# Output folder with calculated features and  ML model predictions
if (DatasetPreprocessParamsTF.eegDataNormalization==''):
    outDirFeatures = '/home/pliu/git_repo/10_datasets/' + dataset + '_Features/'
    outPredictionsFolder = '/home/pliu/git_repo/10_datasets/' + dataset + '_TrainingResults' +'_CNN' +'_'+'/01_Kfolder_CNN' + '_WinStep[' + str(
        winParamsTF.winLen) + ',' + str(winParamsTF.winStep) + ']'+ '/'
else:
    outDirFeatures= '/home/pliu/git_repo/10_datasets/'+ dataset+ '_Features_'+DatasetPreprocessParamsTF.eegDataNormalization+'/'
    outPredictionsFolder = '/home/pliu/git_repo/10_datasets/' + dataset + '_TrainingResults_' + DatasetPreprocessParamsTF.eegDataNormalization +'_'+ '/01_General_TF' + '_WinStep[' + str(
        winParamsTF.winLen) + ',' + str(winParamsTF.winStep) + ']_' + '-'.join(
        winParamsTF.featNames) + '/'
os.makedirs(os.path.dirname(outDirFeatures), exist_ok=True)
os.makedirs(os.path.dirname(outPredictionsFolder), exist_ok=True)

# # testing that folders are correct
# print(os.path.exists(rootDir))
# # print(os.listdir('../../../../../'))

# # #####################################################
# # # # STANDARTIZE DATASET - Only has to be done once
# # print('STANDARDIZING DATASET')
# # # .edf as output
# # if (dataset=='CHBMIT'):
# #     # standardizeDataset(rootDir, outDir, origMontage='bipolar-dBanana')  # for CHBMIT
# #     standardizeDataset(rootDir, outDir, electrodes= DatasetPreprocessParamsCNN.channelNamesToKeep_Bipolar,  inputMontage=Montage.BIPOLAR,ref='bipolar-dBanana' )  # for CHBMIT
# # else:
# #     standardizeDataset(rootDir, outDir, ref=DatasetPreprocessParamsCNN.refElectrode) #for all datasets that are unipolar (SeizIT and Siena)

# # # if we want to change output format
# # standardizeDataset(rootDir, outDir, outFormat='csv')
# # standardizeDataset(rootDir, outDir, outFormat='parquet.gzip')

# # # #####################################################
# # # EXTRACT ANNOTATIONS - Only has to be done once
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

# ################################### 
# LOAD ALL DATA OF ALL SUBJECTS
# print('LOADING ALL DATA')
# Create list of all subjects
GeneralParamsTF.patients = [ f.name for f in os.scandir(outDir) if f.is_dir() ]
GeneralParamsTF.patients.sort() #Sorting them
# GeneralParams.patients=GeneralParams.patients[0:3]
# dataAllSubj= loadAllSubjData(dataset, outDirFeatures, GeneralParamsTF.patients, FeaturesParamsTF.featNames,DatasetPreprocessParamsTF.channelNamesToKeep, TrueAnnotationsFile)

# #############################################
# Pre-process features
# normalizer = None
# # if config['norm_from']:
# #     with open(config['norm_from'], 'rb') as f:
# #         norm_dict = pickle.load(f)
# #     normalizer = Normalizer(**norm_dict)
# # elif config['normalization'] is not None:
# normalizer = Normalizer(config['normalization'])
# my_data.feature_df.loc[train_indices] = normalizer.normalize(my_data.feature_df.loc[train_indices])
# if not config['normalization'].startswith('per_sample'):
#     # get normalizing values from training set and store for future use
#     norm_dict = normalizer.__dict__
#     with open(os.path.join(config['output_dir'], 'normalization.pickle'), 'wb') as f:
#         pickle.dump(norm_dict, f, pickle.HIGHEST_PROTOCOL)
# if normalizer is not None:
#     if len(val_indices):
#         val_data.feature_df.loc[val_indices] = normalizer.normalize(val_data.feature_df.loc[val_indices])
#     if len(test_indices):
#         test_data.feature_df.loc[test_indices] = normalizer.normalize(test_data.feature_df.loc[test_indices])
# # ####################################################
# # # TRAIN KFOLDER MODEL

print('TRAINING') # run leave-one-subject-out CV
NonFeatureColumns= ['Subject', 'FileName', 'Time', 'Labels']
AllRes_test=np.zeros((len(GeneralParamsTF.patients),27))

NsubjPerK=int(np.ceil(len(GeneralParamsTF.patients)/GeneralParamsTF.GenCV_numFolds))
for kIndx in range(GeneralParamsTF.GenCV_numFolds):
    patientsToTest=GeneralParamsTF.patients[kIndx*NsubjPerK:(kIndx+1)*NsubjPerK]
    print(GeneralParamsTF.patients)
    print('******')
    print(patientsToTest)
    print('-------')
    #PARAMETER SETUP
    n_classes = 2
    batch_size = 10
    n_channel = len(DatasetPreprocessParamsTF.channelNamesToKeep)
    # FOLDER SETUP
    folderDf = annotationsTrue[annotationsTrue['subject'].isin(patientsToTest)]
    for test_patient in patientsToTest:
        trainPatients = [p for p in patientsToTest if p != test_patient]
        #GENERATE LABEL
        # FOLDER SETUP
        trainFolders = [os.path.join(outDir, p) for p in trainPatients]
        trainLabels = folderDf[folderDf['subject'] != test_patient ]
        testFolder = [os.path.join(outDir, test_patient)]
        testLabels = annotationsTrue[annotationsTrue['subject'] == test_patient]
        # print("testFolder",testFolder)
        # DATA SPILT
        # print("testFolder=",testFolder)
        # print("trainFolders=",trainFolders)
        label_df = annotationsTrue

        testSet = EEGDatasetTest(outDir,testFolder, testLabels, DatasetPreprocessParamsTF.samplFreq, winParamsTF.winLen, winParamsTF.winStep)
        trainSet = EEGDataset(outDir,trainFolders, trainLabels, DatasetPreprocessParamsTF.samplFreq, winParamsTF.winLen, winParamsTF.winStep)
        train_loader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, pin_memory=True,
                                    collate_fn=lambda x: collate_unsuperv(x, max_len=None, mask_compensation=False, task=None, oversample=None))
        test_loader = DataLoader(testSet, batch_size=batch_size, shuffle=True, pin_memory=True,
                                    collate_fn=lambda x: collate_superv(x, max_len=None))
        
        feat_dim = winParamsTF.winLen * DatasetPreprocessParamsTF.samplFreq
        # feat_dim = len(DatasetPreprocessParamsTF.channelNamesToKeep_Unipolar)
        TF_model = TSTransformerEncoder(feat_dim=19, max_len=1024,d_model=512,n_heads=4,num_layers=4,dim_feedforward=2048)
        # print(TF_model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print("Using device:", device)
        # loss_module = MaskedMSELoss(reduction='none')
        loss_module = MaskedMSELoss(reduction='none')
        optimizer = torch.optim.Adam(TF_model.parameters(), lr=1e-3)
        runner = AnomalyRunner(model=TF_model, dataloader=test_loader,device=device,loss_module=loss_module,feat_dim=1024,optimizer=optimizer)
        # epoch_metrics = UnsupervisedRunner.train_epoch(10)
        metrics = []       
        tensorboard_writer = SummaryWriter('/home/pliu/git_repo/Epileptic_Seizure_Project/algorithmTF') 
        # for epoch in range(10):
        #     epoch_metrics = runner.train_epoch(epoch_num=epoch)
        #     print(f"Epoch {epoch} metrics: {epoch_metrics}")
        # # print(epoch_metrics['loss'])
        TF_model.load_state_dict(torch.load('/home/pliu/git_repo/Epileptic_Seizure_Project/algorithmTF/model_epoch_5.pth'))
        test_evaluator = AnomalyRunner(TF_model, test_loader, device, loss_module, feat_dim, 
                                        output_dir='/home/pliu/git_repo/Epileptic_Seizure_Project/algorithmTF')
        
        # aggr_metrics_val, best_metrics, best_value = validate(test_evaluator, tensorboard_writer, None,
        #                                                     epoch=6)
        # metrics_names, metrics_values = zip(*aggr_metrics_val.items())
        # metrics.append(list(metrics_values))
        aggr_metrics_val = validate(test_evaluator, tensorboard_writer, None,
                                                            epoch=6)

        
        
        # aggr_metrics_test, best_metrics, best_value = validate(test_evaluator, tensorboard_writer, config, best_metrics,
        #                                                         best_value, epoch=0)
        
        
        
        
        
        
        
        # for data, labels in test_loader:
        #     print("Data:", data)
        #     print("Labels:", labels)
        # print(testSet)
        # data = PandasTSData(rootDir, testFolder, testLabels)
        # testTF = ImputationDataset(data, indice)
        # print(testTF)
        #temp modify
        # outDir = '/home/pliu/testForCNN/CHBCNNtemp'
        # trainSet = EEGDataset(outDir,trainFolders,trainLabels, DatasetPreprocessParamsCNN.samplFreq, winParamsCNN.winLen, winParamsCNN.winStep)
        # testSet = EEGDataset(outDir,testFolder, testLabels, DatasetPreprocessParamsCNN.samplFreq, winParamsCNN.winLen, winParamsCNN.winStep)
        
        
        
#         train_loader = DataLoader(trainSet, batch_size=batch_size, shuffle=True)
#         test_loader = DataLoader(testSet, batch_size=batch_size, shuffle=True)
        
        
#         # print("train_loader",train_loader)
#         # print("testSet",testSet)
#         all_data = []
#         all_labels = []
#         # for data, labels in test_loader:
#         #     print("Data:", data)
#         #     print("Labels:", labels)
#         #create model
#         model = TSTransformerEncoder()
        
#         for data, labels in train_loader:
#             data = data.to(device)
#             labels = labels.to(device)
#             all_data.append(data)
#             all_labels.append(labels)
#             # print("data",data,";labels=",labels)
#         X_train = torch.cat(all_data)
#         y_train = torch.cat(all_labels)
#         print("X_train.shape=",X_train.shape,"y_train.shape=",y_train.shape)
        
#         test_data = []
#         test_labels = []
#         for data, labels in test_loader:
#             test_data.append(data)
#             test_labels.append(labels)
#         X_val = torch.cat(test_data)
#         y_val = torch.cat(test_labels)
#         print("X_val.shape=",X_val.shape,"y_val.shape=",y_val.shape)

#         #parameter setting
#         feat_dim = 1
#         max_seq_len = 1
#         d_model = 1
#         num_heads = 1
#         num_layers = 1
#         dim_feedforward = 1
#         epoch = 10
#         model = TSTransformerEncoder(feat_dim, max_seq_len, d_model, num_heads,
#                                         num_layers, dim_feedforward)
#         trainer = UnsupervisedRunner(model, train_loader, device, loss_module, my_data.feature_df.shape[1], optimizer=optimizer, l2_reg=output_reg, output_dir=config['output_dir'],
#                                     print_interval=config['print_interval'], console=config['console'], fs=config['fs'], subsample_factor=config['subsample_factor'])
#         aggr_metrics_train = trainer.train_epoch(epoch)
        
#         # evaluator = UnsupervisedRunner(model, loader, device, loss_module,
#         #                             print_interval=config['print_interval'], console=config['console'])













# # # # ####################################################
# # # # # TRAIN KFOLDER MODEL
# # # # ## LOAD ALL DATA OF ALL SUBJECTS
# # # print('LOADING ALL DATA')
# # # Create list of all subjects
# # GeneralParamsCNN.patients = [ f.name for f in os.scandir(outDir) if f.is_dir() ]
# # GeneralParamsCNN.patients.sort() #Sorting them
# # # GeneralParams.patients=GeneralParams.patients[0:3]

# # # dataAllSubj= loadAllSubjData(dataset, outDirFeatures, GeneralParams.patients, FeaturesParams.featNames,DatasetPreprocessParams.channelNamesToKeep, TrueAnnotationsFile)
# # # #
# # print('TRAINING') # run leave-one-subject-out CV
# # NonFeatureColumns= ['Subject', 'FileName', 'Time', 'Labels']
# # AllRes_test=np.zeros((len(GeneralParamsCNN.patients),27))

# # NsubjPerK=int(np.ceil(len(GeneralParamsCNN.patients)/GeneralParamsCNN.GenCV_numFolds))
# # for kIndx in range(GeneralParamsCNN.GenCV_numFolds):
# #     patientsToTest=GeneralParamsCNN.patients[kIndx*NsubjPerK:(kIndx+1)*NsubjPerK]
# #     print(GeneralParamsCNN.patients)
# #     print('******')
# #     print(patientsToTest)
# #     print('-------')
# #     #PARAMETER SETUP
# #     n_classes = 2
# #     batch_size = 256
# #     n_channel = len(DatasetPreprocessParamsCNN.channelNamesToKeep)
# #     # FOLDER SETUP
# #     folderDf = annotationsTrue[annotationsTrue['subject'].isin(patientsToTest)]
# #     for test_patient in patientsToTest:
# #         trainPatients = [p for p in patientsToTest if p != test_patient]
# #         #GENERATE LABEL
# #         # FOLDER SETUP
# #         trainFolders = [os.path.join(outDir, p) for p in trainPatients]
# #         trainLabels = folderDf[folderDf['subject'] != test_patient ]
# #         testFolder = [os.path.join(outDir, test_patient)]
# #         testLabels = annotationsTrue[annotationsTrue['subject'] == test_patient]
# #         # DATA SPILT
# #         # print("testFolder=",testFolder)
# #         # print("trainFolders=",trainFolders)
# #         label_df = annotationsTrue
# #         #temp modify
# #         # outDir = '/home/pliu/testForCNN/CHBCNNtemp'
# #         trainSet = EEGDataset(outDir,trainFolders,trainLabels, DatasetPreprocessParamsCNN.samplFreq, winParamsCNN.winLen, winParamsCNN.winStep)
# #         testSet = EEGDataset(outDir,testFolder, testLabels, DatasetPreprocessParamsCNN.samplFreq, winParamsCNN.winLen, winParamsCNN.winStep)
        
# #         train_loader = DataLoader(trainSet, batch_size=batch_size, shuffle=True)
# #         test_loader = DataLoader(testSet, batch_size=batch_size, shuffle=True)
        
# #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #         print("Using device:", device)
# #         # print("train_loader",train_loader)
# #         # print("testSet",testSet)
# #         all_data = []
# #         all_labels = []
# #         # for data, labels in test_loader:
# #         #     print("Data:", data)
# #         #     print("Labels:", labels)
# #         model = Net(n_channel,n_classes).to(device)
# #         for data, labels in train_loader:
# #             data = data.to(device)
# #             labels = labels.to(device)
# #             all_data.append(data)
# #             all_labels.append(labels)
# #             # print("data",data,";labels=",labels)
# #         X_train = torch.cat(all_data)
# #         y_train = torch.cat(all_labels)
# #         print("X_train.shape=",X_train.shape,"y_train.shape=",y_train.shape)
        
# #         test_data = []
# #         test_labels = []
# #         for data, labels in test_loader:
# #             test_data.append(data)
# #             test_labels.append(labels)
# #         X_val = torch.cat(test_data)
# #         y_val = torch.cat(test_labels)
# #         print("X_val.shape=",X_val.shape,"y_val.shape=",y_val.shape)
# #         # TRAINING

# #         Train_set_chb=(X_train,y_train)
# #         val_dataset_chb=(X_val,y_val)
# #         print("Train_set_chb=",Train_set_chb[0].shape)

# #         Trainer_chb = trainer(model, Train_set_chb, val_dataset_chb, 2)
# #         learning_rate = 0.001
# #         Trainer_chb.compile(learning_rate=learning_rate)
# #         epochs = 20
# #         Tracker = Trainer_chb.train(epochs=epochs, batch_size=64, patience=10, directory='temp.pt')
# #         # print(Tracker)        
# #         # #EVALUATE NAIVE
        
# #         model = Net(n_channel,n_classes)
# #         model.load_state_dict(torch.load('temp.pt'))
# #         model.eval()
# #         all_predictions = []
# #         all_labels = []
# #         with torch.no_grad(): 
# #             for data, labels in test_loader:
# #                 if torch.cuda.is_available():
# #                     data = data.cuda()
# #                     model = model.cuda()
                    
# #                 predictions = model(data)
                
# #                 _, predicted_classes = predictions.max(1)
                
# #                 all_predictions.extend(predicted_classes.cpu().numpy())
# #                 all_labels.extend(labels.cpu().numpy())
# #         threshold = 0.5
# #         with torch.no_grad():
# #             for data, labels in test_loader:
# #                 if torch.cuda.is_available():
# #                     data = data.cuda()
# #                     model = model.cuda()
                    
# #                 outputs = model(data)
# #                 probabilities = torch.softmax(outputs, dim=1)[:, 1]  
# #                 predicted_classes = (probabilities > threshold).long()

# #                 all_predictions.extend(predicted_classes.cpu().numpy())
# #                 all_labels.extend(labels.cpu().numpy())

# #         accuracy = accuracy_score(all_labels, all_predictions)
# #         print(f'Accuracy: {accuracy}')
# #         precision = precision_score(all_labels, all_predictions)
# #         sensitivity = recall_score(all_labels, all_predictions)
# #         F1 = (2*precision*sensitivity) / (precision+sensitivity)

# #         print(f'Precision: {precision}')
# #         print(f'Sensitivity: {sensitivity}')
# #         print(f'F1:{F1}')















# # # # # ##################################
# # # # # folderIn = '/home/pliu/git_repo/10_datasets/SIENA_Standardized'
# # # # # labelFile = '/home/pliu/git_repo/10_datasets/SIENA_Standardized/SIENAAnnotationsTrue.csv'
# # # # # folerInchb = '/home/pliu/git_repo/CNN/CHBMIT_Standardized'
# # # # # labelFilechb = '/home/pliu/git_repo/CNN/CHBMIT_Standardized/CHBMITAnnotationsTrue.csv'
# # #     # folderInchb = '/home/pliu/testForCNN/CHBCNNtemp'
# # #     # labelFilechb = '/home/pliu/testForCNN/CHBCNNtemp/CHBMITAnnotationsTrue.csv'
# # # # # folderInchb = '/home/pliu/git_repo/10_datasets/CHBMIT_Standardized'
# # # # # labelFilechb = '/home/pliu/git_repo/10_datasets/CHBMIT_Standardized/CHBMITAnnotationsTrue.csv'
# # # # # setLabelforCNN('/home/pliu/git_repo/10_datasets/SIENA_Standardized/SIENAAnnotationsTrue.csv')
# # # # ###############################
# # # # split the data#################
# # # # ###############################
# # # # ##SIN
# # # # dataset = EEGDataset(folderIn,labelFile)
# # # # train_size = int(0.8 * len(dataset))
# # # # val_size = len(dataset) - train_size
# # # # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# # # # # ###############################
# # # # # ##CHB
# # # # chbdataset = EEGDataset(folderInchb,labelFilechb)
# # # # train_size_chb = int(0.8 * len(chbdataset))
# # # # val_size_chb = len(chbdataset) - train_size_chb
# # # # train_dataset_chb, val_dataset_chb = random_split(chbdataset, [train_size_chb, val_size_chb])
# # # # # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
# # # # ################################
# # # # ####PARAMETER SETUP
# # # # n_classes = 2
# # # # batch_size = 512
# # # # # ###############################
# # # # # ###SIN
# # # # # n_chans = 19
# # # # # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# # # # # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# # # # # whole_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
# # # # # ###############################
# # # # # ###CHB
# # # # n_chan_chb = 18
# # # # # train_loader_chb = DataLoader(train_dataset_chb, batch_size=batch_size, shuffle=True)
# # # # # val_loader_chb = DataLoader(val_dataset_chb, batch_size=batch_size, shuffle=False)
# # # # # whole_loader_chb = DataLoader(chbdataset, batch_size=batch_size, shuffle=False)
# # # # ####################################
# # # # ### DATASET SIN
# # # # all_data = []
# # # # all_labels = []
# # # # for data, labels in train_loader:
# # # #     all_data.append(data)
# # # #     all_labels.append(labels)
# # # # X_train = torch.cat(all_data)
# # # # y_train = torch.cat(all_labels)
# # # # # print("X_train.shape=",X_train.shape,"y_train.shape=",y_train.shape)
# # # # val_data = []
# # # # val_labels = []
# # # # for data, labels in val_loader:
# # # #     val_data.append(data)
# # # #     val_labels.append(labels)
# # # # X_val = torch.cat(val_data)
# # # # y_val = torch.cat(val_labels)
# # # # # print("X_val.shape=",X_val.shape,"y_val.shape=",y_val.shape)
# # # # ####################################
# # # # # DATASET CHB
# # # # all_data_chb = []
# # # # all_labels_chb = []
# # # # for data, labels in train_loader_chb:
# # # #     all_data_chb.append(data)
# # # #     all_labels_chb.append(labels)
# # # # X_train_chb = torch.cat(all_data_chb)
# # # # y_train_chb = torch.cat(all_labels_chb)
# # # # # print("X_train.shape=",X_train.shape,"y_train.shape=",y_train.shape)
# # # # val_data_chb = []
# # # # val_labels_chb = []
# # # # for data, labels in val_loader_chb:
# # # #     val_data_chb.append(data)
# # # #     val_labels_chb.append(labels)
# # # # X_val_chb = torch.cat(val_data_chb)
# # # # y_val_chb = torch.cat(val_labels_chb)
# # # # print("X_val.shape=",X_val.shape,"y_val.shape=",y_val.shape)
# # # ########################################### 
# # # # ###TRAINING
# # # # print("Training")

# # # # kf = KFold(n_splits=5, shuffle=True, random_state=42)
# # # # print("kf=",kf.split(chbdataset))
# # # # for fold, (train_idx, val_idx) in enumerate(kf.split(chbdataset)):
# # # #     print(f"Training fold {fold + 1}")
# # # #     print("train_idx",train_idx)
# # # #     # 根据索引分割数据集
# # # #     train_data = Subset(chbdataset, train_idx)
# # # #     val_data = Subset(chbdataset, val_idx)

# # # #     # 创建数据加载器
# # # #     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# # # #     val_loader = DataLoader(val_data, batch_size=batch_size)

# # # #     # DATASET CHB
# # # #     all_data_chb = []
# # # #     all_labels_chb = []
# # # #     for data, labels in train_loader:
# # # #         all_data_chb.append(data)
# # # #         all_labels_chb.append(labels)
# # # #     X_train_chb = torch.cat(all_data_chb)
# # # #     y_train_chb = torch.cat(all_labels_chb)
# # # #     # print("X_train.shape=",X_train.shape,"y_train.shape=",y_train.shape)
# # # #     val_data_chb = []
# # # #     val_labels_chb = []
# # # #     for data, labels in val_loader:
# # # #         val_data_chb.append(data)
# # # #         val_labels_chb.append(labels)
# # # #     X_val_chb = torch.cat(val_data_chb)
# # # #     y_val_chb = torch.cat(val_labels_chb)
# # # #     print("X_val.shape=",X_val_chb.shape,"y_val.shape=",y_val_chb.shape)

# # # #     Train_set_chb=(X_train_chb,y_train_chb)
# # # #     val_dataset_chb=(X_val_chb,y_val_chb)
# # # #     print("Train_set_chb=",Train_set_chb[0].shape)

# # # #     # model_kfolder = Net(n_chan_chb, n_classes)
# # # #     # Trainer_chb = trainer(model_kfolder, Train_set_chb, val_dataset_chb, 2)
# # # #     # learning_rate = 0.001
# # # #     # Trainer_chb.compile(learning_rate=learning_rate)
# # # #     # epochs = 10
# # # #     # Tracker = Trainer_chb.train(epochs=epochs, batch_size=64, patience=10, directory='ktest.pt')

# # # #     # print(Tracker)

# # # #     # # # #CHB-MIT
# # # #     model_chb = Net(n_chan_chb,n_classes)
# # # #     model_chb.load_state_dict(torch.load('ktest.pt'))
# # # #     model_chb.eval()
# # # #     all_predictions = []
# # # #     all_labels = []
# # # #     with torch.no_grad(): 
# # # #         for data, labels in val_loader:
# # # #             if torch.cuda.is_available():
# # # #                 data = data.cuda()
# # # #                 model_chb = model_chb.cuda()
                
# # # #             predictions = model_chb(data)
            
# # # #             _, predicted_classes = predictions.max(1)
            
# # # #             all_predictions.extend(predicted_classes.cpu().numpy())
# # # #             all_labels.extend(labels.cpu().numpy())
# # # #     threshold = 0.8
# # # #     with torch.no_grad():
# # # #         for data, labels in val_loader:
# # # #             if torch.cuda.is_available():
# # # #                 data = data.cuda()
# # # #                 model_chb = model_chb.cuda()
                
# # # #             outputs = model_chb(data)
# # # #             probabilities = torch.softmax(outputs, dim=1)[:, 1]  
# # # #             predicted_classes = (probabilities > threshold).long()

# # # #             all_predictions.extend(predicted_classes.cpu().numpy())
# # # #             all_labels.extend(labels.cpu().numpy())

# # # #     accuracy = accuracy_score(all_labels, all_predictions)
# # # #     print(f'Accuracy: {accuracy}')
# # # #     precision = precision_score(all_labels, all_predictions)
# # # #     sensitivity = recall_score(all_labels, all_predictions)
# # # #     F1 = (2*precision*sensitivity) / (precision+sensitivity)

# # # #     print(f'Precision: {precision}')
# # # #     print(f'Sensitivity: {sensitivity}')
# # # #     print(f'F1:{F1}') 