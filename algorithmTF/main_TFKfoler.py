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
model_store = '/home/pliu/git_repo/Epileptic_Seizure_Project/algorithmTF/model_store/'
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
        folder_name = f"run_{test_patient}"
        folder_path = os.path.join(model_store, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
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
        # for data, labels in train_loader:
        #     data, labels = data.to(device), labels.to(device)
        # for data, labels in test_loader:
        #     data, labels = data.to(device), labels.to(device)

        feat_dim = winParamsTF.winLen * DatasetPreprocessParamsTF.samplFreq
        # feat_dim = len(DatasetPreprocessParamsTF.channelNamesToKeep_Unipolar)
        TF_model = TSTransformerEncoder(feat_dim=19, max_len=1024,d_model=512,n_heads=4,num_layers=4,dim_feedforward=2048)
        # print(TF_model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        TF_model.to(device)
        print("Using device:", device)
        loss_module = MaskedMSELoss(reduction='none')

        optimizer = torch.optim.Adam(TF_model.parameters(), lr=1e-3)
        runner = AnomalyRunner(model=TF_model, dataloader=train_loader,device=device,loss_module=loss_module,feat_dim=1024,optimizer=optimizer)
        # epoch_metrics = UnsupervisedRunner.train_epoch(10)
        metrics = []       
        tensorboard_writer = SummaryWriter(folder_path) 
        
        # for epoch in range(10):
        #     epoch_metrics = runner.train_epoch(epoch_num=epoch,outputDir=folder_path)
        #     print(f"Epoch {epoch} metrics: {epoch_metrics}")
        # print(epoch_metrics['loss'])
        TF_model.load_state_dict(torch.load('/home/pliu/git_repo/Epileptic_Seizure_Project/algorithmTF/model_epoch_5.pth'))
        test_evaluator = AnomalyRunner(TF_model, test_loader, device, loss_module, feat_dim, 
                                        output_dir='/home/pliu/git_repo/Epileptic_Seizure_Project/algorithmTF')
        
        aggr_metrics_val, best_metrics, best_value = validate(test_evaluator, tensorboard_writer, None,
                                                            epoch=6)
        # # metrics_names, metrics_values = zip(*aggr_metrics_val.items())
        # # metrics.append(list(metrics_values))
        # aggr_metrics_val = validate(test_evaluator, tensorboard_writer, None,
        #                                                     epoch=6)

        
        
        # aggr_metrics_test, best_metrics, best_value = validate(test_evaluator, tensorboard_writer, config, best_metrics,
        #                                                         best_value, epoch=0)
        
        
        
        