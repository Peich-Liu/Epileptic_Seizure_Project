import sys
sys.path.append(r'../../Epileptic_Seizure_Project')
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
from sklearn.model_selection import KFold
# # # #####################################################
# # # SIENA DATASET
# dataset='SIENA'
# rootDir=  '../../../../../scratch/dan/physionet.org/files/siena-scalp-eeg/1.0.0' #when running from putty
# rootDir=  '../../../../../shares/eslfiler1/scratch/dan/physionet.org/files/siena-scalp-eeg/1.0.0' #when running from remote desktop
# DatasetPreprocessParamsCNN.channelNamesToKeep=DatasetPreprocessParamsCNN.channelNamesToKeep_Unipolar

# # # # SEIZIT DATASET
# # # dataset='SeizIT1'
# # # rootDir=  '../../../../../databases/medical/ku-leuven/SeizeIT1/v1_0' #when running from putty
# # # rootDir=  '../../../../../shares/eslfiler1/databases/medical/ku-leuven/SeizeIT1/v1_0' #when running from remote desktop
# # # DatasetPreprocessParams.channelNamesToKeep=DatasetPreprocessParams.channelNamesToKeep_Unipolar

# # CHBMIT DATASET
dataset='CHBMIT'
rootDir=  '../../../../../scratch/dan/physionet.org/files/chbmit/1.0.0' #when running from putty
rootDir=  '../../../../../shares/eslfiler1/scratch/dan/physionet.org/files/chbmit/1.0.0' #when running from remote desktop
DatasetPreprocessParamsCNN.channelNamesToKeep=DatasetPreprocessParamsCNN.channelNamesToKeep_Bipolar
# # # #####################################################
# # # CREATE FOLDER NAMES
# appendix='_NewNormalization' #if needed
# Output folder for standardized dataset
outDir= '/home/pliu/git_repo/10_datasets/'+ dataset+ '_Standardized'
os.makedirs(os.path.dirname(outDir), exist_ok=True)
# Output folder with calculated features and  ML model predictions
if (DatasetPreprocessParamsCNN.eegDataNormalization==''):
    outDirFeatures = '/home/pliu/git_repo/10_datasets/' + dataset + '_Features/'
    outPredictionsFolder = '/home/pliu/git_repo/10_datasets/' + dataset + '_TrainingResults' +'_CNN' +'_'+'/01_Kfolder_CNN' + '_WinStep[' + str(
        winParamsCNN.winLen) + ',' + str(winParamsCNN.winStep) + ']'+ '/'
else:
    outDirFeatures= '/home/pliu/git_repo/10_datasets/'+ dataset+ '_Features_'+DatasetPreprocessParamsCNN.eegDataNormalization+'/'
    outPredictionsFolder = '/home/pliu/git_repo/10_datasets/' + dataset + '_TrainingResults_' + DatasetPreprocessParamsCNN.eegDataNormalization +'_'+ str(StandardMLParams.traininDataResamplingRatio)+ '/01_General_CNN' + '_WinStep[' + str(
        winParamsCNN.winLen) + ',' + str(winParamsCNN.winStep) + ']_' + '-'.join(
        winParamsCNN.featNames) + '/'
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
# # ####################################################
# # # TRAIN KFOLDER MODEL
# # ## LOAD ALL DATA OF ALL SUBJECTS
# print('LOADING ALL DATA')
# Create list of all subjects
GeneralParamsCNN.patients = [ f.name for f in os.scandir(outDir) if f.is_dir() ]
GeneralParamsCNN.patients.sort() #Sorting them
# GeneralParams.patients=GeneralParams.patients[0:3]
# dataAllSubj= loadAllSubjData(dataset, outDirFeatures, GeneralParamsCNN.patients, None,DatasetPreprocessParamsCNN.channelNamesToKeep, TrueAnnotationsFile)
# print("dataAllSubj=",dataAllSubj)
print('TRAINING') # run leave-one-subject-out CV
NonFeatureColumns= ['Subject', 'FileName', 'Time', 'Labels']
AllRes_test=np.zeros((len(GeneralParamsCNN.patients),27))

patient_indices = {patient: idx for idx, patient in enumerate(GeneralParamsCNN.patients)}
NsubjPerK=int(np.ceil(len(GeneralParamsCNN.patients)/GeneralParamsCNN.GenCV_numFolds))
for kIndx in range(GeneralParamsCNN.GenCV_numFolds):
    patientsToTest=GeneralParamsCNN.patients[kIndx*NsubjPerK:(kIndx+1)*NsubjPerK]
    print(GeneralParamsCNN.patients)
    print('******')
    print(patientsToTest)
    print('-------')
    #PARAMETER SETUP
    n_classes = 2
    batch_size = 256
    n_channel = len(DatasetPreprocessParamsCNN.channelNamesToKeep)
    # FOLDER SETUP
    folderDf = annotationsTrue[annotationsTrue['subject'].isin(patientsToTest)]
    for test_patient in patientsToTest:
        
        patient_index = patient_indices[test_patient]
        trainPatients = [p for p in patientsToTest if p != test_patient]
        print("test_patient=",test_patient)
        print("trainPatients=",trainPatients)
        # FOLDER SETUP
        trainFolders = [os.path.join(outDir, p) for p in trainPatients]
        trainLabels = folderDf[folderDf['subject'] != test_patient ]
        testFolder = [os.path.join(outDir, test_patient)]
        testLabels = annotationsTrue[annotationsTrue['subject'] == test_patient]
        # DATA SPILT
        # print("testFolder=",testFolder)
        # print("trainFolders=",trainFolders)
        label_df = annotationsTrue
        #temp modify
        # outDir = '/home/pliu/testForCNN/CHBCNNtemp'
        trainSet = EEGDataset(outDir,trainFolders,trainLabels, DatasetPreprocessParamsCNN.samplFreq, winParamsCNN.winLen, winParamsCNN.winStep)
        testSet = EEGDatasetTest(outDir,testFolder, testLabels, DatasetPreprocessParamsCNN.samplFreq, winParamsCNN.winLen, winParamsCNN.winStep)
        
        train_loader = DataLoader(trainSet, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(testSet, batch_size=batch_size, shuffle=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        # print("train_loader",train_loader)
        # print("testSet",testSet)
        all_data = []
        all_labels = []
        # for data, labels in test_loader:
        #     print("Data:", data)
        #     print("Labels:", labels)
        model = Net(n_channel,n_classes).to(device)
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)
            all_data.append(data)
            all_labels.append(labels)
            # print("data",data,";labels=",labels)
        X_train = torch.cat(all_data)
        y_train = torch.cat(all_labels)
        print("X_train.shape=",X_train.shape,"y_train.shape=",y_train.shape)
        
        test_data = []
        test_labels = []
        test_labels_for_visual = []
        for data, labels in test_loader:
            test_data.append(data)
            test_labels.append(labels)
            test_labels_for_visual.extend(labels.numpy())
        print("test_data567=",test_data)
        X_val = torch.cat(test_data)
        y_val = torch.cat(test_labels)
        print("X_val.shape=",X_val.shape,"y_val.shape=",y_val.shape)
        # # TRAINING

        # Train_set_chb=(X_train,y_train)
        # val_dataset_chb=(X_val,y_val)
        # print("Train_set_chb=",Train_set_chb[0].shape)

        # Trainer_chb = trainer(model, Train_set_chb, val_dataset_chb, 2)
        # learning_rate = 0.001
        # Trainer_chb.compile(learning_rate=learning_rate)
        # epochs = 20
        # Tracker = Trainer_chb.train(epochs=epochs, batch_size=64, patience=10, directory='temp_{}.pt'.format(test_patient))
        # print(Tracker)        
        # #EVALUATE NAIVE
        (predLabels_test, probabLab_test, acc_test, accPerClass_test) = test_DeepLearningModel(test_loader=test_loader,model_path='temp_{}.pt'.format(test_patient),n_channel=n_channel,n_classes=n_classes)
        print("predLabels_test=",predLabels_test,"probabLab_test",probabLab_test,"acc_test",acc_test,"accPerClass_test", accPerClass_test)
        
        # measure performance
        AllRes_test[patient_index, 0:9] = performance_sampleAndEventBased(predLabels_test, test_labels_for_visual, PerformanceParams)
        # test smoothing - movtest_patient=ng average
        predLabels_MovAvrg = movingAvrgSmoothing(predLabels_test, PerformanceParams.smoothingWinLen,  PerformanceParams.votingPercentage)
        AllRes_test[patient_index, 9:18] = performance_sampleAndEventBased(predLabels_MovAvrg, test_labels_for_visual, PerformanceParams)
        # test smoothing - moving average
        predLabels_Bayes = smoothenLabels_Bayes(predLabels_test, probabLab_test, PerformanceParams.smoothingWinLen, PerformanceParams.bayesProbThresh)
        AllRes_test[patient_index, 18:27] = performance_sampleAndEventBased(predLabels_Bayes, test_labels_for_visual, PerformanceParams)
        outputName = outPredictionsFolder + '/AllSubj_PerformanceAllSmoothing_OldMetrics.csv'
        saveDataToFile(AllRes_test, outputName, 'csv')

        #visualize predictions
        outName=outPredictionsFolder + '/'+ test_patient+'_PredictionsInTime'
        plotPredictionsMatchingInTime(test_labels_for_visual, predLabels_test, predLabels_MovAvrg, predLabels_Bayes, outName, PerformanceParams)

        # print("predLabels_test",predLabels_test)

#         # Saving predicitions in time
#         dataToSave = np.vstack((test_labels_for_visual, probabLab_test, predLabels_test, predLabels_MovAvrg,  predLabels_Bayes)).transpose()   # added from which file is specific part of test set
#         dataToSaveDF=pd.DataFrame(dataToSave, columns=['TrueLabels', 'ProbabLabels', 'PredLabels', 'PredLabels_MovAvrg', 'PredLabels_Bayes'])
#         outputName = outPredictionsFolder + '/Subj' + test_patient + '_'+'CNN'+'_TestPredictions.csv'
#         saveDataToFile(dataToSaveDF, outputName, 'parquet.gzip')

#         # CREATE ANNOTATION FILE
#         predlabels= np.vstack((probabLab_test, predLabels_test, predLabels_MovAvrg,  predLabels_Bayes)).transpose().astype(int)
# #         testPredictionsDF=pd.concat([testData[NonFeatureColumns].reset_index(drop=True), pd.DataFrame(predlabels, columns=['ProbabLabels', 'PredLabels', 'PredLabels_MovAvrg', 'PredLabels_Bayes'])] , axis=1)
#         annotationsTrue=readDataFromFile(TrueAnnotationsFile)
#         annotationAllPred=createAnnotationFileFromPredictions(testPredictionsDF, annotationsTrue, 'PredLabels_Bayes')
#         if (patIndx==0):
#             annotationAllSubjPred=annotationAllPred
#         else:
#             annotationAllSubjPred = pd.concat([annotationAllSubjPred, annotationAllPred], axis=0)
#         #save every time, just for backup
#         PredictedAnnotationsFile = outPredictionsFolder + '/' + dataset + 'AnnotationPredictions.csv'
#         annotationAllSubjPred.sort_values(by=['filepath']).to_csv(PredictedAnnotationsFile, index=False)

        
        
        
        
        
        
        
        
        
# #         # model = Net(n_channel,n_classes)
# #         # model.load_state_dict(torch.load('temp.pt'))
# #         # model.eval()
# #         # all_predictions = []
# #         # all_labels = []
# #         # with torch.no_grad(): 
# #         #     for data, labels in test_loader:
# #         #         if torch.cuda.is_available():
# #         #             data = data.cuda()
# #         #             model = model.cuda()
                    
# #         #         predictions = model(data)
                
# #         #         _, predicted_classes = predictions.max(1)
                
# #         #         all_predictions.extend(predicted_classes.cpu().numpy())
# #         #         all_labels.extend(labels.cpu().numpy())
# #         # threshold = 0.5
# #         # with torch.no_grad():
# #         #     for data, labels in test_loader:
# #         #         if torch.cuda.is_available():
# #         #             data = data.cuda()
# #         #             model = model.cuda()
                    
# #         #         outputs = model(data)
# #         #         probabilities = torch.softmax(outputs, dim=1)[:, 1]  
# #         #         predicted_classes = (probabilities > threshold).long()

# #         #         all_predictions.extend(predicted_classes.cpu().numpy())
# #         #         all_labels.extend(labels.cpu().numpy())

# #         # accuracy = accuracy_score(all_labels, all_predictions)
# #         # print(f'Accuracy: {accuracy}')
# #         # precision = precision_score(all_labels, all_predictions)
# #         # sensitivity = recall_score(all_labels, all_predictions)
# #         # F1 = (2*precision*sensitivity) / (precision+sensitivity)

# #         # print(f'Precision: {precision}')
# #         # print(f'Sensitivity: {sensitivity}')
# #         # print(f'F1:{F1}')
# # #