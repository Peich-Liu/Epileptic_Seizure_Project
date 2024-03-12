import sys
import os
sys.path.append(r'../../Epileptic_Seizure_Project')
from loadEeg.loadEdf import *
from algorithmCnn.parametersSetupCNN import *
from algorithmCnn.architecture import *
from algorithmCnn.trainer import *
from VariousFunctionsLib import  *
from evaluate.evaluate import *

from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import Subset
from torch import nn
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import KFold
from pandas import *
import csv
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
def trainCNNExp02():
    #####################################################
    #Dataset Setting
    dataset = DatasetPreprocessParamsCNN.dataset
    if dataset == 'SIENA':
        ##SIENA DATASET
        rootDir=  '../../../../../scratch/dan/physionet.org/files/siena-scalp-eeg/1.0.0' #when running from putty
        rootDir=  '../../../../../shares/eslfiler1/scratch/dan/physionet.org/files/siena-scalp-eeg/1.0.0' #when running from remote desktop
        DatasetPreprocessParamsCNN.channelNamesToKeep=DatasetPreprocessParamsCNN.channelNamesToKeep_Unipolar
    elif dataset == 'SeizIT1':
        # SEIZIT DATASET
        rootDir=  '../../../../../databases/medical/ku-leuven/SeizeIT1/v1_0' #when running from putty
        rootDir=  '../../../../../shares/eslfiler1/databases/medical/ku-leuven/SeizeIT1/v1_0' #when running from remote desktop
        DatasetPreprocessParamsCNN.channelNamesToKeep=DatasetPreprocessParamsCNN.channelNamesToKeep_Bipolar
    elif dataset == 'CHBMIT':
        # CHBMIT DATASET
        rootDir=  '../../../../../scratch/dan/physionet.org/files/chbmit/1.0.0' #when running from putty
        rootDir=  '../../../../../shares/eslfiler1/scratch/dan/physionet.org/files/chbmit/1.0.0' #when running from remote desktop
        DatasetPreprocessParamsCNN.channelNamesToKeep=DatasetPreprocessParamsCNN.channelNamesToKeep_Bipolar
    # # # #####################################################
    # # # CREATE FOLDER NAMES
    # appendix='_NewNormalization' #if needed
    # Output folder for standardized dataset
    outDir= 'DataStore/'+ dataset+ '_Standardized'
    os.makedirs(os.path.dirname(outDir), exist_ok=True)
    # Output folder with calculated features and  ML model predictions
    if (DatasetPreprocessParamsCNN.eegDataNormalization==''):
        outPredictionsFolder = outDir + dataset + 'EXP02data_new_TrainingResults' +'_'+'/01_Kfolder_CNN' + '_WinStep[' + str(
            winParamsCNN.winLen) + ',' + str(winParamsCNN.winStep) + ']'+'/'
    else:
        outPredictionsFolder = outDir + dataset + '_new_new_TrainingResults_' + DatasetPreprocessParamsCNN.eegDataNormalization +'_'+  '/01_General_CNN' + '_WinStep[' + str(
            winParamsCNN.winLen) + ',' + str(winParamsCNN.winStep) + ']_' + '/'
    # os.makedirs(os.path.dirname(outDirFeatures), exist_ok=True)
    os.makedirs(os.path.dirname(outPredictionsFolder), exist_ok=True)
    # # testing that folders are correct
    # print(os.path.exists(rootDir))
    # # print(os.listdir('../../../../../'))
    # #####################################################
    # # STANDARTIZE DATASET - Only has to be done once
    # print('STANDARDIZING DATASET')
    # .edf as output
    # if (dataset=='CHBMIT'):
    #     # standardizeDataset(rootDir, outDir, origMontage='bipolar-dBanana')  # for CHBMIT
    #     standardizeDataset(rootDir, outDir, electrodes= DatasetPreprocessParamsCNN.channelNamesToKeep_Bipolar,  inputMontage=Montage.BIPOLAR,ref='bipolar-dBanana' )  # for CHBMIT
    # else:
    #     standardizeDataset(rootDir, outDir, ref=DatasetPreprocessParamsCNN.refElectrode) #for all datasets that are unipolar (SeizIT and Siena)
    # if we want to change output format
    # standardizeDataset(rootDir, outDir, electrodes= DatasetPreprocessParamsCNN.channelNamesToKeep_Bipolar,  inputMontage=Montage.BIPOLAR,ref='bipolar-dBanana')  # .edf as output
    # standardizeDataset(rootDir, outDir, electrodes= DatasetPreprocessParamsCNN.channelNamesToKeep_Bipolar,  inputMontage=Montage.BIPOLAR,ref='bipolar-dBanana', outFormat='csv')
    # standardizeDataset(rootDir, outDir, outFormat='parquet.gzip')
    # # # #####################################################
    # # # EXTRACT ANNOTATIONS - Only has to be done once
    if (dataset=='CHBMIT'):
        from loadAnnotations.CHBMITAnnotationConverter import convertAllAnnotations, checkIfRawDataExists
    elif (dataset == 'SIENA'):
        from loadAnnotations.sienaAnnotationConverter import convertAllAnnotations, checkIfRawDataExists
    elif (dataset=='SeizIT1'):
        from loadAnnotations.seizeitAnnotationConverter import convertAllAnnotations, checkIfRawDataExists
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
    # GeneralParamsCNN.patients=GeneralParamsCNN.patients[:5]
    # dataAllSubj= loadAllSubjData(dataset, outDirFeatures, GeneralParamsCNN.patients, None,DatasetPreprocessParamsCNN.channelNamesToKeep, TrueAnnotationsFile)
    # print("dataAllSubj=",dataAllSubj)
    # ##########################
    ###TRAINING
    print('TRAINING') # run leave-one-subject-out CV
    AllRes_test=np.zeros((len(GeneralParamsCNN.patients),27))

    patient_indices = {patient: idx for idx, patient in enumerate(GeneralParamsCNN.patients)}
    NsubjPerK=int(np.ceil(len(GeneralParamsCNN.patients)/GeneralParamsCNN.GenCV_numFolds))
    for kIndx in range(GeneralParamsCNN.GenCV_numFolds):
        patientsToTest=GeneralParamsCNN.patients[kIndx*NsubjPerK:(kIndx+1)*NsubjPerK]
        print(patientsToTest)
        # patientsToTest=GeneralParamsCNN.patients[kIndx*NsubjPerK:(kIndx+1)*NsubjPerK]
        # print(GeneralParamsCNN.patients)
        print('******')
        print(patientsToTest)
        print('-------')
        #PARAMETER SETUP
        n_classes = 2
        batch_size = 32
        n_channel = len(DatasetPreprocessParamsCNN.channelNamesToKeep)
        # FOLDER SETUP
        folderDf = annotationsTrue[annotationsTrue['subject'].isin(patientsToTest)]
        for p, test_patient in enumerate(patientsToTest):
            patIndx = kIndx*NsubjPerK+p
            patient_index = patient_indices[test_patient]
            trainPatients = [p for p in patientsToTest if p != test_patient]

            # FOLDER SETUP
            trainFolders = [os.path.join(outDir, p) for p in trainPatients]
            trainLabels = folderDf[folderDf['subject'] != test_patient ]
            testFolder = [os.path.join(outDir, test_patient)]
            testLabels = folderDf[folderDf['subject'] == test_patient ]
            # DATA SPILT
            label_df = annotationsTrue
            #temp modify
            trainSet = EEGDataset(outDir,trainFolders,trainLabels, DatasetPreprocessParamsCNN.samplFreq, winParamsCNN.winLen, winParamsCNN.winStep, DatasetPreprocessParamsCNN.eegDataNormalization)
            train_loader = DataLoader(trainSet, batch_size=batch_size, shuffle=False)
            # testSet = EEGDatasetTest(outDir,testFolder, testLabels, DatasetPreprocessParamsCNN.samplFreq, winParamsCNN.winLen, winParamsCNN.winStepTest, DatasetPreprocessParamsCNN.eegDataNormalization)
            
            testSet = EEGDatasetTest(outDir,testFolder, testLabels, DatasetPreprocessParamsCNN.samplFreq, winParamsCNN.winLen, winParamsCNN.winStep, DatasetPreprocessParamsCNN.eegDataNormalization)
            test_data_pred = []
            info_path = 'info_' + test_patient + '.csv'
            #### load data info
            for i in range(len(testSet)):
                _, label, additional_info = testSet[i]
                row = {
                    'Subject': additional_info['subject'],
                    'FileName': additional_info['filepath'],
                    'Time': additional_info['startTime'],
                    'Labels': label.item()  
                }
                test_data_pred.append(row)
            test_data_df = pd.DataFrame(test_data_pred)
            test_loader = DataLoader(testSet, batch_size=batch_size, shuffle=False,collate_fn=custom_collate_fn)

            all_data = []
            all_labels = []
            # model = Net(n_channel,n_classes).to(device)
            model = Net(n_channel,n_classes)
            test_data = []
            test_labels = []
            test_labels_for_visual = []
            for data,labels,_ in test_loader:
                # data = data.to(device)
                # labels = labels.to(device)
                # data = batch[0]
                # labels = batch[1]
                test_data.append(data)
                test_labels.append(labels)
                test_labels_for_visual.extend(labels.numpy())
            X_test = torch.cat(test_data)
            y_test = torch.cat(test_labels)
            # print("test_labels_for_visual",test_labels_for_visual)
            np.savetxt('test_labels_for_visual.txt', test_labels_for_visual)
            
            for data, labels in train_loader:
                # data = data.to(device)
                # labels = labels.to(device)
                all_data.append(data)
                all_labels.append(labels)
            # print("trainlabel",all_labels)
                # print("data",data,";labels=",labels)
            X_train = torch.cat(all_data)
            y_train = torch.cat(all_labels)
            # X_train.to(device)
            # y_train.to(device)
            print("X_train.shape=",X_train.shape,"y_train.shape=",y_train.shape)


            # ########################################## 
            # #TRAINING
            Train_set_chb=(X_train,y_train)
            # val_dataset_chb=(X_val,y_val)
            val_dataset_chb=(X_test,y_test)
            
            print("Train_set_chb=",Train_set_chb[0].shape)

            Trainer_chb = trainer(model, Train_set_chb, val_dataset_chb, 2)
            learning_rate = 0.001
            Trainer_chb.compile(learning_rate=learning_rate)
            epochs = 20
            print(test_patient)
            Tracker = Trainer_chb.train(epochs=epochs, batch_size=64, patience=10, directory='CNN_{}.pt'.format(test_patient))
            filename = f'training_output_subject_{test_patient}.csv'
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Epoch Number", "Train Loss", "Val Loss"])

                for epoch, (train_loss, val_loss) in enumerate(zip(Tracker['train_tracker'], Tracker['val_tracker'])):
                    writer.writerow([epoch, train_loss, val_loss])

            print(f"Training data saved to {filename}")
            print(Tracker)        
            # ########################################## 
            # #EVALUATE
            (predLabels_test, probabLab_test, acc_test, accPerClass_test) = test_DeepLearningModel(test_loader=test_loader,model_path='CNN_{}.pt'.format(test_patient),n_channel=n_channel,n_classes=n_classes)
            # print("predLabels_test=",predLabels_test,"probabLab_test",probabLab_test,"acc_test",acc_test,"accPerClass_test", accPerClass_test)
            # print()
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
            
            # test_labels_for_visual.csv('cnn_label.csv')
            #visualize predictions
            outName=outPredictionsFolder + '/'+ test_patient+'_PredictionsInTime'
            plotPredictionsMatchingInTime(test_labels_for_visual, predLabels_test, predLabels_MovAvrg, predLabels_Bayes, outName, PerformanceParams)

            test_labels_for_visual = np.array(test_labels_for_visual)
            # test_labels_for_visual_reshaped = test_labels_for_visual.reshape(-1, 1)
            # print("test_labels_for_visual",test_labels_for_visual_reshaped)
            # predLabels = np.max(probabLab_test, axis=1)
            predLabels = probabLab_test[:,1]
            print("probabLab_test",predLabels)
            print("predLabels_test",predLabels_test)
            print("predLabels_MovAvrg",predLabels_MovAvrg)
            # # Saving predicitions in time
            # #############
            # #need modify#
            # #############
            dataToSave = np.vstack((test_labels_for_visual, predLabels, predLabels_test, predLabels_MovAvrg,  predLabels_Bayes)).transpose()   # added from which file is specific part of test set
            dataToSaveDF=pd.DataFrame(dataToSave, columns=['TrueLabels', 'ProbabLabels', 'PredLabels', 'PredLabels_MovAvrg', 'PredLabels_Bayes'])
            outputName = outPredictionsFolder + '/Subj' + test_patient + '_'+'CNN'+'_TestPredictions.csv'
            saveDataToFile(dataToSaveDF, outputName, 'parquet.gzip')

            # # CREATE ANNOTATION FILE
            predlabels= np.vstack((predLabels, predLabels_test, predLabels_MovAvrg,  predLabels_Bayes)).transpose().astype(int)
            testPredictionsDF=pd.concat([test_data_df, pd.DataFrame(predlabels, columns=['ProbabLabels', 'PredLabels', 'PredLabels_MovAvrg', 'PredLabels_Bayes'])] , axis=1)
            # print(testPredictionsDF)
            annotationsTrue=readDataFromFile(TrueAnnotationsFile)
            # print(annotationsTrue)
            # quit()
            annotationAllPred=createAnnotationFileFromPredictions(testPredictionsDF, annotationsTrue, 'PredLabels_MovAvrg')
            print(annotationAllPred)
            # quit()
            if (patIndx==0):
                annotationAllSubjPred=annotationAllPred
            else:
                annotationAllSubjPred = pd.concat([annotationAllSubjPred, annotationAllPred], axis=0)
            #save every time, just for backup
            PredictedAnnotationsFile = outPredictionsFolder + '/' + dataset + 'AnnotationPredictions.csv'
            annotationAllSubjPred.sort_values(by=['filepath']).to_csv(PredictedAnnotationsFile, index=False)
            # break
    ############################################################
    #EVALUATE PERFORMANCE  - Compare two annotation files
    print('EVALUATING PERFORMANCE')
    labelFreq=1/winParamsCNN.winStep
    TrueAnnotationsFile = outDir + '/' + dataset + 'CNNAnnotationsTrue.csv'
    PredictedAnnotationsFile = outPredictionsFolder + '/' + dataset + 'AnnotationPredictions.csv'
    # Calcualte performance per file by comparing true annotations file and the one created by ML training
    paramsPerformance = scoring.EventScoring.Parameters(
        toleranceStart=PerformanceParams.toleranceStart,
        toleranceEnd=PerformanceParams.toleranceEnd,
        minOverlap=PerformanceParams.minOveralp,
        maxEventDuration=PerformanceParams.maxEventDuration,
        minDurationBetweenEvents=PerformanceParams.minDurationBetweenEvents)
    # performancePerFile= evaluate2AnnotationFiles(TrueAnnotationsFile, PredictedAnnotationsFile, labelFreq)
    performancePerFile= evaluate2AnnotationFiles(TrueAnnotationsFile, PredictedAnnotationsFile, [], labelFreq, paramsPerformance)
    # save performance per file
    PerformancePerFileName = outPredictionsFolder + '/' + dataset + 'PerformancePerFile.csv'
    performancePerFile.sort_values(by=['filepath']).to_csv(PerformancePerFileName, index=False)

    # Calculate performance per subject
    GeneralParamsCNN.patients = [ f.name for f in os.scandir(outDir) if f.is_dir() ]
    GeneralParamsCNN.patients.sort() #Sorting them
    PerformancePerFileName = outPredictionsFolder + '/' + dataset + 'PerformancePerFile.csv'
    performacePerSubj= recalculatePerfPerSubject(PerformancePerFileName, GeneralParamsCNN.patients, labelFreq, paramsPerformance)
    PerformancePerSubjName = outPredictionsFolder + '/' + dataset + 'PerformancePerSubj.csv'
    performacePerSubj.sort_values(by=['subject']).to_csv(PerformancePerSubjName, index=False)
    # plot performance per subject
    plotPerformancePerSubj(GeneralParamsCNN.patients, performacePerSubj, outPredictionsFolder)

    print("GeneralParamsCNN.patients=",GeneralParamsCNN.patients)
    ### PLOT IN TIME
    for patIndx, pat in enumerate(GeneralParamsCNN.patients):
        print(pat)
        InName = outPredictionsFolder + '/Subj' + pat + '_CNN_TestPredictions.csv.parquet.gzip'
        data= readDataFromFile(InName)

        # visualize predictions
        outName = outPredictionsFolder + '/' + pat + '_PredictionsInTime2'
        plotPredictionsMatchingInTime(data['TrueLabels'].to_numpy(), data['PredLabels'].to_numpy(), data['PredLabels_MovAvrg'].to_numpy(), data['PredLabels_Bayes'].to_numpy(), outName, PerformanceParams)

        y_true = data['TrueLabels'].values
        y_scores = data['ProbabLabels'].values

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)

        auc = roc_auc_score(y_true, y_scores)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        outName_ROC = outPredictionsFolder + pat + '_PredictionsInTimeROC'
        plt.savefig(outName_ROC)
#debug        
# trainCNNKfolder()