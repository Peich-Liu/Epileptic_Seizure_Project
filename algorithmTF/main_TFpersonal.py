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
from sklearn.metrics import roc_curve, roc_auc_score
# from parametersSetupTF import PandasTSData
from loss import MaskedMSELoss, NoFussCrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
# # # #####################################################
# # # # SIENA DATASET
# dataset='SIENA'
# rootDir=  '../../../../../scratch/dan/physionet.org/files/siena-scalp-eeg/1.0.0' #when running from putty
# rootDir=  '../../../../../shares/eslfiler1/scratch/dan/physionet.org/files/siena-scalp-eeg/1.0.0' #when running from remote desktop
# DatasetPreprocessParamsTF.channelNamesToKeep=DatasetPreprocessParamsTF.channelNamesToKeep_Unipolar

# # DatasetPreprocessParamsCNN.channelNamesToKeep=DatasetPreprocessParamsCNN.channelNamesToKeep_Unipolar
# # # # SEIZIT DATASET
# dataset='SeizIT1'
# rootDir=  '../../../../../databases/medical/ku-leuven/SeizeIT1/v1_0' #when running from putty
# rootDir=  '../../../../../shares/eslfiler1/databases/medical/ku-leuven/SeizeIT1/v1_0' #when running from remote desktop
# DatasetPreprocessParamsTF.channelNamesToKeep=DatasetPreprocessParamsTF.channelNamesToKeep_Unipolar

# CHBMIT DATASET
dataset='CHBMIT'
rootDir=  '../../../../../scratch/dan/physionet.org/files/chbmit/1.0.0' #when running from putty
rootDir=  '../../../../../shares/eslfiler1/scratch/dan/physionet.org/files/chbmit/1.0.0' #when running from remote desktop
DatasetPreprocessParamsTF.channelNamesToKeep=DatasetPreprocessParamsTF.channelNamesToKeep_Bipolar
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
    outPredictionsFolder = '/home/pliu/git_repo/10_datasets/' + dataset + 'personal_TrainingResults' +'_Transformer' +'_'+'/01_personal_Transformer' + '_WinStep[' + str(
        winParamsTF.winLen) + ',' + str(winParamsTF.winStep) + ']'+ '/'
else:
    outDirFeatures= '/home/pliu/git_repo/10_datasets/'+ dataset+ '_Features_'+DatasetPreprocessParamsTF.eegDataNormalization+'/'
    outPredictionsFolder = '/home/pliu/git_repo/10_datasets/' + dataset + 'new_TrainingResults_' + DatasetPreprocessParamsTF.eegDataNormalization +'_'+ '/01_General_TF' + '_WinStep[' + str(
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
# GeneralParamsTF.patients=GeneralParamsTF.patients[:]
# dataAllSubj= loadAllSubjData(dataset, outDirFeatures, GeneralParamsTF.patients, FeaturesParamsTF.featNames,DatasetPreprocessParamsTF.channelNamesToKeep, TrueAnnotationsFile)

# #############################################
# # # TRAIN GENERAL

print('TRAINING') # run leave-one-subject-out CV
NonFeatureColumns= ['Subject', 'FileName', 'Time', 'Labels']
AllRes_test=np.zeros((len(GeneralParamsTF.patients),27))

for patIndx, pat in enumerate(GeneralParamsTF.patients):
    print(GeneralParamsTF.patients)
    print('******')
    test_patient = pat
    print(test_patient)
    print('-------')
    #PARAMETER SETUP
    n_classes = 2
    batch_size = 16
    n_channel = len(DatasetPreprocessParamsTF.channelNamesToKeep)
    # FOLDER SETUP
    folderDf = annotationsTrue
    folder_name = f"run_personal_{test_patient}"
    folder_path = os.path.join(model_store, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    trainPatients = [p for p in GeneralParamsTF.patients if p == test_patient]
    #GENERATE LABEL
    # FOLDER SETUP
    trainFolders = [os.path.join(outDir, p) for p in trainPatients]
    trainLabels = folderDf[folderDf['subject'] != test_patient ]
    testFolder = [os.path.join(outDir, test_patient)]
    testLabels = annotationsTrue[annotationsTrue['subject'] == test_patient]
    print("testFolder",testFolder)
    #####DATA SPILT
    print("testFolder=",testFolder)
    print("trainFolders=",trainFolders)
    label_df = annotationsTrue

    testSet = EEGDatasetTestTF(outDir,testFolder, testLabels, DatasetPreprocessParamsTF.samplFreq, winParamsTF.winLen, winParamsTF.winStep)
    trainSet = EEGDataset(outDir,trainFolders, trainLabels, DatasetPreprocessParamsTF.samplFreq, winParamsTF.winLen, winParamsTF.winStep)
    test_data_pred = []
    info_path = 'info_' + test_patient + '.csv'
    for i in range(len(testSet)):
        _, label, additional_info,_ = testSet[i]
        row = {
            'Subject': additional_info['subject'],
            'FileName': additional_info['filepath'],
            'Time': additional_info['startTime'],
            'Labels': label.item()  
        }
        test_data_pred.append(row)
    test_data_df = pd.DataFrame(test_data_pred)
    test_data_df.to_csv(f'info_{test_patient}.csv',index=False)
    # test_data_df = read_csv(info_path)
    print("test_data_df=",test_data_df)
    train_loader = DataLoader(trainSet, batch_size=batch_size, shuffle=False, pin_memory=True,
                                collate_fn=lambda x: collate_unsuperv(x, max_len=None, mask_compensation=False, task=None, oversample=None))
    test_loader = DataLoader(testSet, batch_size=batch_size, shuffle=False, pin_memory=True,
                                collate_fn=lambda x: collate_superv(x, max_len=None))
    test_labels_for_visual = []
    for data,labels,_,_ in test_loader:
        test_labels_for_visual.extend(labels.numpy())
    # test_labels_for_visual = np.array(test_labels_for_visual)
    # test_labels_for_visual = test_labels_for_visual.reshape(-1, 1)
    # for data, labels in train_loader:
    #     data, labels = data.to(device), labels.to(device)
    # for data, labels in test_loader:
    #     data, labels = data.to(device), labels.to(device)

    feat_dim = winParamsTF.winLen * DatasetPreprocessParamsTF.samplFreq
    # feat_dim = len(DatasetPreprocessParamsTF.channelNamesToKeep_Unipolar)
    TF_model = TSTransformerEncoder(feat_dim=n_channel, max_len=1024,d_model=512,n_heads=4,num_layers=4,dim_feedforward=2048)
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
    all_epoch_metrics = []
    for epoch in range(10):
        epoch_metrics = runner.train_epoch(epoch_num=epoch,outputDir=folder_path)
        print(f"Epoch {epoch} metrics: {epoch_metrics}")
        all_epoch_metrics.append(epoch_metrics)

    df = pd.DataFrame(all_epoch_metrics)
    csv_file_path = os.path.join(folder_path, 'all_epoch_metrics.csv')
    df.to_csv(csv_file_path, index=False)
    print(epoch_metrics['loss'])
    TF_model.load_state_dict(torch.load(folder_path+'/model_epoch_9.pth'))
    # TF_model.load_state_dict(torch.load('/home/pliu/git_repo/Epileptic_Seizure_Project/algorithmTF/model_store/run_PN00/model_epoch_8.pth'),map_location=torch.device('cpu'))
    test_evaluator = AnomalyRunner(TF_model, test_loader, device, loss_module, feat_dim, 
                                    output_dir='/home/pliu/git_repo/Epileptic_Seizure_Project/algorithmTF')
    
    aggr_metrics_val,predLabels_test, probabLab_test = validate(test_evaluator, tensorboard_writer, None,
                                                        epoch=200)
    
    print("predLabels_test=",predLabels_test,"probabLab_test",probabLab_test,"acc_test")
    # print()
    # measure performance
    AllRes_test[patIndx, 0:9] = performance_sampleAndEventBased(predLabels_test, test_labels_for_visual, PerformanceParamsTF)
    # test smoothing - movtest_patient=ng average
    predLabels_MovAvrg = movingAvrgSmoothing(predLabels_test, PerformanceParamsTF.smoothingWinLen,  PerformanceParamsTF.votingPercentage)
    AllRes_test[patIndx, 9:18] = performance_sampleAndEventBased(predLabels_MovAvrg, test_labels_for_visual, PerformanceParamsTF)
    # test smoothing - moving average
    predLabels_Bayes = smoothenLabels_Bayes(predLabels_test, probabLab_test, PerformanceParamsTF.smoothingWinLen, PerformanceParamsTF.bayesProbThresh)
    AllRes_test[patIndx, 18:27] = performance_sampleAndEventBased(predLabels_Bayes, test_labels_for_visual, PerformanceParamsTF)
    outputName = outPredictionsFolder + '/AllSubj_PerformanceAllSmoothing_OldMetrics.csv'
    saveDataToFile(AllRes_test, outputName, 'csv')
    
    # test_labels_for_visual.csv('cnn_label.csv')
    #visualize predictions
    outName=outPredictionsFolder + '/'+ test_patient+'_PredictionsInTime'
    plotPredictionsMatchingInTime(test_labels_for_visual, predLabels_test, predLabels_MovAvrg, predLabels_Bayes, outName, PerformanceParamsTF)

    test_labels_for_visual = np.array(test_labels_for_visual)
    # test_labels_for_visual_reshaped = test_labels_for_visual.reshape(-1, 1)
    # print("test_labels_for_visual",test_labels_for_visual_reshaped)
    # predLabels = np.max(probabLab_test, axis=1)
    predLabels = probabLab_test
    # predLabels_test = np.array(predLabels_test).reshape(-1, 1)
    # predLabels_MovAvrg = np.array(predLabels_MovAvrg).reshape(-1, 1)
    # predLabels_Bayes = np.array(predLabels_Bayes).reshape(-1, 1)
    print("test_label_for",test_labels_for_visual.shape)
    print("predLabels",predLabels.shape)
    # print("test_label_for_shape",len(test_labels_for_visual))
    print("probabLab_test",predLabels.shape)
    print("predLabels_test",predLabels_test.shape)
    print("predLabels_MovAvrg",predLabels_MovAvrg.shape)


    # # Saving predicitions in time
    # #############
    # #need modify#
    # #############
    dataToSave = np.vstack((test_labels_for_visual, predLabels, predLabels_test, predLabels_MovAvrg,  predLabels_Bayes)).transpose()   # added from which file is specific part of test set
    dataToSaveDF=pd.DataFrame(dataToSave, columns=['TrueLabels', 'ProbabLabels', 'PredLabels', 'PredLabels_MovAvrg', 'PredLabels_Bayes'])
    outputName = outPredictionsFolder + '/Subj' + test_patient + '_'+'Transformer'+'_TestPredictions.csv'
    saveDataToFile(dataToSaveDF, outputName, 'parquet.gzip')
    save_path = outputName + 'parquet.gzip'
    # # CREATE ANNOTATION FILE
    predlabels= np.vstack((predLabels, predLabels_test, predLabels_MovAvrg,  predLabels_Bayes)).transpose().astype(int)
    testPredictionsDF=pd.concat([test_data_df, pd.DataFrame(predlabels, columns=['ProbabLabels', 'PredLabels', 'PredLabels_MovAvrg', 'PredLabels_Bayes'])] , axis=1)
    annotationsTrue=readDataFromFile(TrueAnnotationsFile)
    print(annotationsTrue)
    annotationAllPred=createAnnotationFileFromPredictions(testPredictionsDF, annotationsTrue, 'PredLabels_Bayes')
    if (patIndx==0):
        annotationAllSubjPred=annotationAllPred
    else:
        annotationAllSubjPred = pd.concat([annotationAllSubjPred, annotationAllPred], axis=0)
    #save every time, just for backup
    PredictedAnnotationsFile = outPredictionsFolder + '/' + dataset + 'AnnotationPredictions.csv'
    annotationAllSubjPred.sort_values(by=['filepath']).to_csv(PredictedAnnotationsFile, index=False)

    
############################################################
#EVALUATE PERFORMANCE  - Compare two annotation files

outPredictionsFolder = '/home/pliu/git_repo/10_datasets/good_SIENA_performancenew0.8_Transformer_TrainingResults'
for patIndx, pat in enumerate(GeneralParamsTF.patients):   
        result_file = outPredictionsFolder + '/Subj' + pat + '_TestPredictions.csv.parquet.gzip'
        # testData= dataAllSubj[dataAllSubj['Subject'] == pat]
        testData = test_data_df
        print("result_file",result_file)
        predlabels = pd.read_parquet(result_file)
        testPredictionsDF=pd.concat([testData[NonFeatureColumns].reset_index(drop=True), pd.DataFrame(predlabels, columns=['ProbabLabels', 'PredLabels', 'PredLabels_MovAvrg', 'PredLabels_Bayes'])] , axis=1)

        annotationsTrue = readDataFromFile(TrueAnnotationsFile)
        annotationAllPred=createAnnotationFileFromPredictions(testPredictionsDF, annotationsTrue, 'PredLabels_MovAvrg')
        if (patIndx==0):
            annotationAllSubjPred=annotationAllPred
        else:
            annotationAllSubjPred = pd.concat([annotationAllSubjPred, annotationAllPred], axis=0)
        #save every time, just for backup
        PredictedAnnotationsFileLoad = outPredictionsFolder + '/' + dataset + 'loadAnnotationPredictions.csv'
        # print(PredictedAnnotationsFile)
        # quit()
        annotationAllSubjPred.sort_values(by=['filepath']).to_csv(PredictedAnnotationsFileLoad, index=False)
print('EVALUATING PERFORMANCE')
labelFreq=1/winParamsTF.winStep
TrueAnnotationsFile = outDir + '/' + dataset + 'AnnotationsTrue.csv'
PredictedAnnotationsFile = outPredictionsFolder + '/' + dataset + 'AnnotationPredictions.csv'
# print()
# Calcualte performance per file by comparing true annotations file and the one created by ML training
paramsPerformance = scoring.EventScoring.Parameters(
    toleranceStart=PerformanceParamsTF.toleranceStart,
    toleranceEnd=PerformanceParamsTF.toleranceEnd,
    minOverlap=PerformanceParamsTF.minOveralp,
    maxEventDuration=PerformanceParamsTF.maxEventDuration,
    minDurationBetweenEvents=PerformanceParamsTF.minDurationBetweenEvents)
# performancePerFile= evaluate2AnnotationFiles(TrueAnnotationsFile, PredictedAnnotationsFile, labelFreq)
performancePerFile= evaluate2AnnotationFiles(TrueAnnotationsFile, PredictedAnnotationsFile, [], labelFreq, paramsPerformance)
# save performance per file
PerformancePerFileName = outPredictionsFolder + '/' + dataset + 'PerformancePerFile.csv'
performancePerFile.sort_values(by=['filepath']).to_csv(PerformancePerFileName, index=False)

# Calculate performance per subject
GeneralParamsTF.patients = [ f.name for f in os.scandir(outDir) if f.is_dir() ]
GeneralParamsTF.patients.sort() #Sorting them
PerformancePerFileName = outPredictionsFolder + '/' + dataset + 'PerformancePerFile.csv'
performacePerSubj= recalculatePerfPerSubject(PerformancePerFileName, GeneralParamsTF.patients, labelFreq, paramsPerformance)
PerformancePerSubjName = outPredictionsFolder + '/' + dataset + 'PerformancePerSubj.csv'
performacePerSubj.sort_values(by=['subject']).to_csv(PerformancePerSubjName, index=False)
# plot performance per subject
plotPerformancePerSubj(GeneralParamsTF.patients, performacePerSubj, outPredictionsFolder)


# ### PLOT IN TIME
for patIndx, pat in enumerate(GeneralParamsTF.patients):
    print(pat)
    InName = outPredictionsFolder + '/Subj' + pat + '_Transformer' + '_TestPredictions.csv.parquet.gzip'
    data= readDataFromFile(InName)

    # visualize predictions
    outName = outPredictionsFolder + '/' + pat + '_PredictionsInTimeLoad'
    plotPredictionsMatchingInTime(data['TrueLabels'].to_numpy(), data['PredLabels'].to_numpy(), data['PredLabels_MovAvrg'].to_numpy(), data['PredLabels_Bayes'].to_numpy(), outName, PerformanceParamsTF)



# ### FIND OPTIMAL PROCESSING PARAMETERS FOR ALL SUBJ TOGETHER
# # load all predictions in time
# TestDifferentPostprocessingParams(outPredictionsFolder, dataset, GeneralParams, StandardMLParams)

print('EVALUATING PERFORMANCE')
labelFreq=1/winParamsTF.winStep
TrueAnnotationsFile = outDir + '/' + dataset + 'AnnotationsTrue.csv'
PredictedAnnotationsFile = outPredictionsFolder + '/' + dataset + 'AnnotationPredictions.csv'
# Calcualte performance per file by comparing true annotations file and the one created by ML training
paramsPerformance = scoring.EventScoring.Parameters(
toleranceStart=PerformanceParamsTF.toleranceStart,
toleranceEnd=PerformanceParamsTF.toleranceEnd,
minOverlap=PerformanceParamsTF.minOveralp,
maxEventDuration=PerformanceParamsTF.maxEventDuration,
minDurationBetweenEvents=PerformanceParamsTF.minDurationBetweenEvents)
# performancePerFile= evaluate2AnnotationFiles(TrueAnnotationsFile, PredictedAnnotationsFile, labelFreq)
performancePerFile= evaluate2AnnotationFiles(TrueAnnotationsFile, PredictedAnnotationsFile, [], labelFreq, paramsPerformance)
# save performance per filex
PerformancePerFileName = outPredictionsFolder + '/' + dataset + 'PerformancePerFile.csv'
performancePerFile.sort_values(by=['filepath']).to_csv(PerformancePerFileName, index=False)

# Calculate performance per subject
GeneralParamsTF.patients = [ f.name for f in os.scandir(outDir) if f.is_dir() ]
GeneralParamsTF.patients.sort() #Sorting them
PerformancePerFileName = outPredictionsFolder + '/' + dataset + 'PerformancePerFile.csv'
performacePerSubj= recalculatePerfPerSubject(PerformancePerFileName, GeneralParamsTF.patients, labelFreq, paramsPerformance)
PerformancePerSubjName = outPredictionsFolder + '/' + dataset + 'PerformancePerSubj.csv'
performacePerSubj.sort_values(by=['subject']).to_csv(PerformancePerSubjName, index=False)
# plot performance per subject
plotPerformancePerSubj(GeneralParamsTF.patients, performacePerSubj, outPredictionsFolder)

print("GeneralParamsTF.patients=",GeneralParamsTF.patients)
### PLOT IN TIME
for patIndx, pat in enumerate(GeneralParamsTF.patients):
    InName = outPredictionsFolder + '/Subj' + pat + '_Transformer_TestPredictions.csv.parquet.gzip'
    data= readDataFromFile(InName)

    # visualize predictions
    outName = outPredictionsFolder + '/' + pat + '_PredictionsInTime2'
    plotPredictionsMatchingInTime(data['TrueLabels'].to_numpy(), data['PredLabels'].to_numpy(), data['PredLabels_MovAvrg'].to_numpy(), data['PredLabels_Bayes'].to_numpy(), outName, PerformanceParamsTF)

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



