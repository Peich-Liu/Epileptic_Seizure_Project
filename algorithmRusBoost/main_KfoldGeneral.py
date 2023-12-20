import sys
# sys.path.append(r'../../Epileptic_Seizure_Project')
sys.path.append(r'../../Epileptic_Seizure_Project/algorithmRusBoost')
from loadEeg.loadEdf import *
from algorithmRusBoost.parametersSetupRUS import *
from VariousFunctionsLib import  *
from evaluate.evaluate import *
import os
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def trainRusKfolder():
    #####################################################
    #Dataset Setting
    dataset = DatasetPreprocessParams.dataset
    if dataset == 'SIENA':
        ##SIENA DATASET
        rootDir=  '../../../../../scratch/dan/physionet.org/files/siena-scalp-eeg/1.0.0' #when running from putty
        rootDir=  '../../../../../shares/eslfiler1/scratch/dan/physionet.org/files/siena-scalp-eeg/1.0.0' #when running from remote desktop
        DatasetPreprocessParams.channelNamesToKeep=DatasetPreprocessParams.channelNamesToKeep_Unipolar
    elif dataset == 'SeizIT1':
        # SEIZIT DATASET
        rootDir=  '../../../../../databases/medical/ku-leuven/SeizeIT1/v1_0' #when running from putty
        rootDir=  '../../../../../shares/eslfiler1/databases/medical/ku-leuven/SeizeIT1/v1_0' #when running from remote desktop
        DatasetPreprocessParams.channelNamesToKeep=DatasetPreprocessParams.channelNamesToKeep_Bipolar
    elif dataset == 'CHBMIT':
        # CHBMIT DATASET
        rootDir=  '../../../../../scratch/dan/physionet.org/files/chbmit/1.0.0' #when running from putty
        rootDir=  '../../../../../shares/eslfiler1/scratch/dan/physionet.org/files/chbmit/1.0.0' #when running from remote desktop
        DatasetPreprocessParams.channelNamesToKeep=DatasetPreprocessParams.channelNamesToKeep_Bipolar
    #####################################################
    # SET DIFFERENT PARAMETERS
    # Set features to use (it will be in the ouput folder name)
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
        outDirFeatures = '/home/pliu/git_repo/10_datasets/' + dataset + '_multi_Features/'
        outPredictionsFolder = '/home/pliu/git_repo/10_datasets/' + dataset + '_multi_TrainingResults' +'_'+StandardMLParams.trainingDataResampling +'_'+ str(StandardMLParams.traininDataResamplingRatio)+'/01_GeneralKfold_' + StandardMLParams.modelType + '_WinStep[' + str(
            FeaturesParams.winLen) + ',' + str(FeaturesParams.winStep) + ']_' + '-'.join(
            FeaturesParams.featNames) + appendix+ '/'
    else:
        outDirFeatures= '/home/pliu/git_repo/10_datasets/'+ dataset+ '_multi_Features'+DatasetPreprocessParams.eegDataNormalization+'/'
        outPredictionsFolder = '/home/pliu/git_repo/10_datasets/' + dataset + '_multi_TrainingResults_' + DatasetPreprocessParams.eegDataNormalization +'_'+StandardMLParams.trainingDataResampling+'_'+ str(StandardMLParams.traininDataResamplingRatio)+ '/01_GeneralKfold_' + StandardMLParams.modelType + '_WinStep[' + str(
            FeaturesParams.winLen) + ',' + str(FeaturesParams.winStep) + ']_' + '-'.join(
            FeaturesParams.featNames) + appendix+ '/'
    os.makedirs(os.path.dirname(outDirFeatures), exist_ok=True)
    os.makedirs(os.path.dirname(outPredictionsFolder), exist_ok=True)
    # testing that folders are correct
    # print(os.path.exists(rootDir))
    # print(os.listdir('../../../../../'))
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

    # # #####################################################
    # # EXTRACT ANNOTATIONS - Only has to be done once

    if (dataset=='CHBMIT'):
        from loadAnnotations.CHBMITAnnotationConverter import convertAllAnnotations, checkIfRawDataExists
    elif (dataset == 'SIENA'):
        from loadAnnotations.sienaAnnotationConverter import convertAllAnnotations, checkIfRawDataExists
    elif (dataset=='SeizIT1'):
        from loadAnnotations.seizeitAnnotationConverter import convertAllAnnotations, checkIfRawDataExists
    TrueAnnotationsFile = outDir + '/' + dataset + 'AnnotationsTrue.csv'
    os.makedirs(os.path.dirname(TrueAnnotationsFile), exist_ok=True)
    annotationsTrue= convertAllAnnotations(rootDir, TrueAnnotationsFile )#234
    # annotationsTrue=annotationsTrue.sort_values(by=['subject', 'session'])
    # check if all files in annotationsTrue actually exist in standardized dataset
    # (if there were problems with files they might have been excluded, so exclude those files)
    # TrueAnnotationsFile = outDir + '/' + dataset + 'AnnotationsTrue.csv'
    # annotationsTrue=pd.read_csv(TrueAnnotationsFile)
    annotationsTrue= checkIfRawDataExists(annotationsTrue, outDir)#234
    annotationsTrue.to_csv(TrueAnnotationsFile, index=False)
    TrueAnnotationsFile = outDir + '/' + dataset + 'AnnotationsTrue.csv'
    annotationsTrue=pd.read_csv(TrueAnnotationsFile)

    #load annotations - if we are not extracting them above
    TrueAnnotationsFile = outDir + '/' + dataset + 'AnnotationsTrue.csv'
    annotationsTrue=pd.read_csv(TrueAnnotationsFile)
    # print("TrueAnnotationsFile=",TrueAnnotationsFile)
    # #####################################################
    # # EXTRACT FEATURES AND SAVE TO FILES - Only has to be done once
    # calculateFeaturesForAllFiles(outDir, outDirFeatures, DatasetPreprocessParams, FeaturesParams, DatasetPreprocessParams.eegDataNormalization, outFormat ='parquet.gzip' )

    # # # CALCULATE KL DIVERGENCE OF FEATURES
    # GeneralParams.patients = [ f.name for f in os.scandir(outDir) if f.is_dir() ]
    # GeneralParams.patients.sort() #Sorting them
    # FeaturesParams.allFeatNames = constructAllfeatNames(FeaturesParams)
    # calculateKLDivergenceForFeatures(dataset, GeneralParams.patients , outDirFeatures, TrueAnnotationsFile, FeaturesParams)

    # # # # ####################################################
    # # TRAIN GENERALIZED MODEL
    #
    ## LOAD ALL DATA OF ALL SUBJECTS
    print('LOADING ALL DATA')
    # Create list of all subjects
    GeneralParams.patients = [ f.name for f in os.scandir(outDir) if f.is_dir() ]
    GeneralParams.patients.sort() #Sorting them
    GeneralParams.patients=GeneralParams.patients[0:5]
    print("outDirFeatures",outDirFeatures)
    print("featNames",FeaturesParams.featNames)
    print("dataset",dataset)
    print("TrueAnnotationsFile",TrueAnnotationsFile)
    print("GeneralParams.patients",GeneralParams.patients)
    print("DatasetPreprocessParams.channelNamesToKeep",DatasetPreprocessParams.channelNamesToKeep)
    dataAllSubj= loadAllSubjData(dataset, outDirFeatures, GeneralParams.patients, FeaturesParams.featNames,DatasetPreprocessParams.channelNamesToKeep, TrueAnnotationsFile)
    print(dataAllSubj)

    ##################################
    print('TRAINING') # run leave-one-subject-out CV
    NonFeatureColumns= ['Subject', 'FileName', 'Time', 'Labels']
    AllRes_test=np.zeros((len(GeneralParams.patients),27))

    NsubjPerK=int(np.ceil(len(GeneralParams.patients)/GeneralParams.GenCV_numFolds))
    for kIndx in range(GeneralParams.GenCV_numFolds):
        patientsToTest=GeneralParams.patients[kIndx*NsubjPerK:(kIndx+1)*NsubjPerK]
        print('******')
        print(patientsToTest)
        print('-------')

        # trainData = dataAllSubj[dataAllSubj['Subject'] not in patientsToTest]
        trainData = dataAllSubj[~dataAllSubj['Subject'].isin(patientsToTest) ]
        # print(trainData)
        # print('-------')
        for p in range(NsubjPerK):
            patIndx=kIndx*NsubjPerK+p
            pat=GeneralParams.patients[patIndx]
            print(pat)
            testData= dataAllSubj[dataAllSubj['Subject'] == pat]
            print(testData['Labels'].to_numpy())
            # quit()
            testDataFeatures= testData.loc[:, ~testData.columns.isin(NonFeatureColumns)]
            trainDataFeatures = trainData.loc[:, ~trainData.columns.isin(NonFeatureColumns)]

            #normalize data
            trainDataFeatures = trainDataFeatures.loc[:,~trainDataFeatures.columns.duplicated()]
            testDataFeatures = testDataFeatures.loc[:,~testDataFeatures.columns.duplicated()]
            if (FeaturesParams.featNorm == 'Norm'):
                # testDataFeatures= normalizeData(testDataFeatures)
                # trainDataFeatures = normalizeData(trainDataFeatures)
                (trainDataFeatures, testDataFeatures) = normalizeTrainAndTestData(trainDataFeatures, testDataFeatures)
                trainDataFeatures=removeExtremeValues(trainDataFeatures)
                testDataFeatures=removeExtremeValues(testDataFeatures)
                #remove useless feature columns
                colsToDrop=[]
                colsToDrop=removeFeaturesIfExtreme(trainDataFeatures, colsToDrop)
                colsToDrop=removeFeaturesIfExtreme(testDataFeatures, colsToDrop)
                colsToDrop=list(set(colsToDrop))
                trainDataFeatures=trainDataFeatures.drop(labels=colsToDrop, axis='columns')
                testDataFeatures=testDataFeatures.drop(labels=colsToDrop, axis='columns')

            ## STANDARD ML LEARNING
            if (StandardMLParams.trainingDataResampling != 'NoResampling'):
                (Xtrain, ytrain) = datasetResample(trainDataFeatures.to_numpy(), trainData['Labels'].to_numpy(),
                                                StandardMLParams.trainingDataResampling,
                                                StandardMLParams.traininDataResamplingRatio, randState=42)
            else:
                Xtrain = trainDataFeatures.to_numpy()
                ytrain = trainData['Labels'].to_numpy()

            MLstdModel = train_StandardML_moreModelsPossible(Xtrain, ytrain, StandardMLParams)
            # MLstdModel = train_StandardML_moreModelsPossible(testDataFeatures.to_numpy(), testData['Labels'].to_numpy(), StandardMLParams)
            # testing
            (predLabels_test, probabLab_test, acc_test, accPerClass_test) = test_StandardML_moreModelsPossible(testDataFeatures.to_numpy(), testData['Labels'].to_numpy(),MLstdModel)

            # measure performance
            AllRes_test[patIndx, 0:9] = performance_sampleAndEventBased(predLabels_test, testData['Labels'].to_numpy(), PerformanceParams)
            # test smoothing - moving average
            predLabels_MovAvrg = movingAvrgSmoothing(predLabels_test, PerformanceParams.smoothingWinLen,  PerformanceParams.votingPercentage)
            AllRes_test[patIndx, 9:18] = performance_sampleAndEventBased(predLabels_MovAvrg, testData['Labels'].to_numpy(), PerformanceParams)
            # test smoothing - moving average
            predLabels_Bayes = smoothenLabels_Bayes(predLabels_test, probabLab_test, PerformanceParams.smoothingWinLen, PerformanceParams.bayesProbThresh)
            AllRes_test[patIndx, 18:27] = performance_sampleAndEventBased(predLabels_Bayes, testData['Labels'].to_numpy(), PerformanceParams)
            outputName = outPredictionsFolder + '/AllSubj_PerformanceAllSmoothing_OldMetrics.csv'
            saveDataToFile(AllRes_test, outputName, 'csv')
            
            #visualize predictions
            print("testData['Labels'].to_numpy()",testData['Labels'].to_numpy())
            testData['Labels'].to_csv('temp.csv')
            outName=outPredictionsFolder + '/'+ pat+'_PredictionsInTime'
            plotPredictionsMatchingInTime(testData['Labels'].to_numpy(), predLabels_test, predLabels_MovAvrg, predLabels_Bayes, outName, PerformanceParams)
        
            print("probabLab_test",probabLab_test)
            # Saving predicitions in time
            dataToSave = np.vstack((testData['Labels'].to_numpy(), probabLab_test, predLabels_test, predLabels_MovAvrg,  predLabels_Bayes)).transpose()   # added from which file is specific part of test set
            dataToSaveDF=pd.DataFrame(dataToSave, columns=['TrueLabels', 'ProbabLabels', 'PredLabels', 'PredLabels_MovAvrg', 'PredLabels_Bayes'])
            outputName = outPredictionsFolder + '/Subj' + pat + '_'+StandardMLParams.modelType+'_TestPredictions.csv'
            saveDataToFile(dataToSaveDF, outputName, 'parquet.gzip')

            # CREATE ANNOTATION FILE
            predlabels= np.vstack((probabLab_test, predLabels_test, predLabels_MovAvrg,  predLabels_Bayes)).transpose().astype(int)
            testPredictionsDF=pd.concat([testData[NonFeatureColumns].reset_index(drop=True), pd.DataFrame(predlabels, columns=['ProbabLabels', 'PredLabels', 'PredLabels_MovAvrg', 'PredLabels_Bayes'])] , axis=1)
            # print("testData[NonFeatureColumns].reset_index(drop=True)",testData[NonFeatureColumns].reset_index(drop=True))
            annotationsTrue=readDataFromFile(TrueAnnotationsFile)
            annotationAllPred=createAnnotationFileFromPredictions(testPredictionsDF, annotationsTrue, 'PredLabels_MovAvrg')
            if (patIndx==0):
                annotationAllSubjPred=annotationAllPred
            else:
                annotationAllSubjPred = pd.concat([annotationAllSubjPred, annotationAllPred], axis=0)
            #save every time, just for backup
            PredictedAnnotationsFile = outPredictionsFolder + '/' + dataset + 'AnnotationPredictions.csv'
            annotationAllSubjPred.sort_values(by=['filepath']).to_csv(PredictedAnnotationsFile, index=False)


    #############################################################
    #EVALUATE PERFORMANCE  - Compare two annotation files
    print('EVALUATING PERFORMANCE')
    labelFreq=1/FeaturesParams.winStep
    TrueAnnotationsFile = outDir + '/' + dataset + 'AnnotationsTrue.csv'
    PredictedAnnotationsFile = outPredictionsFolder + '/' + dataset + 'AnnotationPredictions.csv'
    #temp
    # PredictedAnnotationsFile = '/home/pliu/01_General_RUSboost_WinStep[0.391,0.391]_withNetwork_0.95th/SIENAAnnotationPredictions.csv'

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
    GeneralParams.patients = [ f.name for f in os.scandir(outDir) if f.is_dir() ]
    GeneralParams.patients.sort() #Sorting them
    PerformancePerFileName = outPredictionsFolder + '/' + dataset + 'PerformancePerFile.csv'
    performacePerSubj= recalculatePerfPerSubject(PerformancePerFileName, GeneralParams.patients, labelFreq, paramsPerformance)
    PerformancePerSubjName = outPredictionsFolder + '/' + dataset + 'PerformancePerSubj.csv'
    performacePerSubj.sort_values(by=['subject']).to_csv(PerformancePerSubjName, index=False)
    # plot performance per subject
    plotPerformancePerSubj(GeneralParams.patients, performacePerSubj, outPredictionsFolder)


    ### PLOT IN TIME
    for patIndx, pat in enumerate(GeneralParams.patients):
        print(pat)
        InName = outPredictionsFolder + 'Subj' + pat + '_' + StandardMLParams.modelType + '_TestPredictions.csv.parquet.gzip'
        data= readDataFromFile(InName)

        # visualize predictions
        outName = outPredictionsFolder + '/' + pat + '_PredictionsInTime'
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


    # ### FIND OPTIMAL PROCESSING PARAMETERS FOR ALL SUBJ TOGETHER
    # # load all predictions in time
    # TestDifferentPostprocessingParams(outPredictionsFolder, dataset, GeneralParams, StandardMLParams)
