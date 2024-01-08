from algorithmRusBoost.parametersSetupRUS import *
from algorithmCnn.parametersSetupCNN import *
from algorithmTF.parametersSetupTF import *
from algorithmLight.parametersSetupCNNLight import *

class parametersUpdate():
    def __init__(self, params):
        self.params = params
    
    def setup(self, algorithmType):
        print(algorithmType)
        if algorithmType == 'RusBoost':
            DatasetPreprocessParams.updateDatasetPreprocessParams(self.params)
            FeaturesParams.updateFeaturesParams(self.params)
            
        elif algorithmType == 'CNN':
            DatasetPreprocessParamsCNN.updateDatasetPreprocessParams(self.params)
            winParamsCNN.updateWinParamsCNN(self.params)
            
        elif algorithmType == 'CNNLight':
            DatasetPreprocessParamsCNNLight.updateDatasetPreprocessParams(self.params)
            winParamsCNNLight.updateWinParamsCNN(self.params)
            
        elif algorithmType == 'Transformer':
            DatasetPreprocessParamsCNNLight.updateDatasetPreprocessParams(self.params)
            winParamsTF.updateWinParamsTF(self.params)
            