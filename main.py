# import sys
# sys.path.append(r'../../Epileptic_Seizure_Project/algorithmRusBoost')
import argparse
from parametersUpdate import parametersUpdate
import yaml
from algorithmRusBoost import main_KfoldGeneral, main_general, main_personal
from algorithmCnn import main_CNNKfoler


def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_training_function(train_type, algorithm):
    if train_type == "Kfolder" and algorithm == "RusBoost":
        print("get_training_function")
        return main_KfoldGeneral.trainRusKfolder
    elif train_type == "personal" and algorithm == "RusBoost":
        return main_personal.trainRusPersonal
    elif train_type == "general" and algorithm == "RusBoost":
        return main_general.trainRusGeneral
    elif train_type == "Kfolder" and algorithm == "CNN":
        return main_CNNKfoler.trainCNNKfolder
    elif train_type == "personal" and algorithm == "CNN":
        raise ValueError("Data too large, Only for Kfolder")
    elif train_type == "general" and algorithm == "CNN":
        raise ValueError("Data too large, Only for Kfolder") 
    elif train_type == "Kfolder" and algorithm == "Transformer":
        return 
    elif train_type == "personal" and algorithm == "Transformer":
        return 
    elif train_type == "general" and algorithm == "Transformer":
        return 

    else:
        raise ValueError("Unknown training type")

def main():
    # command

    parser = argparse.ArgumentParser(description="framework of seizure detection")
    
    default_Unipolar = ('Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6')
    default_Bipolar = ('Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fz-Cz', 'Cz-Pz', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2')
    parser.add_argument("--algorithm", type=str,default='RusBoost', choices=['RusBoost', 'CNN','Transformer'], help="algorithm name")
    parser.add_argument("--dataset", type=str,default='SIENA', choices=['SIENA', 'CHBMIT','SeizIT1'], help="dataset name")
    parser.add_argument("--trainType", type=str,default='Kfolder', choices=['Kfolder', 'general','personal'], help="different algorithm has different requirement, can check in the conf files")
    
    args = parser.parse_args()
    algorithm = args.algorithm
    configPath = 'conf/'+algorithm + '.yaml'
    config = load_yaml_config(configPath)
    # print("algorithm",args.algorithm,"trainType",args.trainType)
    paramUpdate = parametersUpdate({'Unipolar':config['DatasetPreprocessParams']['Unipolar'],
                                    'Bipolar':config['DatasetPreprocessParams']['Bipolar'],
                                    'refElectrode':config['DatasetPreprocessParams']['refElectrode'],
                                    'samplFreq':config['DatasetPreprocessParams']['samplFreq'],
                                    'eegDataNormalization':config['DatasetPreprocessParams']['eegDataNormalization'],
                                    'winLen':config['FeaturesParams']['winLen'],
                                    'winStep':config['FeaturesParams']['winStep'],
                                    'featNorm':config['FeaturesParams']['featNorm'],
                                    'dataset':args.dataset
                                    })

    paramUpdate.setup(args.algorithm)
    trainFunction = get_training_function(args.trainType, args.algorithm)
    trainFunction()

if __name__ == "__main__":
    main()
