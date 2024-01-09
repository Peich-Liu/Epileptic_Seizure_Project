# Epileptic_Seizure_Project
## Installation

use 
```bash
https://github.com/Peich-Liu/Epileptic_Seizure_Project.git
```
to clone the code to the computer

### Conda

Create a conda env
```bash
conda create --name 'your_conda_name' python=3.11
```

install requirements with:

```bash
python -m pip install -r requirements.txt
```

Before each run, activate the environment with:

```bash
conda activate 'your_conda_name'
```

## Usage
### Running
it is an example of the CHBMIT dataset, RusBoost algorithm, and Leave-one-out validation
```bash
python main.py --algorithm RusBoost  --dataset CHBMIT --trainType general
```
Available List:
--dataset: CHBMIT, SIENA, SeizIT1
--algorithm: RusBoost, CNN, Transformer, CNNLight
--trainType: general, personal, Kfolder

notice: not every algorithm has all of the validation methods, you can only use the validation method we mentioned in the report. The available validation methods are:

![image text](https://github.com/Peich-Liu/Epileptic_Seizure_Project/blob/main/available_method.png)



