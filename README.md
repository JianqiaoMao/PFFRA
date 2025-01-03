# PFFRA: Permutation Feature-based Frequency Response Analysis [![DOI](https://sandbox.zenodo.org/badge/654937709.svg)](https://sandbox.zenodo.org/doi/10.5072/zenodo.24051)
PFFRA: An Interpretable Machine Learning technique to analyse the contribution of features in the frequency domain. This method is inspired by permutation feature importance analysis but aims to quantify and analyse the time-series predictive model's mechanism from a global perspective.

### Citing

Please use one of the following to cite the code of this repository.

```
@article{doi:10.1080/23744731.2023.2239081,
author = {Jianqiao Mao, Ryan Grammenos and Konstantinos Karagiannis},
title = {Data analysis and interpretable machine learning for HVAC predictive control: A case-study based implementation},
journal = {Science and Technology for the Built Environment},
volume = {29},
number = {7},
pages = {698-718},
year = {2023},
publisher = {Taylor & Francis},
doi = {10.1080/23744731.2023.2239081},
URL = { 
        https://doi.org/10.1080/23744731.2023.2239081
},
eprint = { 
        https://doi.org/10.1080/23744731.2023.2239081
}
}
```

Or,

```
@article{mao2021interpreting,
  title={Interpreting machine learning models for room temperature prediction in non-domestic buildings},
  author={Mao, Jianqiao and Ryan, Grammenos},
  journal={arXiv preprint arXiv:2111.13760},
  year={2021}
```

## Installation and getting started

We currently offer seamless installation with `pip`. 

Simply:
```
pip install PFFRA
```

Alternatively, download the current distribution of the package, and run:
```
pip install .
```
in the root directory of the decompressed package.

## Background

### Permutation Feature Importance Analysis

It is important to understand and disclose the most important features that contribute to the predictions through a reasonable measuring scheme, because this knowledge can help machine learning engineers improve model transparency and reveal unwanted behaviours [1].

Permutation feature importance is another model-agnostic method originally applied in Random Forests (RF) [2]. This method measures the feature importance by measuring how much metrics score, e.g., Mean Square Error (MSE), drops when Out-of-Bag (OOB) dataset without a certain feature is used. In practice, the feature which is under-test is replaced with noises instead of directly removing, for example, a common approach to introduce the noise is shuffling the feature values [3][4].

### Permutation Feature-based Frequency Response Analysis

Existing IML techniques are not designed to explain time-series machine learning models. In our recent work [5], we propose a new global interpretation technique called Permutation Feature-based Frequency Response Analysis (**PF-FRA**) to investigate the contribution of interested features in the frequency domain. Briefly, **PF-FRA** compares frequency responses of the given model based on the permuted dataset. These generated spectrums enable the user to identify the interested features' contribution to different frequency ranges. In this case, the user may identify if a feature leads to short- or long-term trend in the time-series model's predictions.

To check more detail about the PF-FRA algorithm, please find it [here](https://www.tandfonline.com/doi/full/10.1080/23744731.2023.2239081).

## Algorithm

#### Algorithm: Permutation Feature-based Frequency Response Analysis (PF-FRA)
![pffra_algorithm](https://github.com/user-attachments/assets/eec8ff6b-e479-4d95-9284-eed84fbf6384)

## Example Demo.

For a simple example of using, please check the Jupyter Notebook [file](https://github.com/JianqiaoMao/PFFRA/blob/main/Tutorial.ipynb).

In this tutorial, we generate a synthetic dataset with 3 sinusoidal signals containing different frequency components (10, 37, 39Hz) as features. The target variable is the linear combination of the three components. And we train a XGBM regressor for a demonstration purpose, which means, by the end of the tutorial, we aim to reveal these features' contribution to the prediction series in frequency domain.

1. Import PFFRA lib and other required libs.

```python
import PFFRA
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
```

2. Generate synthetic data for regression task.

```python
t = np.linspace(0,10,1000)
x1 = np.sin(2 * np.pi * 10 * t)
x2 = np.sin(2 * np.pi * 37 * t)
x3 = np.cos(2* np.pi * 39 * t)

X = np.vstack((x1, x2, x3)).T
y = 5*x1+2*x2+3*x3
```

3. Train a XGBM model and generate predictions.

```python
model = XGBRegressor(n_estimators = 5)
model = model.fit(X, y)
y_hat = model.predict(X)

plt.figure(figsize = (8, 6))
plt.plot(t, y_hat)
plt.xlabel("time(t)")
plt.ylabel("y_hat")
plt.title("XGBM predicted time series")
```
![image](https://github.com/JianqiaoMao/PFFRA/assets/60654068/c0a384a3-f623-4079-a520-b565961ad342)


4. Apply PFFRA to draw spectrum figures directly for a given interested feature index using 'mean' mode.

```python
pffra = PFFRA.PermFeatureFreqRespoAnalysis(y = y, X = X, model = model, interested_feature_index = 1)
pffra.show_spectrum(sample_rate = 100, mode = 'mean', rename_feature = "x2=sin(2 *pi*37*t)")
```
![FT_demo](https://github.com/user-attachments/assets/3922800b-6d01-4cf3-a742-beed312ba0ac)



5. Investigate multiple features and output their permutation spectrum data using 'shuffle' mode (**New feature in v0.1.2**).

```python
fig = plt.figure(figsize = (14,6))
ax1 = fig.add_subplot(121)
for i in range(X.shape[1]):
    # instantiate PFFRA
    pffra = PFFRA.PermFeatureFreqRespoAnalysis(y = y, X = X, model = model, interested_feature_index = i)
    # Generate permutation dataset
    X_interested_feature, X_other_feature = pffra.permuted_dataset(mode = 'shuffle')
    # Predict target variables using the permutated datasets
    pred_interested_feature, pred_other_feature, pred_all_feature = pffra.permu_pred(X_interested_feature, X_other_feature)
    # Generate spectrums for analysis
    spectrums = pffra.gen_spectrum(pred_interested_feature = pred_interested_feature, 
                                   pred_other_feature = pred_other_feature, 
                                   pred_all_feature = pred_all_feature, 
                                   sample_rate = 100)
    # Unpack spectrums
    spectrum_interested_i = spectrums[0]
    spectrum_all = spectrums[2]
    spectrum_true = spectrums[3]
    frq_range = spectrums[4]
    
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16  
    plt.rcParams["xtick.labelsize"] = "large"
    plt.rcParams["ytick.labelsize"] = "large"
    ax1.plot(frq_range[1:], spectrum_interested_i[1:], 
             label = "x_{} (DC: {:.2f})".format(i, spectrum_interested_i[0]))
    ax1.legend()
    ax1.set_xlabel("Frequency(Hz)")
    ax1.set_ylabel("Magnitude")
    ax1.set_title("Permuted Frequency Responses for each component")

ax2 = fig.add_subplot(122)
ax2.plot(frq_range[1:], spectrum_all[1:], 
         label = "{} (DC: {:.2e})".format("all features", spectrum_all[0]))
ax2.plot(frq_range[1:], spectrum_true[1:], 
         label = "{} (DC: {:.2e})".format("True y", spectrum_true[0]))
ax2.legend()
ax2.set_xlim(frq_range[1], frq_range[-1])
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Magnitude")
ax2.set_title("Frequency Responses for the all components")
plt.show()
```
![FT_demo2](https://github.com/user-attachments/assets/f714673d-3395-485d-bf90-c2643a907ee4)


## Reference

*[1] Hooker, Sara, et al. "A benchmark for interpretability methods in deep neural networks." arXiv preprint arXiv:1806.10758 (2018).*

*[2] Breiman, Leo. "Random forests." Machine learning 45.1 (2001): 5-32.*

*[3] Fisher, Aaron, Cynthia Rudin, and Francesca Dominici. "All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models
Simultaneously." Journal of Machine Learning Research 20.177 (2019): 1-81.*

*[4] Wei, Pengfei, Zhenzhou Lu, and Jingwen Song. "Variable importance analysis: A comprehensive review." Reliability Engineering & System Safety 142 (2015): 399-432.*

*[5] Mao, Jianqiao, and Grammenos Ryan. "Interpreting machine learning models for room temperature prediction in non-domestic buildings." arXiv preprint arXiv:2111.13760 (2021).*

---

## v1.0.1 Update

1. Release a new feature enabling wavelet transform-based PFFRA for time-freqency analysis and interpretation. Details can be checked in the new released tutorial notebook. Figure below shows the results:

![wavelet_demo](https://github.com/JianqiaoMao/PFFRA/assets/60654068/50c049bf-6d06-4c7c-afc3-876b9cc1bc3a)


## v0.1.2 Update

1. Open more customised interfaces to give users more freedom to use blocks in the PF-FRA algorithm.
  - Now each built-in method of ``PermFeatureFreqRespoAnalysis`` object is callable to allow users to input customised parameters.
  - Add ``shuffle`` mode for feature permutation.
2. Correct some minor typos in the code to improve its readability.

