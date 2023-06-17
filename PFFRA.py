# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 23:11:20 2023

@author: Jianqiao Mao
"""
import numpy as np
import matplotlib.pyplot as plt
        
class PermFeatureFreqRespoAnalysis:
    def __init__(self, X, y, model, interested_feature_index, sample_rate, mode='mean'):
        """
        Initialize the PF_FRA class.

        Parameters:
        - X: Input data as an array.
        - y: Target data as an array.
        - model: Trained model object that has a `predict` method.
        - interested_feature_index: Index or indices of the features to analyze.
        - sample_rate: Sample rate of the data.
        - mode: Mode for generating permuted datasets. Options: 'mean', 'median', 'most_frequent'.

        """
        self.y = np.array(y)
        self.X = np.array(X)
        self.model = model
        self.interested_feature_index = interested_feature_index
        self.sample_rate = sample_rate
        self.mode = mode
        

    def permuted_dataset(self):
        """
        Generate permuted datasets for the interested feature(s).

        Returns:
        - X_interested_feature: Permuted dataset with other feature(s) replaced by mean/median/most frequent values.
        - X_other_feature: Permuted dataset with interested feature(s) replaced by mean/median/most frequent values.
        """
        if isinstance(self.interested_feature_index, int):
            self.interested_feature_index = [self.interested_feature_index]

        other_feature_index = np.setdiff1d(np.arange(self.X.shape[1]), self.interested_feature_index)

        if self.mode == 'mean':
            perm_value = np.mean(self.X, axis=0)
        elif self.mode == 'median':
            perm_value = np.median(self.X, axis=0)
        elif self.mode == 'most_frequent':
            perm_value = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=self.X)
        else:
            raise ValueError("Invalid mode. Choose either 'mean', 'median', or 'most_frequent'.")

        X_interested_feature = self.X.copy()
        X_interested_feature[:, other_feature_index] = perm_value[other_feature_index]

        X_other_feature = self.X.copy()
        X_other_feature[:, self.interested_feature_index] = perm_value[self.interested_feature_index]

        return X_interested_feature, X_other_feature

    def permu_pred(self):
        """
        Generate predictions for the permuted datasets.

        Returns:
        - pred_interested_feature: Predictions for the dataset with other feature(s) permuted.
        - pred_other_feature: Predictions for the dataset with interested feature(s) permuted.
        """
        X_interested_feature, X_other_feature = self.permuted_dataset()
        pred_interested_feature = self.model.predict(X_interested_feature)
        pred_other_feature = self.model.predict(X_other_feature)

        return pred_interested_feature, pred_other_feature

    def one_sided_fft(self, signal):
        """
        Perform one-sided FFT on the input signal.

        Parameters:
        - signal: Input signal as a 1-D numpy array.

        Returns:
        - one_sided_fourier: One-sided FFT of the signal.
        - freq_axis: Frequency axis corresponding to the FFT values.
        """
        n = signal.shape[0]
        freq_resolution = self.sample_rate / n

        signal_fourier = np.fft.fft(signal)
        one_sided_fourier = np.abs(signal_fourier[:n // 2] * freq_resolution)

        freq_axis = np.linspace(0, self.sample_rate / 2, n // 2)

        return one_sided_fourier, freq_axis

    def gen_specturm(self):
        """
        Generate frequency spectra for the predictions.

        Returns:
        - specturm_interested: Frequency spectrum for predictions with other feature(s) permuted.
        - specturm_other: Frequency spectrum for predictions with interested feature(s) permuted.
        - specturm_all: Frequency spectrum for the original predictions.
        - frq_range: Frequency range corresponding to the spectra.
        """
        pred_interested_feature, pred_other_feature = self.permu_pred()

        specturm_other, frq_range = self.one_sided_fft(pred_other_feature)
        specturm_interested, frq_range = self.one_sided_fft(pred_interested_feature)
        specturm_all, frq_range = self.one_sided_fft(self.y)

        return specturm_interested, specturm_other, specturm_all, frq_range

    def show(self, rename_feature='Interested Feature'):
        """
        Display the frequency responses.

        Parameters:
        - rename_feature: Name to display for the interested feature.

        Returns:
        None
        """
        specturm_intereted, specturm_other, specturm_all, frq_range = self.gen_specturm()
        frq_range = frq_range[1:]

        DC_intereted_feature = specturm_intereted[0]
        specturm_intereted_feature = specturm_intereted[1:]

        DC_other_feature = specturm_other[0]
        specturm_permu_feature = specturm_other[1:]

        DC_all = specturm_all[0]
        specturm_all = specturm_all[1:]

        fig = plt.figure()
        plt.rcParams["axes.labelsize"] = 14
        plt.rcParams["axes.titlesize"] = 16
        plt.rcParams["xtick.labelsize"] = "large"
        plt.rcParams["ytick.labelsize"] = "large"
        ax1 = fig.add_subplot(211)
        ax1.plot(frq_range, specturm_intereted_feature, label="{} (DC: {:.2f})".format(rename_feature, DC_intereted_feature))
        ax1.plot(frq_range, specturm_permu_feature, label="Other Features (DC: {:.2f})".format(DC_other_feature))
        ax1.legend()
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Magnitude")
        ax1.set_title("Frequency Responses for the {} Feature and Others".format(rename_feature))
        ax2 = fig.add_subplot(212)
        ax2.plot(frq_range, specturm_all, label="All features (DC: {:.2f})".format(DC_all), c="g")
        ax2.legend()
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude")
        ax2.set_title("Frequency Responses for all features")
        
        plt.tight_layout()
        plt.show()
     
        
        
        
