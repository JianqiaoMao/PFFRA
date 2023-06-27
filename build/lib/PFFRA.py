# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 23:11:20 2023

@author: Jianqiao Mao
"""
import numpy as np
import matplotlib.pyplot as plt
        
class PermFeatureFreqRespoAnalysis:
    def __init__(self, X, y, model, interested_feature_index):
        """
        Initialize the PermFeatureFreqRespoAnalysis class.

        Parameters:
        - X: Feature data as an array.
        - y: Target data as an array.
        - model: Trained model object that has a `predict` method.
        - interested_feature_index: Index or indices of the features to analyze.

        """
        self.model = model
        self.interested_feature_index = interested_feature_index
        self.X = np.array(X)
        self.y = np.array(y)
        

    def permuted_dataset(self, mode='mean'):
        """
        Generate permuted datasets for the interested feature(s).

        Parameters:
        - mode: Mode for generating permuted datasets. Options: 'mean', 'median', 'most_frequent', 'shuffle'. Default: 'mean'.

        Returns:
        - X_interested_feature: Permuted dataset with other feature(s) replaced by mean/median/most frequent values.
        - X_other_feature: Permuted dataset with interested feature(s) replaced by mean/median/most frequent values.
        """
        
        if isinstance(self.interested_feature_index, int):
            self.interested_feature_index = [self.interested_feature_index]

        other_feature_index = np.setdiff1d(np.arange(self.X.shape[1]), self.interested_feature_index)

        if mode == 'mean':
            perm_value = np.mean(self.X, axis=0).reshape(1,-1)
        elif mode == 'median':
            perm_value = np.median(self.X, axis=0).reshape(1,-1)
        elif mode == 'most_frequent':
            perm_value = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=self.X).reshape(1,-1)
        elif mode == 'shuffle':
            perm_value = self.X.copy()
            np.random.shuffle(perm_value)
        else:  
            raise ValueError("Invalid mode. Choose either 'mean', 'median', 'most_frequent', or 'shuffle'.")

        X_interested_feature = self.X.copy()
        X_interested_feature[:, other_feature_index] = perm_value[:,other_feature_index]

        X_other_feature = self.X.copy()
        X_other_feature[:, self.interested_feature_index] = perm_value[:,self.interested_feature_index]

        return X_interested_feature, X_other_feature

    def permu_pred(self, X_interested_feature, X_other_feature):
        """
        Generate predictions for the permuted datasets.
        
        Parameters:
        - X_interested_feature: Feature data with other feature(s) permuted.
        - X_other_feature: Feature data with interested feature(s) permuted.

        Returns:
        - pred_interested_feature: Predictions for the dataset with other feature(s) permuted.
        - pred_other_feature: Predictions for the dataset with interested feature(s) permuted.
        - pred_all_feature: Predictions for the original dataset with all features.
        """
        pred_interested_feature = self.model.predict(X_interested_feature)
        pred_other_feature = self.model.predict(X_other_feature)
        pred_all_feature = self.model.predict(self.X)

        return pred_interested_feature, pred_other_feature, pred_all_feature

    def one_sided_fft(self, signal, sample_rate):
        """
        Perform one-sided FFT on the input signal.

        Parameters:
        - signal: Input signal as a 1-D numpy array.
        - sample_rate: Sample rate of the data.

        Returns:
        - one_sided_fourier: One-sided FFT of the signal.
        - freq_axis: Frequency axis corresponding to the FFT values.
        """
        n = signal.shape[0]
        freq_resolution = sample_rate / n

        signal_fourier = np.fft.fft(signal)
        one_sided_fourier = np.abs(signal_fourier[:n // 2] * freq_resolution)

        freq_axis = np.linspace(0, sample_rate / 2, n // 2)

        return one_sided_fourier, freq_axis

    def gen_spectrum(self, pred_interested_feature, pred_other_feature, pred_all_feature, sample_rate):
        """
        Generate frequency spectra for the predictions.
        - pred_interested_feature: Predictions for the dataset with other feature(s) permuted.
        - pred_other_feature: Predictions for the dataset with interested feature(s) permuted.
        - pred_all_feature: Predictions for the original dataset with all features.
        - sample_rate: Sample rate of the data.        

        Returns:
        - spectrum_interested: Frequency spectrum for predictions with other feature(s) permuted.
        - spectrum_other: Frequency spectrum for predictions with interested feature(s) permuted.
        - spectrum_all: Frequency spectrum for the original predictions.
        - spectrum_true: Frequency spectrum for the original true target variable.
        - frq_range: Frequency range corresponding to the spectra.
        """

        spectrum_other, frq_range = self.one_sided_fft(pred_other_feature, sample_rate)
        spectrum_interested, frq_range = self.one_sided_fft(pred_interested_feature, sample_rate)
        spectrum_all, frq_range = self.one_sided_fft(pred_all_feature, sample_rate)
        spectrum_true, frq_range = self.one_sided_fft(self.y, sample_rate)

        return spectrum_interested, spectrum_other, spectrum_all, spectrum_true, frq_range

    def show(self, sample_rate, mode = 'mean', rename_feature='Interested Feature'):
        """
        Display the frequency responses.

        Parameters:
        - sample_rate: Sample rate of the data.
        - rename_feature: Name to display for the interested feature.
        - mode: Mode for generating permuted datasets. Options: 'mean', 'median', 'most_frequent', 'shuffle'. Default: 'mean'.

        Returns:
        None
        """
        
        X_interested_feature, X_other_feature = self.permuted_dataset(mode)
        pred_interested_feature, pred_other_feature, pred_all_feature = self.permu_pred(X_interested_feature, X_other_feature)
        spectrum_interested, spectrum_other, spectrum_all, spectrum_true, frq_range = self.gen_spectrum(pred_interested_feature, pred_other_feature, pred_all_feature, sample_rate)
        
        frq_range = frq_range[1:]

        DC_intereted_feature = spectrum_interested[0]
        spectrum_intereted_feature = spectrum_interested[1:]

        DC_other_feature = spectrum_other[0]
        spectrum_permu_feature = spectrum_other[1:]

        DC_all = spectrum_all[0]
        spectrum_all = spectrum_all[1:]
        
        DC_true = spectrum_true[0]
        spectrum_true = spectrum_true[1:]
        
        fig = plt.figure()
        plt.rcParams["axes.labelsize"] = 14
        plt.rcParams["axes.titlesize"] = 16
        plt.rcParams["xtick.labelsize"] = "large"
        plt.rcParams["ytick.labelsize"] = "large"
        ax1 = fig.add_subplot(211)
        ax1.plot(frq_range, spectrum_intereted_feature, label="{} (DC: {:.2f})".format(rename_feature, DC_intereted_feature))
        ax1.plot(frq_range, spectrum_permu_feature, label="Other Features (DC: {:.2f})".format(DC_other_feature))
        ax1.legend()
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_title("Frequency Responses for the {} Feature and Others".format(rename_feature))
        ax2 = fig.add_subplot(212)
        ax2.plot(frq_range, spectrum_all, label="All features (DC: {:.2f})".format(DC_all), c="g")
        ax2.plot(frq_range, spectrum_true, label = "True y (DC:{:.2F}".format(DC_true), c = 'k')
        ax2.legend()
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude")
        ax2.set_title("Frequency Response for all features and the true target variable")
        
        plt.tight_layout()
        plt.show()
        