'''
DISTRIBUTION STATEMENT F: Further dissemination only as directed by Army Rapid Capabilities Office, or higher DoD authority.

    Data Rights Notice

        This software was produced for the U. S. Government under Basic Contract No. W15P7T-13-C-A802, and is subject to the Rights in Noncommercial Computer Software and Noncommercial Computer Software Documentation Clause 252.227-7014 (FEB 2012)
        Copyright 2017 The MITRE Corporation. All Rights Reserved.

    .. note ::
        Classes is this module:
            :class:`LoadModRecData`

    :author: Bill Urrego - wurrego@mitre.org
    :date: 08/15/16

'''
from license import DATA_RIGHTS
from version import __version__

__author__ = "Bill Urrego - wurrego@mtire.org"
__license__ = DATA_RIGHTS
__version__ = __version__
__last_modified__ = '9/04/17'

""" INCLUDES"""
import pickle
import numpy as np
from random import shuffle
import random
import json, os, sys
import datetime
import matplotlib.pyplot as plt
from numpy.fft import *
TAG = '[Data Loader] - '


class LoadModRecData:
    ''' Class for working with modulation recognition data sets

        .. note::

            The following methods are provided in this class:

                * :func:`__init__`
                * :func:`loadData`
                * :func:`split`
                * :func:`get_batch_from_indicies`
                * :func:`get_batch_from_indicies_withSNRthrehsold`
                * :func:`get_random_batch`
                * :func:`inspect_signal`
                * :func:`train_batch_iter`
                * :func:`get_batch_from_indicies_withSNRthrehsold`

    '''

    def __init__(self, datafile, trainRatio, validateRatio, testRatio, load_mods=None, load_snrs=None, num_samples_per_key=None, verbose=True, snrDict=None):
        ''' init

            .. note::

                calls :func:`loadData` and :func:`split`
        '''

        # check python version
        self.python_version_3 = False
        if sys.version_info >= (3, 0):
            self.python_version_3 = True
        self.verbose = verbose
        
        print(TAG + "Loading Datafile, ", datafile)
  
        self.signalData, self.oneHotLabels, self.signalLabels, self.snrLabels = self.loadData(datafile, load_mods=load_mods, load_snrs=load_snrs, num_samples_per_key=num_samples_per_key, snrDict=snrDict)
        self.train_idx, self.val_idx, self.test_idx = self.split(trainRatio, validateRatio, testRatio)
        if self.verbose:
            print(TAG + "Done.\n")

    def loadData(self, fname, load_mods, load_snrs, num_samples_per_key, snrDict):
        '''  Load dataset from pickled file '''
        # load data from files
        with open(fname, 'rb') as f:
            if self.python_version_3:
                self.dataCube = pickle.load(f, encoding='latin-1')
                dataCubeKeyIndices = list(zip(*self.dataCube))
            else:
                self.dataCube = pickle.load(f)
                dataCubeKeyIndices = zip(*self.dataCube)

        # get all mod types
        if load_mods is None:
            self.modTypes = np.unique(dataCubeKeyIndices[0])
        else:
            self.modTypes = load_mods

        # get all SNR values
        if load_snrs is None:
            self.snrValues = np.unique(dataCubeKeyIndices[1])
        else:
            self.snrValues = load_snrs

        # create one-hot vectors for each mod type
        oneHotArrays = np.eye(len(self.modTypes), dtype=int)
        
        snr_to_one_hot_index = {}
        if snrDict:
            oneHotLength = len(set(snrDict.values()))
            snr_to_one_hot_index = snrDict
        else:
            for i in range(len(self.snrValues)):
                snr_to_one_hot_index[self.snrValues[i]] = i
            oneHotLength = len(self.snrValues)
            
        snrOneHotArrays = np.eye(oneHotLength, dtype=int)

        # Count Number of examples
        if self.verbose:
            print(TAG + "Counting Number of Examples in Dataset...")
        number_of_examples = 0
        for modType in self.modTypes:
            for snrValue in self.snrValues:
                if num_samples_per_key:
                    number_of_examples = number_of_examples + num_samples_per_key
                else:    
                    number_of_examples = number_of_examples + len(self.dataCube[modType, snrValue])
        if self.verbose:
            print (TAG + 'Number of Examples in Dataset: ' + str(number_of_examples))

        # pre-allocate arrays
        signalData = [None] * number_of_examples
        oneHotLabels = [None] * number_of_examples
        signalLabels = [None] * number_of_examples
        snrLabels = [None] * number_of_examples

        # for each mod type ... for each snr value ... add to signalData, signalLabels, and create one-Hot vectors
        example_index = 0
        one_hot_index = 0
        self.instance_shape = None
        
            
        for modType in self.modTypes:
            if self.verbose:
                print(TAG + "[Modulation Dataset] Adding Collects for: " + str(modType))
            for snrValue in self.snrValues:
                # get data for key,value
                if num_samples_per_key:
                    collect = self.dataCube[modType, snrValue][:num_samples_per_key]
                else:
                    collect = self.dataCube[modType, snrValue]
                # print(modType, snrValue, collect.shape)
                for instance in collect:
                    signalData[example_index] = instance
                    signalLabels[example_index] = (modType, snrValue)
                    oneHotLabels[example_index] = oneHotArrays[one_hot_index]
                    snrLabels[example_index] = snrOneHotArrays[snr_to_one_hot_index[snrValue]]              
                    example_index += 1

                    if self.instance_shape is None:
                        self.instance_shape = np.shape(instance)

            one_hot_index += 1  # keep track of iteration for one hot vector generation

        # convert to np.arrays
        if self.verbose:
            print(TAG + "Converting to numpy arrays...")
        signalData = np.asarray(signalData)
        oneHotLabels = np.asarray(oneHotLabels)
        signalLabels = np.asarray(signalLabels)
        snrLabels = np.asarray(snrLabels)

        # Shuffle data
        if self.verbose:
            print(TAG + "Shuffling Data...")
        """ signalData_shuffled, signalLabels_shuffled, oneHotLabels_shuffled """
        # Randomly shuffle data, use predictable seed
        np.random.seed(2017)
        shuffle_indices = np.random.permutation(np.arange(len(signalLabels)))
        signalData_shuffled = signalData[shuffle_indices]
        signalLabels_shuffled = signalLabels[shuffle_indices]
        snrLabels_shuffled = snrLabels[shuffle_indices]
        oneHotLabels_shuffled = oneHotLabels[shuffle_indices]

        return signalData_shuffled, oneHotLabels_shuffled, signalLabels_shuffled, snrLabels_shuffled

    def split(self, trainRatio, validateRatio, testRatio):
        '''  split dataset into train, validation, and test '''

        # Split data into train/validate/test via indexing
        if self.verbose:
            print(TAG + "Splitting Data...")

        # Determine how many samples go into each set
        [num_sigs, num_samples] = np.shape(self.oneHotLabels)
        num_train = np.int(np.floor(num_sigs * trainRatio))
        num_val = np.int(np.floor(num_sigs * validateRatio))
        num_test = np.int(num_sigs - num_train - num_val)
        if self.verbose:
            print(TAG + 'Train Size: ' + str(num_train) + ' Validation Size: ' + str(num_val) + ' Test Size: ' + str(num_test))
        # Generate a random permutation of the sample indicies
        rand_perm = np.random.permutation(num_sigs)

        # Asssign Indicies to sets
        train_idx = rand_perm[0:num_train]
        val_idx = rand_perm[num_train:num_train + num_val]
        #test_idx = rand_perm[num_sigs-num_test:]
        test_idx = rand_perm[num_train + num_val:]

        return train_idx, val_idx, test_idx

    def get_indicies_withSNRthrehsold(self, indicies, snrThreshold_lowBound, snrThreshold_upperBound):
        '''  get batch from indicies with SNR threshold  - (inclusive >=lowerBound, <= upperBound) '''

        filteredIndicies = []
        i = 0
        for snrValue in self.signalLabels[indicies][:, 1]:
            if int(snrValue) >= int(snrThreshold_lowBound) and int(snrValue) <= int(snrThreshold_upperBound):
                filteredIndicies.extend([indicies[i]])
            i += 1

        return filteredIndicies

    def get_batch_from_indicies(self, indicies):
        '''  get batch from indicies  '''

        batch_x = self.signalData[indicies]
        batch_y = self.oneHotLabels[indicies]
        batch_y_labels = self.signalLabels[indicies]

        # return the batch
        return zip(batch_x, batch_y, batch_y_labels)

    def get_random_batch(self, index_list, batch_size):
        ''' get batch of specific size from dataset '''

        rand_pool = np.random.choice(np.shape(index_list)[0], size=batch_size, replace=False)
        rand_idx = index_list[rand_pool[0:batch_size]]

        # use the indices to get the data for x and y
        batch_x = self.signalData[rand_idx]
        batch_y = self.oneHotLabels[rand_idx]
        batch_y_labels = self.signalLabels[rand_idx]

        # return the batch
        return zip(batch_x, batch_y, batch_y_labels)

    def batch_iter(self, data_indicies, batch_size, num_epochs, use_shuffle=True, yield_snr=False, train_snr=False):
        '''  provide generator for iteration of the training indicies created during initialization '''

        # iteration - one batch_size from data
        # epoch - one pass through all of the data

        data_size = len(data_indicies)
        num_batches_per_epoch = int(len(data_indicies) / batch_size)

        for epoch in range(num_epochs):

            # Shuffle the indices at each epoch
            if use_shuffle:
                shuffle(data_indicies)

            # loop across all example data one batch at a time
            for batch_num in range(num_batches_per_epoch):
                # determine start index
                start_index = batch_num * batch_size

                # determine end index, min of iteration vs end of data
                end_index = min((batch_num + 1) * batch_size, data_size)

                # get indices of the instances in the batch
                indices_of_instances_in_batch = data_indicies[start_index:end_index]

                # use the indices to get the data for x and y
                batch_x = self.signalData[indices_of_instances_in_batch]
                batch_y = self.oneHotLabels[indices_of_instances_in_batch]
                batch_y_labels = self.signalLabels[indices_of_instances_in_batch]
                batch_snr_one_hot = self.snrLabels[indices_of_instances_in_batch]

                # return a training batch
                #yield zip(batch_x, batch_y, batch_y_labels)
                if train_snr:
                    batch_y = batch_snr_one_hot
                if yield_snr:
                    yield batch_x, batch_y, batch_y_labels
                else:
                    yield batch_x, batch_y

    def inspect_signal(self, index, modulation, snr, cdata, time, number_of_samples_in_instance, sample_rate,
                       start_freq, stop_freq, interactive=None):
        ''' inspects a specific instance within the collection. Plots FFT and I,Q.  Prints to console raw I Q amplitudes

        :param index: index of instance in collection
        :type index: int
        :param modulation: the modulation schema of the instnace
        :type modulation: str
        :param snr: the SNR of the instance
        :type snr: int
        :param cdata: the complex samples for the instance
        :type cdata: numpy.array dtype='complex64'
        :param time: time series values
        :type time: list of type float
        :param number_of_samples_in_instance: number of samples in the instance
        :type number_of_samples_in_instance: int
        :param sample_rate: sampling rate of the collect
        :type sample_rate: float
        :param start_freq: start frequency in Hz of the collect
        :type start_freq: int
        :param stop_freq: stop frequency in Hz of the collect
        :type stop_freq: int
        :param interactive: optional value for display of each plot (if True) requires user to close plot before next is displayed)
        :type interactive: bool

        :return: N/A


        .. note:

            Using this method will result in the following outputs:

                * Plots will be saved to :attr:`plots_out_dir`
                * Console will display raw data (real valued I and Q) as well as complex samples for each instance
                * All inspected instances will be written to a json file with the following format::

                    {
                        "time_stamp_1": {"y": [y_data], "x": [x_data], "i": [real_valued_i], "q": [real_valued_q], "t": [time_series_values] },
                        "time_stamp_2": {"y": [y_data], "x": [x_data], "i": [real_valued_i], "q": [real_valued_q], "t": [time_series_values] },
                        ...,
                        "time_stamp_N": {"y": [y_data], "x": [x_data], "i": [real_valued_i], "q": [real_valued_q], "t": [time_series_values] }
                    }
        '''

        # Set FFT Bin Size
        fft_bin_size = number_of_samples_in_instance

        # Determine number of FFTs
        numFfts = int(np.floor(len(cdata) / fft_bin_size))

        # setup x axis information for plotting
        startBin = start_freq - (sample_rate / 2)
        stopBin = stop_freq + (sample_rate / 2)
        fftStep = sample_rate / fft_bin_size

        # output parameters
        plots_out_dir = 'test_instance_plots/'
        out_file = 'test_data_instances.json'
        databuffer = {}
        ind = 0
        ind_iq = 0

        # for required FFTs
        for k in range(numFfts):

            # take FFT
            fftData = np.fft.fftshift(np.fft.fft(cdata[ind:ind + fft_bin_size], norm='ortho'))
            absFft = 20.0 * np.log10(np.abs(fftData))

            # generate x,y data points
            x_axis = list(np.arange(startBin, stopBin, fftStep))[1:]
            y_axis = absFft.tolist()

            # set timestamp with precision of 1ms
            timestamp = datetime.datetime.utcnow().strftime("%d-%b-%Y %H:%M:%S")
            timePrecision = np.round(datetime.datetime.utcnow().microsecond / 1e3)
            timestamp = timestamp + "." + str(int(timePrecision))

            # get real valued I and Q
            sig_I = cdata.real.tolist()
            sig_Q = cdata.imag.tolist()

            # store data for this instance
            databuffer[timestamp] = (
            {"x": x_axis, "y": y_axis, "i": sig_I[ind_iq:ind_iq + number_of_samples_in_instance],
             "q": sig_Q[ind_iq:ind_iq + number_of_samples_in_instance],
             "t": time[ind_iq:ind_iq + number_of_samples_in_instance]})

            # generate plots
            fig, ax = plt.subplots(nrows=2, ncols=1)

            ax[0].plot(x_axis, y_axis, 'r')  # plotting the spectrum
            ax[0].set_xlabel('Freq (CBB)')
            ax[0].set_ylabel('|Y(freq)|')
            plt.title('FFT ' + modulation + ' ' + str(snr) + ' dB ' + str(index))

            ax[1].plot(time[ind_iq:ind_iq+number_of_samples_in_instance], sig_I[ind_iq:ind_iq+number_of_samples_in_instance], 'r')  # plotting the time series data
            ax[1].plot(time[ind_iq:ind_iq+number_of_samples_in_instance], sig_Q[ind_iq:ind_iq+number_of_samples_in_instance], 'b')  # plotting the time series data
            ax[1].set_xlabel('Time')
            ax[1].set_ylabel('Amplitude')
            plt.title('IQ ' + modulation + ' ' + str(snr) + ' dB ' + str(index))

            # create the plots save directory if it does not exist
            if not os.path.isdir(plots_out_dir):
                os.makedirs(plots_out_dir)

            plt.savefig(
                os.path.join(plots_out_dir, 'fft_iq_' + str(index) + '_' + modulation + '_' + str(snr) + 'dB' + '.png'))

            # setup console printing of data
            cdata_print = str(cdata).replace('\n', '')
            cdata_print = str(cdata_print).replace('j', 'j,')
            cdata_print = cdata_print[:-2]  # remove last , and bracket
            cdata_print = cdata_print + ']'  # add last bracket back in

            # print to console (for copy/paste into MATLAB or python for further analysis)
            print ('\n')
            print (modulation + '_' + str(snr) + str(index) + '_I = ' + str(sig_I))
            print (modulation + '_' + str(snr) + str(index) + '_Q = ' + str(sig_Q))
            print ('instance_' + modulation + '_' + str(snr) + str(index) + '_complex = ' + cdata_print)

            # show plot if interactive
            if interactive:
                plt.show()

            # close the plot
            plt.close()

            # increment iteration parameters
            ind = ind + fft_bin_size
            ind_iq = ind_iq + number_of_samples_in_instance

        # store all saved instacne data to a json file
        with open(out_file, 'w') as fp:
            json.dump(databuffer, fp)


def get_complex_samples_for_instance(collect, index=None):
    '''Returns an instance worth of complex samples from the collection provided

    :param collect: a python collection of modulations and SNR values
    :type collect: dict['modulation',snr]
    :param index: a specific index into the collection. Optional parameter, if None a random index will be selected
    :type index: int

    :returns: the complex data and index of instance
    :rtype: tuple

    '''

    # determine index
    if index is None:
        index = random.randint(0, len(collect) - 1)

    # get samples
    sample_buffer = collect[index]

    # transform to complex
    cdata = sample_buffer[0::2] + 1.0j * sample_buffer[1::2]
    cdata = np.complex64(cdata)

    # return complex data and index the instance was pulled from
    return cdata, index


if __name__ == "__main__":
    ''' Test Application that uses LoadModRecData '''

    # path for dataset to load
    train_data_file_path = ''

    # Load data
    print ('Loading RadioML Data...')
    data = LoadModRecData(train_data_file_path, .7, .2, .1)

    # Number of Samples in Instance
    number_of_samples_in_instance = data.instance_shape[1]

    # for each modulation type, inspect some instances of data
    for modType in data.modTypes:
        min_SNR = -10
        max_SNR = 10
        snr_step = 4

        # get random snr within valid range
        snrValue = random.randrange(min_SNR, max_SNR, snr_step)

        # get a collection of signals
        collection_of_instances = data.dataCube[modType, snrValue]

        # get the complex samples for a specific instance within the collection
        cdata, index = get_complex_samples_for_instance(collection_of_instances)

        # setup inspect
        sample_rate = number_of_samples_in_instance
        Ts = 1.0 / sample_rate
        t = np.arange(0, len(cdata[0])) * Ts

        # inspect signal
        data.inspect_signal(index, modType, snrValue, cdata[0], t.tolist(), number_of_samples_in_instance, sample_rate, 0, 0)

