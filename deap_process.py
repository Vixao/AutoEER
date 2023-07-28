import os
import sys
import math
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn import preprocessing
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data)
	return y

def read_file(file):
	data = sio.loadmat(file)
	data = data['data']
	# print(data.shape)
	return data

def compute_DE(signal):
	variance = np.var(signal,ddof=1)
	return math.log(2*math.pi*math.e*variance)/2

def decompose(file):
	# trial*channel*sample
	start_index = 384 #3s pre-trial signals
	data = read_file(file)
	shape = data.shape
	frequency = 128

	decomposed_de = np.empty([0,4,240])

	base_DE = np.empty([0,128])

	for trial in range(40):
		temp_base_DE = np.empty([0])
		temp_base_theta_DE = np.empty([0])
		temp_base_alpha_DE = np.empty([0])
		temp_base_beta_DE = np.empty([0])
		temp_base_gamma_DE = np.empty([0])

		temp_de = np.empty([0,240])

		for channel in range(32):
			trial_signal = data[trial,channel,384:]
			base_signal = data[trial,channel,:384]
			#****************compute base DE****************
			base_theta = butter_bandpass_filter(base_signal, 4, 8, frequency, order=3)
			base_alpha = butter_bandpass_filter(base_signal, 8,14, frequency, order=3)
			base_beta = butter_bandpass_filter(base_signal,14,31, frequency, order=3)
			base_gamma = butter_bandpass_filter(base_signal,31,45, frequency, order=3)

			base_theta_DE = (compute_DE(base_theta[:128])+compute_DE(base_theta[128:256])+compute_DE(base_theta[256:]))/3
			base_alpha_DE =(compute_DE(base_alpha[:128])+compute_DE(base_alpha[128:256])+compute_DE(base_alpha[256:]))/3
			base_beta_DE =(compute_DE(base_beta[:128])+compute_DE(base_beta[128:256])+compute_DE(base_beta[256:]))/3
			base_gamma_DE =(compute_DE(base_gamma[:128])+compute_DE(base_gamma[128:256])+compute_DE(base_gamma[256:]))/3

			temp_base_theta_DE = np.append(temp_base_theta_DE,base_theta_DE)
			temp_base_gamma_DE = np.append(temp_base_gamma_DE,base_gamma_DE)
			temp_base_beta_DE = np.append(temp_base_beta_DE,base_beta_DE)
			temp_base_alpha_DE = np.append(temp_base_alpha_DE,base_alpha_DE)

			theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
			alpha = butter_bandpass_filter(trial_signal, 8, 14, frequency, order=3)
			beta = butter_bandpass_filter(trial_signal, 14, 31, frequency, order=3)
			gamma = butter_bandpass_filter(trial_signal, 31, 45, frequency, order=3)

			DE_theta = np.zeros(shape=[0],dtype = float)
			DE_alpha = np.zeros(shape=[0],dtype = float)
			DE_beta =  np.zeros(shape=[0],dtype = float)
			DE_gamma = np.zeros(shape=[0],dtype = float)

			for index in range(240):
				DE_theta =np.append(DE_theta,compute_DE(theta[index*32:(index+1)*32]))
				DE_alpha =np.append(DE_alpha,compute_DE(alpha[index*32:(index+1)*32]))
				DE_beta =np.append(DE_beta,compute_DE(beta[index*32:(index+1)*32]))
				DE_gamma =np.append(DE_gamma,compute_DE(gamma[index*32:(index+1)*32]))
			temp_de = np.vstack([temp_de,DE_theta])
			temp_de = np.vstack([temp_de,DE_alpha])
			temp_de = np.vstack([temp_de,DE_beta])
			temp_de = np.vstack([temp_de,DE_gamma])
		temp_trial_de = temp_de.reshape(-1,4,240)
		decomposed_de = np.vstack([decomposed_de,temp_trial_de])

		temp_base_DE = np.append(temp_base_theta_DE,temp_base_alpha_DE)
		temp_base_DE = np.append(temp_base_DE,temp_base_beta_DE)
		temp_base_DE = np.append(temp_base_DE,temp_base_gamma_DE)
		base_DE = np.vstack([base_DE,temp_base_DE])
	decomposed_de = decomposed_de.reshape(-1,32,4,240).transpose([0,3,2,1]).reshape(-1,4,32).reshape(-1,128)
	print("base_DE shape:",base_DE.shape)
	print("trial_DE shape:",decomposed_de.shape)
	return base_DE,decomposed_de

def get_labels(file):
	#0 valence, 1 arousal, 2 dominance, 3 liking
	av_labels=[]
	valence_labels = sio.loadmat(file)["labels"][:,0]>5	# valence labels
	arousal_labels = sio.loadmat(file)["labels"][:,1]>5	# arousal labels
	for i in range(40):
		valence=valence_labels[i]
		arousal=arousal_labels[i]
		if valence==1 and arousal==1:
			label=3
		elif valence==1 and arousal==0:
			label=2
		elif valence==0 and arousal==1:
			label=1
		elif valence==0 and arousal==0:
			label=0
		av_labels=np.append(av_labels,label)
	#print(av_labels)
	final_valence_labels = np.empty([0])
	final_arousal_labels = np.empty([0])
	final_av_labels = np.empty([0])
	for i in range(len(valence_labels)):
		for j in range(0,60):
			final_valence_labels = np.append(final_valence_labels,valence_labels[i])
			final_arousal_labels = np.append(final_arousal_labels,arousal_labels[i])
			final_av_labels = np.append(final_av_labels,av_labels[i])
	print("labels:",final_valence_labels.shape)
	return final_arousal_labels,final_valence_labels,final_av_labels

def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

def feature_normalize(data):
	mean = data[data.nonzero()].mean()
	sigma = data[data. nonzero ()].std()
	data_normalized = data
	data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma
	return data_normalized

def get_features(band_index):
	feature_index = np.empty(0)
	for i in band_index:
		band = np.array(range((i-1)*32,i*32))
		feature_index = np.append(feature_index,band)
	feature_index = list(map(int,feature_index))
	return feature_index
	
def get_vector_deviation(vector1,vector2):
	return vector1-vector2

def get_dataset_deviation(trial_data,base_data):
	new_dataset = np.empty([0,128])
	for i in range(0,9600):
		base_index = i//240
		# print(base_index)
		base_index = 39 if base_index == 40 else base_index
		new_record = get_vector_deviation(trial_data[i],base_data[base_index]).reshape(1,128)
		# print(new_record.shape)
		new_dataset = np.vstack([new_dataset,new_record])
		#print("new shape:",new_dataset.shape)
	return new_dataset

if __name__ == '__main__':
	dataset_dir = "/home/wyx/data/DEAP/DEAP/"

	result_dir = "/home/wyx/data/DEAP_DE/3D_2400_32_4_4/"
	if os.path.isdir(result_dir)==False:
		os.makedirs(result_dir)

	for file in os.listdir(dataset_dir):
		print("processing: ",file,"......")
		file_path = os.path.join(dataset_dir,file)
		base_DE,trial_DE = decompose(file_path)
		arousal_labels,valence_labels,av_labels= get_labels(file_path)
		data=get_dataset_deviation(trial_DE,base_DE)
		data=data.reshape(9600,4,32).reshape(2400,4,4,32).transpose(0,1,3,2).transpose(0,3,2,1).transpose(0,2,1,3)
		print("data.shape",data.shape)
		sio.savemat(result_dir+"DE_"+file,{"data":data,"valence_labels":valence_labels,"arousal_labels":arousal_labels,"av_labels":av_labels})
