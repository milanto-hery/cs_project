# Import Libraries

import os
import numpy as np
import librosa.display
import librosa
import pickle
from matplotlib import pyplot as plt
import pandas as pd
import time
import gc
import concurrent.futures
import zipfile 
from zipfile import ZipFile
from os.path import basename
from AnnotationReader import *
from CS import *

class Preprocess_CS_2D:
    
    def __init__(self, species_folder, segment_duration,
                 positive_class, background_class,            
                 n_fft, hop_length, n_mels, f_min, f_max,file_type, audio_extension, solver, compression_rate):
        self.species_folder = species_folder
        self.solver = solver
        self.compression_rate = compression_rate
        self.segment_duration = segment_duration
        self.positive_class = positive_class
        self.background_class = background_class
        self.audio_path = self.species_folder + '/Audio/'
        self.annotations_path = self.species_folder + '/Annotations/'
        self.saved_data_path = self.species_folder + '/Saved_Data/'
        self.training_files = self.species_folder + '/DataFiles/TrainingFiles.txt'
        self.n_ftt = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.file_type = file_type
        self.audio_extension = audio_extension
        
    def update_audio_path(self, audio_path):
        self.audio_path = self.species_folder + '/'+ audio_path
        
    def read_audio_file(self, file_name):
        '''
        file_name: string, name of file including extension, e.g. "audio1.wav"
        
        '''
        # Get the path to the file
        audio_folder = os.path.join(file_name)
        
        # Read the amplitudes and sample rate
        audio_amps, audio_sample_rate = librosa.load(audio_folder, sr=None)
        
        return audio_amps, audio_sample_rate
    

    def convert_single_to_image(self, audio):
        '''
        Convert amplitude values into a mel-spectrogram.
        '''
        S = librosa.feature.melspectrogram(audio, n_fft=self.n_ftt,hop_length=self.hop_length, 
                                           n_mels=self.n_mels, fmin=self.f_min, fmax=self.f_max)
        
        image = librosa.core.power_to_db(S)
        image_np = np.asmatrix(image)
        image_np_scaled_temp = (image_np - np.min(image_np))
        image_np_scaled = image_np_scaled_temp / np.max(image_np_scaled_temp)
        mean = image.flatten().mean()
        std = image.flatten().std()
        eps=1e-8
        spec_norm = (image - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = (spec_norm - spec_min) / (spec_max - spec_min)
        S1 = spec_scaled
        
        return S1

    def convert_all_to_image(self, segments):
        '''
        Convert a number of segments into their corresponding spectrograms.
        '''
        spectrograms = []
        for segment in segments:
            spectrograms.append(self.convert_single_to_image(segment))

        return np.array(spectrograms)
    
    # Function to compress and reconstruct a single spectrogram.
    def compress_and_reconstruct(self, spectrogram, labels):
        
        cs = CS()
        width, height = spectrogram.shape
        compressed, sensing_matrix = cs.compress(spectrogram, self.compression_rate, 42)
        reconstructed = cs.reconstruct(compressed, sensing_matrix, width, height, self.solver)
        
        return np.asarray(spectrogram), np.asarray(compressed), np.asarray(reconstructed), np.asarray(labels)
        
    def getXY(self, audio_amplitudes, sample_rate, start_sec, annotation_duration_seconds, label, verbose):
        '''
        Extract a number of segments based on the user-annotations.
        If possible, a number of segments are extracted provided
        that the duration of the annotation is long enough. The segments
        are extracted by shifting by 1 second in time to the right.
        Each segment is then augmented a number of times based on a pre-defined
        user value.
        '''

        if verbose == True:
            print ('start_sec', start_sec)
            print ('annotation_duration_seconds', annotation_duration_seconds)
            print ('self.segment_duration ', self.segment_duration )
            
        X_segments = []
        Y_labels = []
            
        # Calculate how many segments can be extracted based on the duration of
        # the annotated duration. If the annotated duration is too short then
        # simply extract one segment. If the annotated duration is long enough
        # then multiple segments can be extracted.
        if annotation_duration_seconds-self.segment_duration < 0:
            segments_to_extract = 1
        else:
            segments_to_extract = annotation_duration_seconds-self.segment_duration+1
            
        if verbose:
            print ("segments_to_extract", segments_to_extract)
            
        if label in self.background_class:
            if segments_to_extract > 10:
                segments_to_extract = 10

        for i in range (0, segments_to_extract):
            if verbose:
                print ('Semgnet {} of {}'.format(i, segments_to_extract-1))
                print ('*******************')
            # The correct start is with respect to the location in time
            # in the audio file start+i*sample_rate
            start_data_observation = start_sec*sample_rate+i*(sample_rate)
            # The end location is based off the start
            end_data_observation = start_data_observation + (sample_rate*self.segment_duration)
            
            # This case occurs when something is annotated towards the end of a file
            # and can result in a segment which is too short.
            if end_data_observation > len(audio_amplitudes):
                continue

            # Extract the segment of audio
            X_audio = audio_amplitudes[start_data_observation:end_data_observation]

            # Determine the actual time for the event
            start_time_seconds = start_sec + i

            if verbose == True:
                print ('start frame', start_data_observation)
                print ('end frame', end_data_observation)
            
            # Extend the augmented segments and labels (and the metadata)
            X_segments.append(X_audio)
            Y_labels.append(label)

        return X_segments, Y_labels
        
    def create_dataset(self, max_threads, batch_size, saved_segments, verbose):
        '''
        Create X and Y values which are inputs to a ML algorithm.
        Annotated files (.svl) are read and the corresponding audio file (.wav)
        is read. A number of segments are extracted and augmented to create the final dataset.
        Annotated files (.svl) are created using SonicVisualiser and it is assumed
        that the "boxes area" layer was used to annotate the audio files.
        '''

        # Initialise lists to store the X and Y values
        X_calls = []
        Y_calls = []
          
        if verbose == True:
            print ('Annotations path:',self.annotations_path+"*.svl")
            print ('Audio path',self.audio_path+"*.WAV")
        
        # Read all names of the training files
        training_files = pd.read_csv(self.training_files, header=None)
        
        # Iterate over each annotation file
        for training_file in training_files.values:
            
            file = training_file[0]
            
            if self.file_type == 'svl':
                # Get the file name without paths and extensions
                file_name_no_extension = file
                #print ('file_name_no_extension', file_name_no_extension)
            if self.file_type == 'raven_caovitgibbons':
                file_name_no_extension = file[file.rfind('-')+1:file.find('.')]
                
            print ('Processing:',file_name_no_extension)
            
            reader = AnnotationReader(file, self.species_folder, self.file_type, self.audio_extension)

            # Check if the .wav file exists before processing
            print(self.audio_path+file_name_no_extension+self.audio_extension)
            if os.path.exists(self.audio_path+file_name_no_extension+self.audio_extension): 
                print('Found file')
                
                # Read audio file
                audio_amps, original_sample_rate = self.read_audio_file(self.audio_path+file_name_no_extension+self.audio_extension)
                print('Original sampling rate: ', original_sample_rate)

                df, audio_file_name = reader.get_annotation_information()

                print('Reading annotations...')
                
                print("done.\n")
                for start_index, row in df.iterrows():
     
                    start_seconds = int(round(row['Start']))
                    end_seconds = int(round(row['End']))
                    label = row['Label']
                    annotation_duration_seconds = end_seconds - start_seconds
                    # Extract augmented audio segments and corresponding binary labels
                    X_data, y_data = self.getXY(audio_amps, original_sample_rate, start_seconds, annotation_duration_seconds, label, verbose)
                  
                    # Append the segments and labels
                    X_calls.extend(X_data)
                    Y_calls.extend(y_data)
        
        # Convert to numpy arrays
        X_calls, Y_calls = np.array(X_calls), np.array(Y_calls)
        segments = X_calls
        store_original_spectrograms = []
        store_compressed_spectrograms = []
        store_reconstructed_spectrograms = []
        store_labels = []
        
        batch_times = []  # To store batch processing times.
        spec_times = []
                
        t_s = time.time()  
        # Split segments into batches and convert to spectrograms in a batch and run compression and reconstruction in parallel:\n')
        print(f'Run compression and reconstruction in parallel:\n')
        for start_index in range(0, len(segments), batch_size):
            end_index = start_index + batch_size
            batch_segments = segments[start_index:end_index]
            batch_labels = Y_calls[start_index:end_index]
            print(f"Batch index: ({start_index} -- {end_index})")
            t_spectrogram_s = time.time()
            batch_spectrograms = self.convert_all_to_image(batch_segments)
            t_spectrogram_e = time.time()
            t_time = t_spectrogram_e - t_spectrogram_s

            print(f'Converting to spectrograms, time: {t_time:.2f} seconds')
            print(f'Original shape: {batch_spectrograms.shape}')
            
            batch_start_time = time.time()
            # Create a thread pool with the desired threads.
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:        
                # Process in parallel the compression and reconstruction of each spectrogram in the batch.
                batch_output = list(executor.map(self.compress_and_reconstruct, batch_spectrograms, batch_labels))
                                    
                # Extract compressed and reconstructed spectrograms from the batch.
                original_spectrograms = [extract[0] for extract in batch_output]
                compressed_spectrograms = [extract[1] for extract in batch_output]
                reconstructed_spectrograms = [extract[2] for extract in batch_output]
                labels = [extract[3] for extract in batch_output]
                                        
                batch_end_time = time.time()

                batch_time = batch_end_time - batch_start_time
                single_spec_time = batch_time / len(batch_spectrograms)
                print(f"Compression and reconstruction time: {batch_time:.2f} seconds")               
                                        
                # Store the results for the current batch.
                store_original_spectrograms.extend(original_spectrograms)
                store_compressed_spectrograms.extend(compressed_spectrograms)
                store_reconstructed_spectrograms.extend(reconstructed_spectrograms)
                store_labels.extend(labels)
                spec_times.append(t_time)
                batch_times.append(batch_time)
                print("done.\n")  
                del batch_spectrograms
                gc.collect()
                    
        t_e = time.time()
        t_process = t_e-t_s  
        # Calculate the average time for batch processing.
        average_spec_time = sum(spec_times) / len(spec_times)
        average_batch_time = sum(batch_times) / len(batch_times)        
            
        average_batch_time = sum(batch_times) / len(batch_times)
        X_raw , X_comp, X_rec, Y_values = np.array(store_original_spectrograms), np.array(store_compressed_spectrograms), np.array(store_reconstructed_spectrograms), np.array(store_labels)

        # Average batch processing time.
        print("_______________________SUMARRY_________________________")
        print(f"Number of segments extracted: {len(X_calls)}")
        print(f"Average time to convert spectrograms: {average_spec_time:.2f} seconds")
        print(f"Average time to compress and reconstruct a batch: {average_batch_time:.2f} seconds")
        print(f'Processing time for the {X_raw.shape[0]} spectrograms: {t_process:.2f} seconds')
        print("done.\n")
        
        
        # Create a folder to save the data in array
        #---------------------------------------------------------------------------------------------------------------------------       
        if not os.path.exists(saved_segments):
            os.makedirs(saved_segments)


        pickle_file_0 = os.path.join(self.species_folder+'/'+saved_segments, f"X_raw_S.pkl")
        with open(pickle_file_0, 'wb') as file:
            pickle.dump(X_raw, file)


        pickle_file_1 = os.path.join(self.species_folder+'/'+saved_segments, f"X_compressed_S.pkl")
        with open(pickle_file_1, 'wb') as file:
            pickle.dump(X_comp, file)

            pickle_file_2 = os.path.join(self.species_folder+'/'+saved_segments, f"X_reconstructed_S.pkl")
        with open(pickle_file_2, 'wb') as file:
            pickle.dump(X_rec, file)

        pickle_file_3 = os.path.join(self.species_folder+'/'+saved_segments, f"Y_S.pkl")
        with open(pickle_file_3, 'wb') as file:
            pickle.dump(Y_values, file)

         # zip_compression level 6
        # --------------------------------------------------------------------------------------------------------------------------
            
        # --------------------------------------------------------------------------------------------------------------------------
        zip_location_0 = os.path.join(self.species_folder+'/'+saved_segments, f"X_raw_S"+".zip")
        text_file_0 = os.path.join(self.species_folder+'/'+saved_segments, f"X_raw_S"+".pkl") 
         
        with zipfile.ZipFile(zip_location_0, 'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zipObj:
            zipObj.write(text_file_0, basename(text_file_0))

        # ------------------------------------------------------------------------------------------------
        zip_location_1 = os.path.join(self.species_folder+'/'+saved_segments, f"X_compressed_S"+".zip")
        text_file_1 = os.path.join(self.species_folder+'/'+saved_segments, f"X_compressed_S"+".pkl") 
         
        with zipfile.ZipFile(zip_location_1, 'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zipObj:
            zipObj.write(text_file_1, basename(text_file_1))
        
        # -----------------------------------------------------------------------------------------------
        zip_location_2 = os.path.join(self.species_folder+'/'+saved_segments, f"X_reconstructed"+".zip")
        text_file_2 = os.path.join(self.species_folder+'/'+saved_segments, f"X_reconstructed_S"+".pkl") 
         
        with zipfile.ZipFile(zip_location_2, 'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zipObj:
            zipObj.write(text_file_2, basename(text_file_2))

        return None
        

