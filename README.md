# Compressed Sensing for 2d

This repository is used to create X, X_reconstructed and Y values which are spectrograms inputs to a ML algorithm. 
X represent the original spectrograms used for the baseline experiments and X_reconstructed created from the compressed data using compressed sensing techniques.
Annotated files (.svl) and the corresponding audio file (.wav) are read. A number of segments are extracted to create the original dataset. 
A random sampling in compressed sensing is applied and give the compressed data, followed by recovery algorithm to get the final reconstruct data.


Make sure you have additional folder contains Audio and Annotations. Annotated files (.svl) are read and the corresponding audio file (.wav) is read. 
A number of segments are extracted and augmented to create the final dataset. Annotated files (.svl) are created using SonicVisualiser and it is assumed that the "boxes area" layer was used to annotate the audio files.
      
### Folder structure:

                  |── Audio
                  |── Annotations
                  |── DataFiles/Training.txt
                  ├── main.py
                  ├── Preprocess_CS_2D.py
                  ├── Preprocessing.py
                  |── CS.py
                  |── Settings.json  
                  └── CS_2D_output/
                  
### Prerequisites

Install dependencies in your terminal:

$ pip install -r requirements.txt             

### How it works:
- Download or clone this repository in your terminal
- Download the required data from this link to make sure you have all the requirements (You can have the Gibbon or Thyolo)
- Extract the three folder Audio, Annotations and DataFiles and put inside this repository in your terminal
- Run the main script
  
  $ pythonmain.py
  
- The outputs automatically saved in the specified folder name (It should be pickle and zip files)


         
