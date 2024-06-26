# Compressed Sensing for 2d

This repository is used to create $X$, $X_{reconstructed}$ and $Y$ values which are spectrograms inputs to a neural network algorithm. 
$X$ represent the original spectrograms used for the baseline experiments and $X_{reconstructed}$ created from the compressed data using compressed sensing techniques.

Make sure you have additional folder contains Audio and Annotations. Annotated files (.svl) and the corresponding audio file (.wav) are read. 
A number of segments are extracted to create the original dataset. A random sampling in compressed sensing is applied to $X$ and give the compressed data as $X_{compressed}$, followed by recovery algorithm to get the final data $X_{reconstructed}$.
      
### Folder structure

                  |── Audio
                  |── Annotations
                  |── DataFiles/Training.txt
                  ├── main.py
                  ├── Preprocess_CS_2D.py
                  ├── AnnotationReade.py
                  |── CS.py
                  |── Settings.json  
                  └── CS_2D_output/
                  
### Prerequisites

Install dependencies in your terminal:

      $ pip install -r requirements.txt             

### Usage
1. Download or clone this repository in your terminal
  
        $ git clone https://github.com/milanto-hery/cs_project.git
    
2. Go to the directory containing the the scripts

         $ cd cs_project/
   
4. Download the required data from the following links to make sure you have all the requirements. (You can have the gibbon or thyolo). Extract the three folders Audio, Annotations, and DataFiles and put them inside the cs_project folder:
      - Hainan Gibbon: [Audio](https://drive.google.com/drive/folders/1xkFkqMIdceuwtHJzhvN3iooxIBVlbizf?usp=drive_link) , [Annotations](https://drive.google.com/drive/folders/1i_NRYObfRkUFPM9--brynGqJx9DQr5nV?usp=drive_link), [DataFiles](https://drive.google.com/drive/folders/1MfkSGr-U2PxwTPVByhy6Qj_1HIge0uG8?usp=drive_link)
      - Thyolo alethe: [Audio](https://drive.google.com/drive/folders/1ACsEKIUZz57mLoNKQm2n_d1HlO2uLcCa?usp=drive_link), [Annotations](https://drive.google.com/drive/folders/1XGeZOmrWyQdC4_435ZU9aQkzM4IH96KV?usp=drive_link), [DataFiles](https://drive.google.com/drive/folders/1-OIMR-GcVCx1Lqp4xOWy5Mhfe7RzXRve?usp=drive_link)


6. Run the main script
  
        $ python3 main.py
  
7. The outputs automatically saved in the specified folder name (It should be pickle and zip files)


         
