# MPD-DF

**01 What is MPD-DF？**

The Multimodal Phenotyping Dataset of Driving Fatigue ** (MPD-DF)** is a publicly available dataset collected from 50 participants through standardized 2-hour driving simulation experiments. This meticulously curated resource integrates multidimensional subjective and objective metrics for comprehensive fatigue assessment, featuring multimodal physiological recordings including:
- 32-channel electroencephalogram (EEG)
- Single-lead electrocardiogram (ECG)
- Dual-channel electrooculogram (EOG)
- Thoracic respiratory effort signals
The supplementary metadata incorporates:
- Fatigue-associated questionnaire assessment results
- Expert physician-annotated fatigue level evaluations

**02 What is the function of DataAlign.py?**

We have open-sourced the preprocessing code named “DataAlign.py” to ensure reproducibility and to facilitate further research. “DataAlign.py” processes MPD-DF data from the Raw Dataset to Preprocessed Data.The key functionalities of the Python script include: 1) signal extraction: isolating EEG, ECG, EOG, and respiratory effort signals from raw data files; 2) temporal alignment: synchronizing signals based on timestamps to ensure temporal consistency; and 3) annotation mapping, which aligns physician-annotated fatigue labels with the corresponding physiological data files. In Python 3.8, the code relied on standard scientific libraries for signal processing and file management.

**Detailed function implementation：** This function preprocesses raw EEG and PSG data along with their annotation files. It extracts ECG, EOG, and respiratory effort (RES) signals from the PSG data and aligns EEG, ECG, EOG, and RES signals with their corresponding labels on a per-second basis. The processed data is saved in .mat format, with signal data stored as time length (in seconds) × number of channels × sampling rate and annotation data stored as 1 × time length (in seconds). After downloading the publicly available raw data, the basic path is assigned to the variable path, and the output directory is assigned to prepath. The processed data are subsequently organized and saved into 50 individual folders, each labeled with a volunteer ID (e.g., 01, 02, …, 50). Each folder contains the corresponding data for EEG, ECG, EOG, and RES.
