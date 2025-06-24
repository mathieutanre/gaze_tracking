# Choose Camera

## Overview

This project integrates two main components: **Lights-ASD** and **Opal23**. The goal is to capture video and audio streams from a webcam, detect the active speaker, and then determine the yaw angle of the speaker's head using head pose estimation.

### Project Structure

```
/MonProjet
├── /images_framework
│   ├── realtime.py               # Script for capturing video and audio, and calculating active speaker probabilities
│   ├── /demo                     # Folder containing the results of the realtime.py script
│   │   ├── scores.pckl           # Pickle file storing the active speaker probability scores
│   │   └── /pycrop               # Folder containing cropped face videos of detected individuals
│   │       ├── cropped_video1.avi
│   │       ├── cropped_video2.avi
│   │       └── ...
├── /Opal23                       # Folder containing the head pose estimation algorithm
│   ├── headpose.py               # Script for estimating the yaw angle of the head
├── main.py                       # Main script to orchestrate the entire pipeline
└── README.md                     # This file
```

### Pipeline

1. **Capture Video and Audio**: 
   - The `realtime.py` script in the `Lights-ASD` folder captures the video and audio streams from the webcam. It computes the active speaker probability scores and crops detected faces, saving them in the `demo/pycrop` folder.

2. **Analyze Active Speaker Scores**:
   - After running `realtime.py`, the active speaker scores are saved in `scores.pckl`. The main script processes these scores, calculates the average for each detected individual, and selects the video with the highest score.

3. **Head Pose Estimation**:
   - The selected video is then passed to the `headpose.py` script in the `images_framework` folder to determine the yaw angle of the active speaker's head.

### Usage

To run the full pipeline:

1. **Run Lights-ASD**:
   ```bash
   python Lights-ASD/realtime.py --videoFolder demo
   ```

2. **Run the Main Script**:
   ```bash
   python main.py
   ```
