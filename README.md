# Self-Driving Car Simulation 🚘

A real-time simulation of an autonomous vehicle’s core logic using computer vision and deep learning. This project detects objects, tracks movement, identifies lanes, decides steering actions, and provides visual feedback through an interactive UI.

## Features

- Real-time object detection using YOLOv11  
- Object tracking with SORT  
- Lane detection using Hough Transform  
- Steering wheel rotation based on lane direction  
- Proximity-based collision alerts  
- Trail and path visualization  
- FPS counter and modular design  

## Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/Al-Pa-Na/Self-Driving-Car.git
cd Self-Driving-Car
pip install -r requirements.txt
````

Place the `yolo11n.pt` model file in the root directory.

## Usage

Run the project:

```bash
python main.py
```

To use a webcam instead of a video file, set `use_webcam = True` in `main.py`.

## Repository Overview

```
main.py                 Entry point  
utils/                  Utility modules  
├── draw.py  
├── lane_detection.py  
├── path_visualizer.py  
├── proximity.py  
├── steering_overlay.py  
├── tracker.py  
yolo/  
└── yolo_detector.py    YOLO wrapper  
assets/  
├── images/             Steering overlay  
└── videos/             Sample videos and logs  
```

## Notes

* Sample outputs are saved in `output.avi` (ignored in git)
* Log of tested videos: `assets/videos/video_testing_notes.txt`
* Fully modular and built from scratch for simulation and experimentation

## Roadmap

* Lane departure warning system
* Speed estimation module
* Traffic sign detection and classification
