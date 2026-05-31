# Bird_or_ship

## Setup & Usage
Run the following commands in order:

```bash
chmod +x setup.sh
./setup.sh
py scripts/bird_or_sheep.py   # trains and saves the model
py scripts/activate_model.py  # loads the model and classifies the target image 
```

> To change the image being classified, update the path at the end of `activate_model.py`.

## Dataset Structure
```
dataset/
├── bird/
│   ├── bird_image_1.jpg
│   └── ...
└── sheep/
    ├── sheep_image_1.jpg
    └── ...
```

## Expected Output
```
This image is most likely a X with Y% confidence.
```