# static-ORB-extractor : SORBE
Finds static ORB features in a video(excluding the dynamic objects), typically for a SLAM scenario

# Requirements
OpenCV 3

# Build Instructions
Clone the repository:
```
git clone https://github.com/srinath2022/static-ORB-extractor.git
```

We provide a script `build.sh` to build. Please make sure you have installed all required dependencies. Execute:
```
cd static-ORB-extractor
chmod +x build.sh
./build.sh
```

This will create directories **out**, **lib**, **build** and executable **sorbe** will be placed in **out** directory.

# Running
Execute the following command.
```
./out/sorbe PATH_TO_SEQUENCE_FOLDER 1 TUM
```