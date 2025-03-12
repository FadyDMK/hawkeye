import camera_calibrate
import os

def main():
    # Get the absolute path to the image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, '../data/blenderSim/ExtractedFrames/frameL2new.png')
    cc = camera_calibrate.CameraCalibrate(image_path)


if __name__ == '__main__':
    main()
