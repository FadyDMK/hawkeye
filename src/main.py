import os
from tracemalloc import start
import tkinter as tk
from front_end import FrameSelectorApp
from sklearn import pipeline
from hawkeye_pipeline import HawkeyePipeline

def main():
    '''To run the pipeline on a video, uncomment the following lines.'''
    # pipeline = HawkeyePipeline(None)

    # pipeline.process_video(start_frame=0, end_frame=146)

    # output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    # os.makedirs(output_dir, exist_ok=True)
    # pipeline.export_results(output_dir)

    # print("\n Showing 3D visualization")
    # pipeline.visualize_results(type="3d")

    # print("\n Showing 2D visualization")
    # pipeline.visualize_results(type="2d")

    ''' Run the pipeline on a single frame with frontend GUI '''
    root = tk.Tk()
    app = FrameSelectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()