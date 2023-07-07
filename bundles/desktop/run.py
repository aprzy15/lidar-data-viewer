import multiprocessing

from lidar_data_viewer.app.main import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main(exec_mode="desktop")
