import cv2 as cv
import numpy as np
from PIL import ImageGrab
from pynput.mouse import Button, Controller, Listener
from pynput.keyboard import Key
import time

def target_capture(template, threshold=0.95):
    try:
        source = cv.imread(template)
        if source is None:
            raise FileNotFoundError(f"Template image '{template}' not found.")

        source_mat = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
        source_height, source_width = source_mat.shape[:2]

        mouse = Controller()

        while True:
            screenshot = ImageGrab.grab(bbox=None)
            screen_mat = np.array(screenshot)

            screen_mat = cv.cvtColor(screen_mat, cv.COLOR_BGR2GRAY)

            result = cv.matchTemplate(screen_mat, source_mat, cv.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            
            if max_val >= threshold:
                target_locations = []
                last_max_loc = None
                while True:
                    new_screenshot = ImageGrab.grab(bbox=None)
                    new_screen_mat = np.array(new_screenshot)
                    new_screen_mat = cv.cvtColor(new_screen_mat, cv.COLOR_BGR2GRAY)

                    new_result = cv.matchTemplate(new_screen_mat, source_mat, cv.TM_CCORR_NORMED)
                    _, _, _, new_max_loc = cv.minMaxLoc(new_result)

                    if new_max_loc not in target_locations:
                        target_locations.append(new_max_loc)
                        last_max_loc = new_max_loc
                    else:
                        center_x = int((new_max_loc[0] + source_width / 2))
                        center_y = int((new_max_loc[1] + source_height / 2))
                        break

                print(f"Target found at {center_x}, {center_y}.")
                mouse.position = (center_x, center_y)
                mouse.press(Button.left)
                mouse.release(Button.left)
                time.sleep(1)
    
            else:
                print("Match not found. Waiting...")
                time.sleep(1)

    except KeyboardInterrupt:
        print("Program interrupted.")

def duck_capture(templates: list, threshold=0.985):
    sources = [cv.imread(template) for template in templates]
    if any(source is None for source in sources):
        raise FileNotFoundError(f"One or more template images not found.")

    source_mats = [cv.cvtColor(source, cv.COLOR_BGR2GRAY) for source in sources]
    source_heights, source_widths = [source_mat.shape[:2] for source_mat in source_mats]

    mouse = Controller()

    while True:
        screenshot = ImageGrab.grab(bbox=None)
        screen_mat = np.array(screenshot)

        screen_mat = cv.cvtColor(screen_mat, cv.COLOR_BGR2GRAY)

        for source_mat, source_height, source_width in zip(source_mats, source_heights, source_widths):
            result = cv.matchTemplate(screen_mat, source_mat, cv.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            print(max_val)
            if max_val >= threshold:
                center_x = int((max_loc[0] + source_width / 2))
                center_y = int((max_loc[1] + source_height / 2))

                target_velocity_x = 2.5
                target_velocity_y = -1.0

                time_interval = 1.0
                x_predicted = center_x + int(target_velocity_x * time_interval)
                y_predicted = center_y + int(target_velocity_y * time_interval)

                print(f"Duck found at {center_x}, {center_y}.")
                print(f"Predicted position: ({x_predicted}, {y_predicted})")

                # Update mouse position
                mouse.position = (x_predicted, y_predicted)
                mouse.press(Button.left)
                mouse.release(Button.left)
                time.sleep(1)
                break
            else:
                print("Match not found. Waiting...")
                time.sleep(1)
    

if __name__ == '__main__':
    target_image_path = './target.png'
    ducks_image_paths = ['./duck_right_leg.png', './duck_left_leg.png']
    #duck_capture(ducks_image_paths)
    target_capture(target_image_path)