import argparse
import csv
import logging
import os
import re
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial

import cv2
import numpy as np
import pytesseract

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_speed_mph(frame):
    cropped_frame = frame[1035:1080, 0:1920]
    hsv_cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 0], dtype=np.uint8)
    upper_white = np.array([0, 0, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv_cropped_frame, lower_white, upper_white)
    white_text_on_black_frame = cv2.bitwise_and(cropped_frame, cropped_frame, mask=mask)
    black_text_on_white_frame = cv2.bitwise_not(white_text_on_black_frame)

    rgb_black_text_on_white_frame = cv2.cvtColor(black_text_on_white_frame, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(rgb_black_text_on_white_frame)
    speed_mph = re.search(r"(?P<speed_mph>[0-9]+)\sMPH", text)

    if not speed_mph:
        return 0
    else:
        return int(speed_mph.group('speed_mph'))


def get_flow(frame_1, frame_2):
    grey_frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
    grey_frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(
        grey_frame_1, grey_frame_2, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )


def process_video(input_path, output_path, file_name):
    try:
        csv_file = open(os.path.join(output_path, "dataset.csv"), 'a+')
        csv_writer = csv.writer(csv_file)

        video_full_path = os.path.join(input_path, file_name)
        video_name = file_name.lower().replace('.mp4', '')

        stream = cv2.VideoCapture(video_full_path)
        frame_num = 0
        _, left_frame = stream.read()

        while True:

            right_frame_success, right_frame = stream.read()
            frame_num += 1

            if not right_frame_success:
                break

            speed_mph = get_speed_mph(right_frame)

            # Try to catch outliers by making sure that the difference between two frames is < 100 mph.
            if speed_mph > 100:
                frame_num += 1
                left_frame = right_frame
                continue

            # left_frame[y, x] == right_frame[y + flow[y, x, 1], x + flow[y, x, 0]]
            flow = get_flow(left_frame, right_frame)

            hsv = np.zeros_like(right_frame)
            hsv[..., 1] = 255
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            LOGGER.info(f"Got {speed_mph} MPH for {file_name} frame {frame_num}.")
            flow_name = f"flow_{video_name}_{frame_num}.jpg"
            flow_path = os.path.join(output_path, flow_name)
            cv2.imwrite(flow_path, flow)

            csv_writer.writerow([flow_name, speed_mph])

            left_frame = right_frame

        LOGGER.info(f"Finished processing {file_name}.")
        csv_file.close()
    except Exception as e:
        LOGGER.exception("Uh oh.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="The directory containing the input videos.", required=True)
    parser.add_argument("-o", "--output", type=str, help="The directory that the output data should be written to.",
                        required=True)

    args = parser.parse_args()

    videos = os.listdir(args.input)
    videos = [video for video in videos if os.path.isfile(os.path.join(args.input, video))]

    if videos:
        os.makedirs(args.output, exist_ok=True)

    with ThreadPoolExecutor(max_workers=16) as executor:
        executor.map(partial(process_video, args.input, args.output), videos)
