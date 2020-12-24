import argparse
import os
import random
import logging

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-t", "--training", type=str,required=True)
    parser.add_argument("-v", "--validation", type=str, required=True)
    parser.add_argument("-s", "--split", type=float, required=True)
    args = parser.parse_args()

    for file in os.listdir(path=args.input):
        rand_int = random.randint(0, 100)
        if rand_int < args.split:
            os.makedirs(args.training, exist_ok=True)
            os.rename(os.path.join(args.input, file), os.path.join(args.training, file))
            LOGGER.info(f"Moved {file} to training.")
        else:
            os.makedirs(args.validation, exist_ok=True)
            os.rename(os.path.join(args.input, file), os.path.join(args.validation, file))
            LOGGER.info(f"Moved {file} to validation.")