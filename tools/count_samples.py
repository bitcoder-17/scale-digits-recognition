from pathlib import Path
import json
from argparse import ArgumentParser
from collections import Counter
import pprint


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('input_dir', type=str)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    json_paths = list(input_dir.glob('*.json'))
    counter = Counter()
    for json_path in json_paths:
        json_dict = json.load(open(json_path, 'rt'))
        if len(json_dict['shapes']) == 0:
            print(f'Skip {json_path} since it contains no box')
            continue

        for shape in json_dict['shapes']:
            label = shape['label']
            counter[label] = counter.get(label, 0) + 1

    pprint.pprint(dict(counter), indent=4)
