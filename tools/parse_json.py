#!/usr/bin/env python3
import json
import os
import argparse

def update_im_names(data):
    """
    Append .jpg to im_name entries in the images list if missing.
    """
    for img in data.get('images', []):
        name = img.get('im_name', '')
        if not name.lower().endswith('.jpg'):
            img['im_name'] = name + '.jpg'
    return data


def main():
    parser = argparse.ArgumentParser(description="Append .jpg to im_name fields in COCO-style JSON annotations.")
    parser.add_argument('input', help="Path to input JSON annotation file")
    parser.add_argument('output', nargs='?', help="Path to output JSON file (default: overwrite input)")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path

    if not os.path.isfile(input_path):
        print(f"Error: file '{input_path}' does not exist.")
        return

    # Load JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Update im_name fields
    data = update_im_names(data)

    # Save updated JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved updated annotation to '{output_path}'")

if __name__ == '__main__':
    main()
