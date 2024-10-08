import json
import gzip
import re
import argparse as ap

import unicodedata as ud

# Function to remove nikkud (Hebrew vowel marks)
def remove_nikkud(text, keep_end=False):
    # Define the regex pattern
    if keep_end:
        return re.sub(r'[\u0591-\u05AF\u05B0-\u05BD\u05BF\u05C1-\u05C2\u05C4-\u05C7]', '', text)
    return re.sub(r'[\u0591-\u05C7]', '', text)

def remove_accents(s):
    return ''.join((c for c in ud.normalize('NFD', s) if ud.category(c) != 'Mn'))

# Process each line of the JSON file
def process_json_line(json_line):
    # Parse the JSON line into a dictionary
    data = json.loads(json_line)
    
    # Create the new dictionary with the required fields
    if args.lang == "Greek":
        processed_data = {
            "location": data["eng_ref"].split('(')[0].split('{')[0].split('[')[0],  # Copy the "heb_ref" field as "location"
            "text": remove_accents(data["text"])  # Remove Nikkud from the "line" field and store as "text"
        }
    elif args.lang == "Hebrew":

        processed_data = {
            "location": data["eng_ref"],  # We're going to keep everything consistent by using English references
            "text": remove_nikkud(data["line"], keep_end=True)  # Remove Nikkud from the "line" field and store as "text"
        }
    
    return processed_data

# Main function to process the input JSON file and output a compressed JSON.gz file
def process_json_file(input_file, output_file, lang):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'wt', encoding='utf-8') as outfile:
        # Iterate over each line in the input file
        for line in infile:
            if line.strip():  # Skip empty lines
                processed_data = process_json_line(line.strip())
                # Write the processed data to the output file as JSON
                json.dump(processed_data, outfile, ensure_ascii=False)
                outfile.write('\n')

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--lang")
    args = parser.parse_args()
    process_json_file(args.input, args.output, args.lang)

