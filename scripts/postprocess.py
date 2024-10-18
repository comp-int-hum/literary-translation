import json
import re
import argparse as ap
from utils import Location


def load_json_lines(file_path):
    """Load JSON lines from a file into a dictionary keyed by location."""
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            location = Location(item['location'])  # Convert to string for consistent keys
            data[location] = item['text']
    return data



if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--inputs", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--src")
    parser.add_argument("--tgt")

    args = parser.parse_args()
    # Example usage
    print(f"loading {args.inputs[0]}")
    data = load_json_lines(args.inputs[0])
    # data = {}
    # for input in args.inputs:
    #     if "_4" in input:
    #         data.update(load_json_lines(input))
    #
    # for input in args.inputs:
    #     if "_5" in input:
    #         data.update(load_json_lines(input))
    #
    # for input in args.inputs:
    #     if "_6" in input:
    #         data.update(load_json_lines(input))
    #
    # for input in args.inputs:
    #     if "_7" in input:
    #         data.update(load_json_lines(input))

    # sanity check
    args.src = args.src.replace('_', ' ')
    outputs = []
    with open(args.output, 'wt') as ofd:
       
        #pattern = r'5\s*\. SRC:.*?\n\s*TGT:\s*(?:"([^"]+)"|(.{30,}?)\s*")'.replace('SRC', args.src).replace('TGT', args.tgt)
        pattern = r'5\s*\. SRC:.*?\n\s*TGT:\s*"([^"]*)'.replace('SRC', args.src).replace('TGT', args.tgt)

        # print(pattern)
        # exit()

        for i, (k,v) in enumerate(data.items()):
    
             
            # pattern = r'5\s*\. SRC:.*?\n\s*TGT:\s*(?:"([^"]+)"|(.{30,}?)\s*")'.replace('SRC', args.src).replace('TGT', args.tgt)
 
            #pattern = r'3\s*\. SRC:.*?\n\s*TGT:\s*"([^"]*)'.replace('SRC', args.src).replace('TGT', args.tgt)
            full_string = re.compile(pattern)
            match = re.search(full_string, v)
            #print(match)
            #match = re.search(r'3\s*\. Ancient Greek:.*?\n\s*English:\s*"([^"]*)', v)
            if match:
                translation = match.group(1)
            else:
                translation = None
                print("None")
            
            ofd.write(json.dumps({"location": k, "text": translation}, ensure_ascii=False)+ "\n")

    


