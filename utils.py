import json

def rround(num, decimals=3):
    return round(num, decimals)

def read_json_lines(filepath):
    # each line in the file is a json object
    with open(filepath, 'r') as f:
        lines = f.readlines()
        # strip any trailing whitespace and remove empty lines
        lines = [line.strip() for line in lines if line.strip()]
        lines = [json.loads(line) for line in lines]
    return lines


def write_json_lines(filepath, data):
    with open(filepath, 'w', encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=True) + '\n')


# read a file and return a list of lines
def read_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        # strip any trailing whitespace and remove empty lines
        lines = [line.strip() for line in lines if line.strip()]
    return lines

# write lines to a file
def write_lines(filepath, lines):
    with open(filepath, 'w', encoding="utf-8") as f:
        for line in lines:
            f.write(line + '\n')


# caculate sum of all durations from a given manifest file
def get_total_duration(manifest_file):
    lines = read_json_lines(manifest_file)
    total_duration = 0  # in seconds
    for line in lines:
        total_duration += float(line['duration'])
    
    return total_duration