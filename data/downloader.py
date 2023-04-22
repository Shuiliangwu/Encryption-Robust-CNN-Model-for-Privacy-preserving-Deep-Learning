import json
import urllib.request

# Assuming that the JSON data is stored in a file called 'data.json'
with open('1k.json') as f:
    json1k = json.load(f)

data1k = []
for p in json1k.values():
    data1k.append(p[0])


data21 = []
with open('21k.txt', 'r') as f21:
    for line in f21:
        line = line.split('\n')[0]
        if not line in data1k:
            data21.append(line)

count = 0
fileurl = "https://image-net.org/data/winter21_whole/"
for synset in data21:
    response = urllib.request.urlopen(fileurl + synset + ".tar")
    size = int(response.headers.get('Content-Length'))/1024/1024  # in MB
    if (size > 130 and count < 100):
        print("downloading "+synset+" file size is ", size, "MB")
        count += 1
        urllib.request.urlretrieve(fileurl + synset + ".tar", synset + ".tar")
