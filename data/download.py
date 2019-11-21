import requests
import os
import glob

def download_file(URL, destination):
    session = requests.Session()
    response = session.get(URL, stream = True)

    save_response_content(response, destination)    


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


print('Downloading dataset...')
if not os.path.isfile('data/hmdb51_org.rar'):
    download_file('http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar', 'data/hmdb51_org.rar')

if not os.path.isdir('data/video'):
    os.makedirs('data/video')
    
os.system('unrar e data/hmdb51_org.rar data/video')

filenames = glob.glob('data/video/*.rar')
for file_name in filenames:
    os.system('unrar x %s data/video' %file_name)
