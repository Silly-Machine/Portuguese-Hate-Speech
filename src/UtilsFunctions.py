import requests 
import zipfile
import os

def download_embeddings(url,embedding,path ="data", chunk_size=128):
    #Download embedding
    savepath = f"{path}/pretrained-{embedding}"
    os.makedirs(savepath) if not os.path.exists (savepath) else "ok"
    
    zipath = f"{path}/pretrained-{embedding}/pretrained-{embedding}.zip"
    response = requests.get(url, stream=True)
    #Save embedding
    with open(zipath, 'wb') as fd:
        for chunk in response.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
    #Unzip embedding
    with zipfile.ZipFile(zipath, 'r') as zip_ref:
        zip_ref.extractall(savepath)
    #Delte zipfile
    os.remove(zipath)