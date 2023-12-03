import gdown
import zipfile

def download_files():
    file_id = '1Hplha9HWYt-XtWn2Bw2mqOelKnmvEQaU'
    url = f'https://drive.google.com/uc?id={file_id}'
    test = 'test.zip'  # Replace with your desired output filename
    gdown.download(url, test, quiet=False)

    # Unzip the file
    with zipfile.ZipFile(test, 'r') as zip_ref:
        zip_ref.extractall('')
    
    
    file_id = '183-vEE-6K08JVeFlHsOnP3kwIeDHGU4U'
    url = f'https://drive.google.com/uc?id={file_id}'
    train = 'train.zip'  # Replace with your desired output filename
    gdown.download(url, train, quiet=False)

    # Unzip the file
    with zipfile.ZipFile(train, 'r') as zip_ref:
        zip_ref.extractall('')  # Replace with your desired extraction path