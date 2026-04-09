import requests

def upload_file(file_path):
    # Get best gofile server
    server_req = requests.get('https://api.gofile.io/servers')
    server = server_req.json()['data']['servers'][0]['name']
    
    # Upload
    url = f"https://{server}.gofile.io/uploadFile"
    response = requests.post(url, files={'file': open(file_path, 'rb')})
    print(response.json()['data']['downloadPage'])

if __name__ == "__main__":
    upload_file(r"C:\Users\Pooja malusare\Desktop\Masters_FL_Project.zip")
