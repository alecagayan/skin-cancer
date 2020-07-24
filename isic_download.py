from isic_api import ISICApi

api = ISICApi(username="aagayan", password="abcd1234*")
#imageList = api.getJson('image?limit=10&offset=2000&sort=name')

studyList = api.getJson('study?limit=50')

print(studyList)