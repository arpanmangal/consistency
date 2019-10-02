import wget

url = 'https://raw.githubusercontent.com/coin-dataset/annotations/master/COIN.json'
json_path = '../../data/coin/COIN_full.json'

filename = wget.download(url, out=json_path)

print ("\nAnnotations Downloaded") 

