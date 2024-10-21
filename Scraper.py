from bs4 import BeautifulSoup
import requests

t = requests.get("https://uppolice.gov.in/pages/en/topmenu/police-units/dg-police-hqrs/en-dgp-up-circulars")

soup = BeautifulSoup(t.text, 'html.parser')

pdfs=[]

for link in soup.find_all('a'):
  if link.get('href')[-3:] == 'pdf' and link.get('href')[:4] == 'site':
    if '2024' in link.get('href'):
      pdfs.append("https://uppolice.gov.in/"+link.get('href'))

for p in pdfs:
    response = requests.get(p)
    filename = p.split('/')[-1]
    with open(filename, 'wb') as pdf_file:
        pdf_file.write(response.content)

