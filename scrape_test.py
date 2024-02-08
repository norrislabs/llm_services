from urllib.request import urlopen
from bs4 import BeautifulSoup
import pyfiglet


def remove_short_lines(text):
    # Split the text into lines
    lines = text.split('\n')

    # Filter out lines shorter than 30 characters
    filtered_lines = [line for line in lines if len(line) >= 30]

    # Join the filtered lines back together
    cleaned_text = '\n'.join(filtered_lines)

    return cleaned_text


print(pyfiglet.figlet_format("Scrape"))

url = "https://www.cnn.com/2024/02/06/business/oakland-crime-business/index.html"
html = urlopen(url).read()
soup = BeautifulSoup(html, features="html.parser")

# kill all script and style elements
for script in soup(["script", "style"]):
    script.extract()  # rip it out

for a in soup.findAll('a', href=True):
    a.extract()

# get text
page_text = soup.get_text()

# break into lines and remove leading and trailing space on each
page_lines = (line.strip() for line in page_text.splitlines())

# break multi-headlines into a line each
chunks = (phrase.strip() for line in page_lines for phrase in line.split("  "))

# drop blank lines
page_text = '\n'.join(chunk for chunk in chunks if chunk)

clean_text = remove_short_lines(page_text)

print(clean_text)
