from bs4 import BeautifulSoup as BS
import re


set_names = [f'articles/set{i}' for i in range(1, 5)]
article_names = [f'a{i}.htm' for i in range(1, 10)]


def read_article(set_name, article_name):
    file_name = f'{set_name}/{article_name}'
    with open(file_name) as file_handler:
        index = file_handler.read()
        parser = BS(index, "lxml")
    text = ''
    for paragraph in parser.find_all('p'):
        text += paragraph.text + ' '
    text = re.sub(r'\[.*?\]+', '', text[:-1])
    text = text.replace('\n', '')

    return text


texts = [[read_article(s, a) for a in article_names] for s in set_names]