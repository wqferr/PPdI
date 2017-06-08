import os

from uuid import uuid4
from sys import argv
from six.moves.urllib.parse import urlparse

from icrawler import ImageDownloader
from icrawler.builtin.google import GoogleFeeder, GoogleParser, GoogleImageCrawler

class Downloader(ImageDownloader):
    def get_filename(self, task, default_ext):
        url_path = urlparse(task['file_url'])[2]
        extension = url_path.split('.')[-1] if '.' in url_path else default_ext
        return '{}.{}'.format(str(uuid4()), extension)

# Args:
# 1 - keyword
# 2 - number of images (default 100)
# 3 - number of threads (default 8)
# 4 - save directory (default data/datasets/fetched/<KEYWORD>)
if __name__ == '__main__':
    keyword = argv[1]

    try:
        n_img = int(argv[2])
    except (ValueError, IndexError):
        n_img = 100

    try:
        n_thr = int(argv[3])
    except (ValueError, IndexError):
        n_thr = 8

    try:
        path = argv[4]
    except IndexError:
        path = os.path.join('data', 'datasets', 'fetched', keyword)

    crawler = GoogleImageCrawler(GoogleFeeder, GoogleParser, Downloader,
            1, 1, n_thr,
            dict(
                backend='FileSystem',
                root_dir=path
            ))

    crawler.crawl(keyword, 0, n_img)
