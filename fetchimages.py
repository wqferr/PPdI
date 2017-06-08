import os

from sys import argv

from icrawler import ImageDownloader
from icrawler.builtin.google import GoogleFeeder, GoogleParser, GoogleImageCrawler

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

    crawler = GoogleImageCrawler(GoogleFeeder, GoogleParser, ImageDownloader,
            1, 1, n_thr,
            dict(
                backend='FileSystem',
                root_dir=path
            ))

    crawler.crawl(keyword, 0, n_img)
