import os

from sys import argv

from icrawler import ImageDownloader
from icrawler.builtin.google import GoogleFeeder, GoogleParser, GoogleImageCrawler

try:
    n_img = int(argv[2])
except (ValueError, IndexError):
    n_img = 100

try:
    n_thr = int(argv[3])
except (ValueError, IndexError):
    n_thr = 8

crawler = GoogleImageCrawler(GoogleFeeder, GoogleParser, ImageDownloader,
                                1, 1, n_thr,
                                dict(
                                    backend='FileSystem',
                                    root_dir=os.path.join('data', argv[1].replace(' ', '_'))
                                ))
crawler.crawl(argv[1], 0, n_img)
