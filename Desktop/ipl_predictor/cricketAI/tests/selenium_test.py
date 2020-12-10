import os
import sys
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

def get_driver():
    options = Options()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    return driver

def get_element_text(HTTPS, BY, VALUE):
    driver = get_driver()
    driver.get(HTTPS)
    element = driver.find_element(BY, VALUE)
    return element.text

class Test(unittest.TestCase):
    def test_element_text(self):
        HTTP_MAIN = 'http://127.0.0.1:8000/main'
        HTTP_ABOUT = 'http://127.0.0.1:8000/about'
        self.assertEqual(get_element_text(HTTP_MAIN, By.CLASS_NAME, 'w3ls-top'), 'Delhi Capitals')
        self.assertEqual(get_element_text(HTTP_ABOUT, By.CLASS_NAME, 'w3l_header'), 'About Us')

if __name__ == '__main__':
        # unittest.main(argv=['ignored', '-v'], exit=False)
        unittest.main()
