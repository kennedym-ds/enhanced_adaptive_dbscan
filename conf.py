# docs/conf.py

import os
import sys
sys.path.insert(0, os.path.abspath('../enhanced_adaptive_dbscan'))

project = 'Enhanced Adaptive DBSCAN'
author = 'Michael Kennedy'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'alabaster'
html_static_path = ['_static']
