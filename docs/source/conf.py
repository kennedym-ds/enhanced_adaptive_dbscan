# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'enhanced_adaptive_dbscan'
copyright = '2024, Michael Kennedy'
author = 'Michael Kennedy'
release = 'November 2024'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	'sphinx.ext.autodoc',
	'sphinx.ext.napoleon',
	'sphinxcontrib.mermaid',
	'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# MyST (Markdown) configuration
source_suffix = {
	'.rst': 'restructuredtext',
	'.md': 'markdown',
}
myst_heading_anchors = 3

# Mermaid configuration (optional tuning)
mermaid_version = "10.9.0"
mermaid_output_format = "raw"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
