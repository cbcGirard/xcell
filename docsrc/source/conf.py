# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import re
sys.path.insert(0, os.path.abspath('../xcell/xcell'))
# import sphinx_rtd_theme
from sphinx_gallery.scrapers import matplotlib_scraper

import pyvista
# necessary when building the sphinx gallery
pyvista.BUILDING_GALLERY = True
pyvista.OFF_SCREEN = True

# -- Project information -----------------------------------------------------

project = 'xcell'
copyright = '2022, CBC Girard'
author = 'CBC Girard'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
  'sphinx.ext.napoleon',
  'autoapi.extension',
  'sphinx.ext.autodoc',
  'sphinx.ext.inheritance_diagram',
  'sphinx_copybutton',
#   'sphinx.ext.autosummary',
#   'sphinx_codeautolink',
  'sphinx_gallery.gen_gallery',
  'sphinx.ext.intersphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['css/xcell.css']

html_context = {"default_mode" : 'auto'}
html_theme_options = {
   "pygment_light_style": "default",
   "pygment_dark_style": "monokai",
   "logo" : {
       "text": "xcell"
   }
}

html_logo = '../../Examples/Geometry/logo.png'



class ResetArgv:
    def __repr__(self):
        return 'ResetArgv'

    def __call__(self, sphinx_gallery_conf, script_vars):
        if script_vars['src_file'] == 'polyCell.py':
            return ['-v']
        else:
            return []
        

# Attempt to force dark-light class in all gallery images (no dimming in dark mode)
class dark_scraper(object):
    def __repr__(self):
        return self.__class__.__name__
    
    def __call__(self,*args, **kwargs):

        rst= matplotlib_scraper(*args, transparent=True, #format='svg', 
                                  **kwargs)#+'\n    :dark-light:'
        return re.sub(r'<img', '<img class="dark-light"',rst)
        
sphinx_gallery_conf = {
     'examples_dirs': '../../Examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated 
     'matplotlib_animations': True,
     'image_scrapers': (dark_scraper(), 'pyvista'),
     'ignore_pattern': 'nongallery',
     'reset_argv': ResetArgv(),
     "backreferences_dir": None

}


#autoapi
autoapi_type = 'python'
autoapi_modules = {'xcell':None}
autoapi_dirs = ['../../xcell', ]

# external API doc links
intersphinx_mapping = {
    'pyvista': ('https://docs.pyvista.org/version/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),

}

