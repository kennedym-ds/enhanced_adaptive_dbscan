# setup.py

from setuptools import setup, find_packages
import pathlib

# Read the contents of README.md
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='enhanced_adaptive_dbscan',  # Replace with your own package name
    version='0.1.1',
    author='Michael Kennedy',
    author_email='michael.patrick.kennedy@outlook.ie',
    description='An Enhanced Adaptive DBSCAN clustering algorithm for semiconductor wafer defects.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/kennedym-ds/enhanced_adaptive_dbscan',  # Replace with your GitHub repo
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Choose your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scikit-learn',
        'plotly',
        'joblib',
        'pandas',
    ],
    extras_require={
        'dev': [
            'pytest',
            'sphinx',
            'wheel',
        ],
    },
    include_package_data=True,
)
