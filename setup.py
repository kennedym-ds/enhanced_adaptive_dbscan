# setup.py

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='enhanced_adaptive_dbscan',  # Replace with your own package name
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='An Enhanced Adaptive DBSCAN clustering algorithm for semiconductor wafer defects.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/enhanced_adaptive_dbscan',  # Replace with your own GitHub repo
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
