from setuptools import Command, find_packages, setup

__lib_name__ = "GraphCellNet"
__lib_version__ = "1.0.0"
__description__ = "GraphCellNet: A Deep Learning Method for Integrated Single-Cell and Spatial Transcriptomic Analysis with Applications in Development and Disease"
__url__ = "https://github.com/DaiRuoYan-123/GraphCellNet-"
__author__ = "RuoyanDai"
__author_email__ = "dai1281311965@gmail.com"
__license__ = "MIT"
__keywords__ = ["Spatial transcriptomics","Graph neural networks" ,"Single-cell RNA sequencing", "Cell type composition", "Kolmogorov-Arnold Network",]
__requires__ = ["requests",]

with open("README.md", "r", encoding="utf-8") as f:
    __long_description__ = f.read()

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ["GraphCellNet"],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
    long_description = """GraphCellNet: A Deep Learning Method for Integrated Single-Cell and Spatial Transcriptomic Analysis with Applications in Development and Disease""",
    long_description_content_type="text/markdown"
)