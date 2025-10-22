from setuptools import setup, find_packages

setup(
    name="qmoe",
    version="0.1.0",
    description="faster moe",
    long_description_content_type="text/markdown",
    author="chenyidong",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/your-package",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",

)
