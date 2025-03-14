from setuptools import setup, find_packages

setup(
    name="3yp",  # Replace with your package name
    version="0.1",  # Update version as needed
    packages=find_packages(where="src"),  # Automatically find subpackages
    package_dir={"": "src"},  # Specify src as the root package
    install_requires=[],  # List dependencies here (e.g., ["numpy", "pandas"])
    author="Dhruv",
    author_email="nan",
    description="A short description of my package",
    long_description=open("README.md").read(),  # Reads from README.md if available
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mypackage",  # Update with your repo URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",  # Specify compatible Python versions
)
