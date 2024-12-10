from setuptools import setup, Extension

# Define the native extension
customalloc_extension = Extension(
    "customalloc",
    sources=["timespace/custom_alloc.c"],
    include_dirs=["timespace/include"],
)

# Call setup
setup(
    name="timespace",
    version="0.0.1",
    description="Track asymptotic complexity in time and space.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Emery Berger",
    author_email="emery.berger@gmail.com",
    url="https://github.com/plasma-umass/timespace",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development",
        "Topic :: Software Development :: Debuggers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires=">=3.8",
    ext_modules=[customalloc_extension],
    packages=["timespace"],
    package_data={"timespace": ["*.c", "*.h"]},
)
