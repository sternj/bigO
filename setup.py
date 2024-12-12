from setuptools import setup, Extension

# Define the native extension
customalloc_extension = Extension(
    "customalloc",
    sources=["bigO/custom_alloc.c"],
    include_dirs=["bigO/include"],
)

# Call setup
setup(
    name="evans_giggles_arden_rad",
    version="0.0.2",
    description="Track asymptotic complexity in time and space.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Emery Berger",
    author_email="emery.berger@gmail.com",
    url="https://github.com/sternj/bigO",
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
    packages=["bigO"],
    package_data={"bigO": ["*.c", "*.h"]},
)
