# Image Compression with K-Means Clustering

This project demonstrates how to perform image compression using the K-Means clustering algorithm, implemented as an interactive web application using Streamlit. The primary goal is to reduce the number of unique colors in an image, effectively compressing the image while retaining its visual quality.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Overview

Image compression is a key technique in reducing the size of image files without compromising too much on the quality. This project leverages the K-Means clustering algorithm to compress images by reducing the number of distinct colors used in the image. By clustering similar colors together and replacing them with a representative color, the image size can be significantly reduced.

## Features

- Compress images using K-Means clustering
- Control the number of clusters (colors) for compression
- Compare original and compressed images
- Interactive web application using Streamlit

## Installation

1. Clone the repository:

```sh
git clone https://github.com/Abubakar-00/Image-Compression-with-K-Means-Clustering.git
cd Image-Compression-with-K-Means-Clustering
```
2. Install the necessary dependencies. This project requires Python and pip.

```sh
pip install -r requirements.txt
```

## Usage

To run the Streamlit application, use the following command:

```sh
streamlit run app.py
```

Follow the instructions in the web application to upload an image, select the number of clusters, and view the compressed image.

## Example

Here are examples of an image before and after compression using K-Means clustering:

### Original Image

![Dog](https://github.com/user-attachments/assets/4e1f0e02-4362-4e73-afe2-01253ff06ca9)

### Compressed Image

![Dog_compressed](https://github.com/user-attachments/assets/6ce51f00-b31b-41a9-8db1-68ba7441c5ed)


You can see that the compressed image retains much of the visual quality of the original image while using fewer colors.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please create an issue or submit a pull request. Please make sure to follow the code of conduct.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
