import streamlit as st
from skimage import io
import numpy as np
import numpy.matlib
import os
import random
from PIL import Image
from io import BytesIO


def init_centroids(X, K):
    c = random.sample(list(X), K)
    return c

def closest_centroids(X, c):
    K = np.size(c, 0)
    idx = np.zeros((np.size(X, 0), 1))
    arr = np.empty((np.size(X, 0), 1))
    for i in range(0, K):
        y = c[i]
        temp = np.ones((np.size(X, 0), 1)) * y
        b = np.power(np.subtract(X, temp), 2)
        a = np.sum(b, axis=1)
        a = np.asarray(a)
        a.resize((np.size(X, 0), 1))
        arr = np.append(arr, a, axis=1)
    arr = np.delete(arr, 0, axis=1)
    idx = np.argmin(arr, axis=1)
    return idx

def compute_centroids(X, idx, K):
    n = np.size(X, 1)
    centroids = np.zeros((K, n))
    for i in range(0, K):
        ci = idx == i
        ci = ci.astype(int)
        total_number = sum(ci)
        ci.resize((np.size(X, 0), 1))
        total_matrix = np.matlib.repmat(ci, 1, n)
        ci = np.transpose(ci)
        total = np.multiply(X, total_matrix)
        centroids[i] = (1 / total_number) * np.sum(total, axis=0)
    return centroids

def run_kMean(X, initial_centroids, max_iters):
    m = np.size(X, 0)
    n = np.size(X, 1)
    K = np.size(initial_centroids, 0)
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros((m, 1))
    for i in range(1, max_iters):
        idx = closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
    return centroids, idx

def compress_image(image_upload, K, max_iters):
    print(image_upload)
    # Convert the PIL image to a numpy array
    uploaded_image_np = np.array(uploaded_image_pil)

    if uploaded_image_np.size == 0:
        st.error("Error: The image is empty.")
        return None  # handle the error as needed

    if len(uploaded_image_np.shape) == 3:  # Check if it's a color image
        rows, cols, _ = uploaded_image_np.shape
    else:  # If it's a grayscale image
        rows, cols = uploaded_image_np.shape[:2]  # Extract the first two elements of the shape
        if len(uploaded_image_np.shape) == 2:
            uploaded_image_np = np.stack((uploaded_image_np,) * 3, axis=-1)  # Convert to a 3-channel image

    uploaded_image_np = uploaded_image_np / 255
    X = uploaded_image_np.reshape(uploaded_image_np.shape[0] * uploaded_image_np.shape[1], 3)

    initial_centroids = init_centroids(X, K)
    centroids, idx = run_kMean(X, initial_centroids, max_iters)

    idx = closest_centroids(X, centroids)
    X_recovered = centroids[idx]
    X_recovered = np.reshape(X_recovered, (rows, cols, 3))

    # Display the compressed image
    st.subheader("Compressed Image")
    st.image(X_recovered, caption="Compressed Image", use_column_width=True)

    # Save the recovered image with the same name and '_compressed' suffix
    output_image_path = f"{image_upload.name.replace('.png', '_compressed.png')}"
    X_recovered_uint8 = (X_recovered * 255).astype(np.uint8)
    io.imsave(output_image_path, X_recovered_uint8)

    return output_image_path




# Streamlit UI
st.title("Image Compression with K-Means Clustering")

# Upload image
image_upload = st.file_uploader("Upload an image :", type=["png"])
if image_upload:
    uploaded_image_pil = Image.open(image_upload)
    uploaded_image_np = np.array(uploaded_image_pil)
    st.image(uploaded_image_np, caption="Uploaded Image", use_column_width=True)


    # Set parameters
    K = st.slider("Number of clusters (bits):", min_value=2, max_value=256, value=16)
    max_iters = st.slider("Number of max iterations:", min_value=1, max_value=100, value=50)

    # Compression button
    if st.button("Compress Image"):
        compressed_image_path = compress_image(image_upload, K, max_iters)
        compressed_image = io.imread(compressed_image_path)

        # Display images
        st.subheader("Original Image vs. Compressed Image")
        st.image([io.imread(image_upload), compressed_image], caption=["Original", "Compressed"], width=300)

        # Display image sizes and compression ratio
        image_content = BytesIO()
        uploaded_image_pil.save(image_content, format='PNG')  # Save the content of the image to BytesIO
        image_content.seek(0)  # Seek to the beginning of the BytesIO object
        original_size = len(image_content.read()) / 1024

        compressed_size = os.stat(compressed_image_path).st_size / 1024
        compression_ratio = original_size / compressed_size

        st.subheader("Image Sizes and Compression Ratio")
        st.write(f"Original Image Size: {original_size:.2f} KB")
        st.write(f"Compressed Image Size: {compressed_size:.2f} KB")
        st.write(f"Compression Ratio: {compression_ratio:.2f}")
                


# Instructions
st.sidebar.subheader("Instructions")
st.sidebar.write("1. Upload an image.")
st.sidebar.write("2. Set the number of clusters (bits) and maximum iterations.")
st.sidebar.write("3. Click the 'Compress Image' button.")
st.sidebar.write("4. View the original and compressed images, along with sizes and compression ratio.")