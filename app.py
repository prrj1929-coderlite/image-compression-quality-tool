
import io
import math

import numpy as np
import streamlit as st
from PIL import Image
from skimage.metrics import structural_similarity

st.set_page_config(
    page_title="Image Compression & Quality Evaluation",
    page_icon="🖼️",
    layout="wide"
)

st.title("Image Compression and Quality Evaluation Tool")
st.write(
    "Analyze compression efficiency and reconstructed image quality."
)

with st.sidebar:
    st.header("Compression Settings")

    method = st.selectbox(
        "Compression Method",
        ["JPEG (Lossy)", "PNG (Lossless)"]
    )

    quality = st.slider(
        "JPEG Quality",
        min_value=10,
        max_value=100,
        value=75,
        step=5,
        disabled=(method == "PNG (Lossless)")
    )

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]
)


def format_bytes(size):
    if size < 1024:
        return f"{size} bytes"
    elif size < 1024 ** 2:
        return f"{size / 1024:.2f} KB"
    return f"{size / (1024 ** 2):.2f} MB"


if uploaded_file is None:
    st.info("Upload an image to start compression and evaluation.")
    st.markdown("""
    ### Evaluation Metrics
    - **Compression Ratio:** Original size / Compressed size
    - **Space Saved:** Percentage of file size reduced
    - **MSE:** Lower values indicate less pixel error
    - **PSNR:** Higher values generally indicate better reconstruction
    - **SSIM:** Values closer to 1 indicate greater structural similarity
    """)
    st.stop()


try:
    original_bytes = uploaded_file.getvalue()
    original_image = Image.open(
        io.BytesIO(original_bytes)
    ).copy()
except Exception as error:
    st.error(f"Unable to read image: {error}")
    st.stop()


st.subheader("Original Image")
st.image(
    original_image,
    caption=uploaded_file.name,
    use_container_width=True
)


if st.button("Compress and Evaluate", type="primary"):

    # Prepare image for the selected format
    if method == "JPEG (Lossy)":
        if original_image.mode not in ["RGB", "L"]:
            image_to_compress = original_image.convert("RGB")
        else:
            image_to_compress = original_image.copy()

        output_buffer = io.BytesIO()
        image_to_compress.save(
            output_buffer,
            format="JPEG",
            quality=quality,
            optimize=True
        )
        output_name = "compressed_image.jpg"
        mime_type = "image/jpeg"

    else:
        image_to_compress = original_image.copy()

        if image_to_compress.mode not in [
            "RGB", "RGBA", "L", "LA"
        ]:
            image_to_compress = image_to_compress.convert("RGB")

        output_buffer = io.BytesIO()
        image_to_compress.save(
            output_buffer,
            format="PNG",
            optimize=True
        )
        output_name = "compressed_image.png"
        mime_type = "image/png"

    compressed_bytes = output_buffer.getvalue()

    reconstructed_image = Image.open(
        io.BytesIO(compressed_bytes)
    ).copy()

    # Convert images to compatible arrays for evaluation
    if (
        image_to_compress.mode in ["RGB", "RGBA"]
        or reconstructed_image.mode in ["RGB", "RGBA"]
    ):
        original_array = np.asarray(
            image_to_compress.convert("RGB"),
            dtype=np.float64
        ) / 255.0

        reconstructed_array = np.asarray(
            reconstructed_image.convert("RGB"),
            dtype=np.float64
        ) / 255.0
    else:
        original_array = np.asarray(
            image_to_compress.convert("L"),
            dtype=np.float64
        ) / 255.0

        reconstructed_array = np.asarray(
            reconstructed_image.convert("L"),
            dtype=np.float64
        ) / 255.0

    # Calculate quality metrics
    mse = float(
        np.mean((original_array - reconstructed_array) ** 2)
    )

    if mse == 0:
        psnr = math.inf
    else:
        psnr = 10 * math.log10(1.0 / mse)

    try:
        if original_array.ndim == 3:
            ssim = structural_similarity(
                original_array,
                reconstructed_array,
                data_range=1.0,
                channel_axis=-1
            )
        else:
            ssim = structural_similarity(
                original_array,
                reconstructed_array,
                data_range=1.0
            )
    except Exception:
        ssim = None

    # File size and compression calculations
    original_size = len(original_bytes)
    compressed_size = len(compressed_bytes)

    compression_ratio = (
        original_size / compressed_size
        if compressed_size > 0 else 0
    )

    space_saved = (
        (original_size - compressed_size)
        / original_size * 100
        if original_size > 0 else 0
    )

    st.session_state["result"] = {
        "original": original_image,
        "reconstructed": reconstructed_image,
        "compressed_bytes": compressed_bytes,
        "output_name": output_name,
        "mime_type": mime_type,
        "original_size": original_size,
        "compressed_size": compressed_size,
        "ratio": compression_ratio,
        "space_saved": space_saved,
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim
    }


# Display saved result after the button reruns the app
if "result" in st.session_state:

    result = st.session_state["result"]

    st.divider()
    st.subheader("Original vs Reconstructed Image")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            result["original"],
            caption="Original Image",
            use_container_width=True
        )

    with col2:
        st.image(
            result["reconstructed"],
            caption="Reconstructed Image",
            use_container_width=True
        )

    st.subheader("Compression Efficiency")

    c1, c2 = st.columns(2)

    c1.metric(
        "Original File Size",
        format_bytes(result["original_size"])
    )

    c2.metric(
        "Compressed File Size",
        format_bytes(result["compressed_size"])
    )

    c3, c4 = st.columns(2)

    c3.metric(
        "Compression Ratio",
        f'{result["ratio"]:.2f}:1'
    )

    c4.metric(
        "Space Saved",
        f'{result["space_saved"]:.2f}%'
    )

    st.subheader("Image Quality Evaluation")

    q1, q2, q3 = st.columns(3)

    q1.metric("MSE", f'{result["mse"]:.8f}')

    if math.isinf(result["psnr"]):
        q2.metric("PSNR", "Infinite")
    else:
        q2.metric("PSNR", f'{result["psnr"]:.2f} dB')

    if result["ssim"] is None:
        q3.metric("SSIM", "Unavailable")
    else:
        q3.metric("SSIM", f'{result["ssim"]:.4f}')

    st.download_button(
        label="Download Compressed Image",
        data=result["compressed_bytes"],
        file_name=result["output_name"],
        mime=result["mime_type"],
        type="primary"
    )

    if result["space_saved"] < 0:
        st.warning(
            "The compressed file is larger than the original file. "
            "This can happen depending on the original format and image content."
        )

    with st.expander("Understand the metrics"):
        st.markdown("""
        - **Compression Ratio:** Higher generally means greater size reduction.
        - **Space Saved:** Positive means the output is smaller; negative means it is larger.
        - **MSE:** Zero means identical pixel values.
        - **PSNR:** Higher generally means lower pixel error.
        - **SSIM:** A value closer to 1 generally indicates greater structural similarity.

        JPEG uses lossy compression. PNG uses lossless compression.
        Quality metrics compare decoded pixel values, while file sizes use the actual input and output bytes.
        """)
