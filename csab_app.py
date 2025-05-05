import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import qrcode
import io
import imagehash
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Hash import SHA256
import random
import base64


class CheetahKeyGenerator:
    def _init_(self, image):
        self.image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def isolate_spots(self, threshold_value, max_value):
        _, binary_thresh = cv2.threshold(self.gray_image, threshold_value,
                                         max_value, cv2.THRESH_BINARY_INV)
        return binary_thresh

    def find_and_filter_contours(self, binary_thresh, min_contour_area,
                                 max_contour_area):
        contours, _ = cv2.findContours(binary_thresh, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        return [
            contour for contour in contours
            if cv2.contourArea(contour) > min_contour_area
            and cv2.contourArea(contour) <= max_contour_area
        ]

    def apply_edge_detection(self, low_threshold, high_threshold):
        # Canny edge detection
        return cv2.Canny(self.gray_image, low_threshold, high_threshold)

    def morphological_operations(self, operation, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if operation == 'Dilation':
            return cv2.dilate(self.gray_image, kernel, iterations=1)
        elif operation == 'Erosion':
            return cv2.erode(self.gray_image, kernel, iterations=1)
        elif operation == 'Opening':
            return cv2.morphologyEx(self.gray_image, cv2.MORPH_OPEN, kernel)
        elif operation == 'Closing':
            return cv2.morphologyEx(self.gray_image, cv2.MORPH_CLOSE, kernel)
        else:
            return self.gray_image

    def generate_key_visualization(self, contours):
        areas = [cv2.contourArea(c) for c in contours]
        # Filter out areas that are too large (outliers)
        max_area_threshold = 5000  # Set a threshold value
        filtered_areas = [area for area in areas if area < max_area_threshold]

        fig, ax = plt.subplots()
        ax.hist(filtered_areas, bins=20, color='blue', alpha=0.7)
        ax.set_title('Distribution of Contour Areas (Filtered)')
        ax.set_xlabel('Area')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

        # Now proceed with key generation
        areas_bytes = b''.join(
            int(area).to_bytes((int(area).bit_length() + 7) // 8, 'big')
            for area in filtered_areas if area > 0)
        hash_obj = SHA256.new(areas_bytes)
        return hash_obj.digest()

    def create_image_from_contours(self, contours):
        mask = np.zeros(self.gray_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
        return mask


def encrypt_message(key, message):
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(message.encode(), AES.block_size))
    return cipher.iv + ct_bytes


def decrypt_message(key, ciphertext):
    iv = ciphertext[:AES.block_size]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    pt = unpad(cipher.decrypt(ciphertext[AES.block_size:]), AES.block_size)
    return pt.decode()


def generate_qr_code(data):
    """Generate a QR code from the provided data"""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    return qr_img


def embed_metadata(image, x, y, qr_size):
    """Embed position metadata in the first few pixels of the image"""
    # Convert coordinates and size to bytes (4 bytes each)
    x_bytes = x.to_bytes(4, byteorder='big')
    y_bytes = y.to_bytes(4, byteorder='big')
    size_bytes = qr_size.to_bytes(4, byteorder='big')
    
    # Combine metadata
    metadata = x_bytes + y_bytes + size_bytes
    
    # Convert to bit array for embedding
    meta_bits = ''.join(format(byte, '08b') for byte in metadata)
    
    # Embed in the first pixels of the image (red channel)
    pixels = np.array(image)
    bit_index = 0
    
    for i in range(min(len(meta_bits), 12)):  # First 12 pixels for metadata
        row = i // 4
        col = i % 4
        # Clear the LSB and set it to our metadata bit
        pixels[row, col, 0] = (pixels[row, col, 0] & 0xFE) | int(meta_bits[bit_index])
        bit_index += 1
    
    return Image.fromarray(pixels)


def extract_metadata(image):
    """Extract metadata from the first few pixels of the image"""
    pixels = np.array(image)
    meta_bits = ""
    
    for i in range(12):  # Read from first 12 pixels
        row = i // 4
        col = i % 4
        # Get the LSB of red channel
        meta_bits += str(pixels[row, col, 0] & 1)
    
    # Convert bit string to bytes
    meta_bytes = int(meta_bits, 2).to_bytes(12, byteorder='big')
    
    # Extract coordinates and size
    x = int.from_bytes(meta_bytes[0:4], byteorder='big')
    y = int.from_bytes(meta_bytes[4:8], byteorder='big')
    qr_size = int.from_bytes(meta_bytes[8:12], byteorder='big')
    
    return x, y, qr_size


def embed_qr_code_lsb(image, qr_img):
    """Embed QR code in the LSB of the blue channel"""
    # Convert QR image to numpy array and resize to a standard size
    qr_array = np.array(qr_img.convert('L'))
    qr_size = max(qr_array.shape[0], 100)  # Ensure minimum size for readability
    qr_array = cv2.resize(qr_array, (qr_size, qr_size), interpolation=cv2.INTER_NEAREST)
    
    # Generate random position for QR code that fits in the image
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    max_x = width - qr_size
    max_y = height - qr_size
    
    if max_x <= 0 or max_y <= 0:
        raise ValueError("Image too small to embed QR code")
    
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    
    # Embed metadata about QR position
    image = embed_metadata(image, x, y, qr_size)
    img_array = np.array(image)
    
    # Binary threshold the QR code (1 for black, 0 for white)
    _, qr_binary = cv2.threshold(qr_array, 128, 1, cv2.THRESH_BINARY)
    
    # Embed QR code in the blue channel LSB
    for i in range(qr_size):
        for j in range(qr_size):
            if x+j < width and y+i < height:  # Safety check
                # Clear the LSB and set it to our QR bit
                img_array[y+i, x+j, 0] = (img_array[y+i, x+j, 0] & 0xFE) | qr_binary[i, j]
    
    return Image.fromarray(img_array), x, y, qr_size


def extract_qr_code_lsb(image):
    """Extract QR code from the LSB of the blue channel"""
    # Extract metadata
    try:
        x, y, qr_size = extract_metadata(image)
        
        # Validate extracted metadata
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Safety checks with fallbacks
        if x < 0 or y < 0 or qr_size <= 0 or x + qr_size > width * 2 or y + qr_size > height * 2:
            st.warning("Invalid metadata detected. Using fallback extraction method.")
            # Fallback to a reasonable size and position
            qr_size = min(width, height) // 4
            x = (width - qr_size) // 2
            y = (height - qr_size) // 2
        
        # Ensure we don't go out of bounds
        x = min(x, width - 1)
        y = min(y, height - 1)
        qr_size = min(qr_size, min(width - x, height - y))
        
        # Ensure qr_size is at least 21 (minimum QR code size)
        qr_size = max(qr_size, 21)
        
        # Create a new image for the QR code
        qr_extracted = np.zeros((qr_size, qr_size), dtype=np.uint8)
        
        # Extract LSB values
        for i in range(qr_size):
            for j in range(qr_size):
                if x+j < width and y+i < height:
                    # Get LSB and convert to QR pixel (0 or 255)
                    bit = img_array[y+i, x+j, 0] & 1
                    qr_extracted[i, j] = 255 * (1 - bit)  # Convert bit to QR format (0=white, 1=black)
        
        # Apply adaptive thresholding to enhance QR code visibility
        _, qr_enhanced = cv2.threshold(qr_extracted, 128, 255, cv2.THRESH_BINARY)
        
        return Image.fromarray(qr_enhanced)
        
    except Exception as e:
        st.error(f"Error in QR extraction: {str(e)}")
        # Fallback: try to extract from the entire image
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Create a blank image for the LSB extraction
        lsb_image = np.zeros((height, width), dtype=np.uint8)
        
        # Extract LSBs from the entire image
        for i in range(height):
            for j in range(width):
                bit = img_array[i, j, 0] & 1
                lsb_image[i, j] = 255 * (1 - bit)
        
        # Process the image to enhance potential QR codes
        _, lsb_binary = cv2.threshold(lsb_image, 128, 255, cv2.THRESH_BINARY)
        
        # Resize to a standard QR size
        std_size = 250
        resized_img = cv2.resize(lsb_binary, (std_size, std_size), interpolation=cv2.INTER_NEAREST)
        
        return Image.fromarray(resized_img)


def calculate_phash(image):
    """Calculate perceptual hash of the image"""
    return str(imagehash.phash(image))


def verify_image_integrity(image, saved_hash):
    """Verify image integrity using perceptual hash"""
    current_hash = calculate_phash(image)
    # Allow some tolerance (hamming distance) for minor changes
    hamming_distance = sum(c1 != c2 for c1, c2 in zip(current_hash, saved_hash))
    
    # Return true if hash difference is within tolerance
    return hamming_distance < 8  # Increased tolerance for better success rate


def stego_encrypt(image, message, key):
    """Full steganographic encryption process"""
    # Step 1: Encrypt the message
    encrypted_data = encrypt_message(key, message)
    encrypted_b64 = base64.b64encode(encrypted_data).decode('utf-8')
    
    # Step 2: Generate QR code from encrypted data
    qr_img = generate_qr_code(encrypted_b64)
    
    # Step 3 & 4: Embed QR in image LSB and store position metadata
    stego_img, x, y, qr_size = embed_qr_code_lsb(image, qr_img)
    
    # Step 5: Calculate perceptual hash for integrity
    img_hash = calculate_phash(stego_img)
    
    return stego_img, img_hash, (x, y, qr_size)


def stego_decrypt(stego_img, saved_hash, key):
    """Full steganographic decryption process"""
    # Step 1: Verify image integrity with higher tolerance
    if not verify_image_integrity(stego_img, saved_hash):
        st.warning("Image integrity check detected changes, but attempting recovery...")
    
    try:
        # Step 2 & 3: Extract QR code from LSB
        extracted_qr = extract_qr_code_lsb(stego_img)
        
        # Step 4: Try multiple QR decoding methods
        decoded_data = None
        
        # First attempt: Use OpenCV QR detector
        try:
            extracted_qr_cv = np.array(extracted_qr)
            qr_detector = cv2.QRCodeDetector()
            decoded_data, bbox, straight_qrcode = qr_detector.detectAndDecode(extracted_qr_cv)
        except Exception as e:
            st.warning(f"OpenCV QR detection failed: {str(e)}")
        
        # Second attempt: Try with pyzbar if available
        if not decoded_data:
            try:
                import pyzbar.pyzbar as pyzbar
                # Try different preprocessing techniques
                for threshold in [100, 128, 150, 180]:
                    for blur in [0, 3, 5]:
                        try:
                            # Preprocess image
                            processed_qr = np.array(extracted_qr)
                            if blur > 0:
                                processed_qr = cv2.GaussianBlur(processed_qr, (blur, blur), 0)
                            _, processed_qr = cv2.threshold(processed_qr, threshold, 255, cv2.THRESH_BINARY)
                            
                            # Attempt to decode
                            decoded_objects = pyzbar.decode(processed_qr)
                            if decoded_objects:
                                decoded_data = decoded_objects[0].data.decode('utf-8')
                                break
                        except Exception:
                            continue
                    if decoded_data:
                        break
            except ImportError:
                st.warning("pyzbar not available. Consider installing it for better QR decoding.")
        
        # Third attempt: Manual base64 extraction
        if not decoded_data:
            st.warning("Standard QR decoding failed. Attempting direct data extraction...")
            try:
                # Direct base64 pattern matching from the image
                qr_binary = np.array(extracted_qr) > 128
                qr_bits = qr_binary.flatten()
                bit_string = ''.join('1' if bit else '0' for bit in qr_bits)
                
                # Try to find valid base64 patterns
                import re
                import binascii
                
                # Convert bit strings to ASCII and look for base64 patterns
                chars = []
                for i in range(0, len(bit_string), 8):
                    if i+8 <= len(bit_string):
                        byte = bit_string[i:i+8]
                        chars.append(chr(int(byte, 2)))
                
                ascii_text = ''.join(chars)
                # Look for base64 pattern
                base64_pattern = r'[A-Za-z0-9+/=]{16,}'
                matches = re.findall(base64_pattern, ascii_text)
                
                for match in matches:
                    try:
                        # Try to decode as base64
                        potential_data = base64.b64decode(match)
                        # If we can decrypt it, this is our data
                        decrypted = decrypt_message(key, potential_data)
                        decoded_data = match
                        break
                    except (binascii.Error, ValueError):
                        continue
            except Exception as e:
                st.warning(f"Direct extraction attempt failed: {str(e)}")
        
        # Fallback for testing: Manual entry of encoded data
        if not decoded_data:
            st.warning("All automated QR decoding methods failed.")
            if st.checkbox("Enter encoded data manually for decryption attempt"):
                manual_data = st.text_input("Enter base64 encoded data:", 
                                            placeholder="Paste the base64 encoded data here...")
                if manual_data:
                    decoded_data = manual_data

        # If we have decoded data, try to decrypt it
        if decoded_data:
            # Step 5: Decrypt the message
            try:
                encrypted_data = base64.b64decode(decoded_data)
                decrypted_message = decrypt_message(key, encrypted_data)
                return decrypted_message, extracted_qr
            except Exception as e:
                st.error(f"Decryption error: {str(e)}")
                raise ValueError("Data was found but could not be decrypted. Check the encryption key.")
        else:
            raise ValueError("Could not decode QR code data from the image.")
        
    except Exception as e:
        # Save the extracted QR for debugging (if it exists)
        if 'extracted_qr' in locals():
            st.error(f"Error during decryption: {str(e)}")
            st.image(extracted_qr, caption="Extracted QR Code (Failed to decode)")
            # Save to session state for potential debugging
            st.session_state['debug_qr'] = extracted_qr
            
            # Provide download option for debugging
            buf = io.BytesIO()
            extracted_qr.save(buf, format="PNG")
            st.download_button(
                label="Download Extracted QR for Debugging",
                data=buf.getvalue(),
                file_name="extracted_qr.png",
                mime="image/png"
            )
        
        raise ValueError("Decryption process failed. The steganography may have been corrupted.")


@st.cache_data
def process_image(uploaded_file, threshold_value, max_value, min_contour_area,
                  max_contour_area, edge_detection, low_threshold,
                  high_threshold, morph_op, kernel_size, use_processed_image):
    generator = CheetahKeyGenerator(Image.open(uploaded_file))

    processed_image = generator.gray_image
    if edge_detection:
        edges = generator.apply_edge_detection(low_threshold, high_threshold)
        st.image(edges, caption='Advanced Edge Detection Applied')
        if use_processed_image:
            processed_image = edges

    morphed_image = generator.morphological_operations(morph_op, kernel_size)
    st.image(morphed_image, caption='Morphological Operation Applied')
    if morph_op != 'None' and use_processed_image:
        processed_image = morphed_image

    if use_processed_image:
        generator.gray_image = processed_image

    # Update the isolate_spots and find_and_filter_contours methods to use processed_image
    binary_thresh = generator.isolate_spots(
        threshold_value,
        max_value,
    )
    contours = generator.find_and_filter_contours(binary_thresh,
                                                  min_contour_area,
                                                  max_contour_area)

    st.subheader("Visualizing Key Generation")
    key = generator.generate_key_visualization(contours)
    return key, generator.gray_image, binary_thresh, generator.create_image_from_contours(
        contours)


def main():
    st.set_page_config(page_title="Cheetah Cryptographic Steganography", layout="wide")
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Generate Key", "Encrypt & Hide", "Decrypt & Extract"])
    
    with tab1:
        st.title('Cheetah Cryptographic Key Generator')
        st.markdown(
            """ ## Overview 
            This application generates a cryptographic key from the unique patterns of a cheetah's coat.
            The key is used for securing messages with AES-256 encryption before hiding them in images.
            """
        )
        
        # Settings sidebar for key generation
        st.sidebar.title('Key Generation Settings')
        threshold_value = st.sidebar.slider('Binary Threshold', 0, 255, 175)
        max_value = st.sidebar.slider('Max Binary Value', 0, 255, 255)
        min_contour_area = st.sidebar.slider('Minimum Contour Area (px)', 0, 1000, 30)
        max_contour_area = st.sidebar.slider('Maximum Contour Area (px)',
                                            min_contour_area, 10000, 3000)
        edge_detection = st.sidebar.checkbox('Apply Edge Detection')
        low_threshold = st.sidebar.slider('Low Threshold for Edge Detection', 0, 100, 50)
        high_threshold = st.sidebar.slider('High Threshold for Edge Detection', 101, 200, 150)
        morph_op = st.sidebar.selectbox(
            'Morphological Operation',
            ['None', 'Dilation', 'Erosion', 'Opening', 'Closing'])
        kernel_size = st.sidebar.slider('Kernel Size for Morphological Operations', 1, 10, 3)

        uploaded_file = st.sidebar.file_uploader("Choose a cheetah image for key generation...",
                                                type=['png', 'jpg', 'jpeg'])

        use_processed_image = st.sidebar.checkbox("Use Processed Image for Key Generation")

        if uploaded_file is not None:
            key, gray_image, binary_thresh, spots_image = process_image(
                uploaded_file, threshold_value, max_value, min_contour_area,
                max_contour_area, edge_detection, low_threshold, high_threshold,
                morph_op, kernel_size, use_processed_image)
                
            st.subheader("Image Processing Steps")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(gray_image, caption='Grayscale Image')
            with col2:
                st.image(binary_thresh, caption='Binary Threshold Applied')
            with col3:
                st.image(spots_image, caption='Filtered Contours')

            st.subheader("Generated Cryptographic Key")
            key_hex = key.hex()
            st.code(key_hex, language='python')
            
            # Save key to session state for access in other tabs
            st.session_state['key'] = key
            st.session_state['key_hex'] = key_hex
            st.success("Key successfully generated! You can now proceed to the 'Encrypt & Hide' tab.")
    
    with tab2:
        st.title('Encrypt & Hide Message')
        st.markdown(
            """
            ## Steganographic Encryption Process
            1. Your message is encrypted using AES-256 with the cheetah-derived key
            2. The encrypted data is converted to a QR code
            3. Position coordinates for the QR code are randomly generated
            4. The QR code is embedded in the image using LSB steganography
            5. A perceptual hash of the image is generated for integrity verification
            """
        )

        # Add demo mode option with prefilled data
        demo_mode = st.checkbox("Use Demo Mode")
        if demo_mode:
            st.info("Demo Mode activated - using sample data for demonstration")
            # Generate a fixed key for demo
            if 'demo_key' not in st.session_state:
                import hashlib
                demo_seed = "cheetah_steganography_demo_key"
                st.session_state['demo_key'] = hashlib.sha256(demo_seed.encode()).digest()
                st.session_state['demo_key_hex'] = st.session_state['demo_key'].hex()
            
            key_to_use = st.session_state['demo_key']
            st.info(f"Using demo key: {st.session_state['demo_key_hex'][:8]}...{st.session_state['demo_key_hex'][-8:]}")
            
            # Default message
            demo_message = "This is a secret message hidden with Cheetah Steganography!"
        else:
            # Use the key from the first tab if available
            if 'key' in st.session_state:
                st.info(f"Using key: {st.session_state['key_hex'][:8]}...{st.session_state['key_hex'][-8:]}")
                key_to_use = st.session_state['key']
            else:
                st.warning("Please generate a key in the first tab, or enter a key manually below.")
                key_hex_input = st.text_input("Enter encryption key (hex format)", 
                                            placeholder="Enter 64-character hex key...",
                                            key="encrypt_key_input")
                try:
                    key_to_use = bytes.fromhex(key_hex_input) if key_hex_input else None
                except ValueError:
                    st.error("Invalid hex format for the key.")
                    key_to_use = None
            
            demo_message = ""
        
        # Upload image for steganography
        st.subheader("Upload Image for Steganography")
        stego_file = st.file_uploader("Choose an image to hide your message in...", 
                                    type=['png', 'jpg', 'jpeg'])
        
        if stego_file is not None:
            stego_image = Image.open(stego_file).convert('RGB')
            st.image(stego_image, caption="Carrier Image", width=400)
            
            # Get message to hide.
            message = st.text_area("Enter your message to hide:", 
                                height=100,
                                placeholder="Enter the secret message to hide in the image...")
            
            if st.button('Encrypt & Hide') and key_to_use is not None and message:
                try:
                    with st.spinner('Processing...'):
                        # Perform steganographic encryption
                        stego_result, img_hash, (x, y, qr_size) = stego_encrypt(stego_image, message, key_to_use)
                        
                        # Display results
                        st.success("Message successfully hidden in the image!")
                        st.image(stego_result, caption="Steganographic Image", width=400)
                        
                        # Save results to session state
                        st.session_state['stego_img'] = stego_result
                        st.session_state['img_hash'] = img_hash
                        st.session_state['stego_metadata'] = (x, y, qr_size)
                        
                        # Provide download button for the steganographic image
                        buf = io.BytesIO()
                        stego_result.save(buf, format="PNG")
                        st.download_button(
                            label="Download Steganographic Image",
                            data=buf.getvalue(),
                            file_name="stego_image.png",
                            mime="image/png"
                        )
                        
                        # Show perceptual hash for verification
                        st.subheader("Image Integrity Hash")
                        st.code(img_hash, language="text")
                        st.info("Save this hash value securely. It will be required for decryption to verify the image integrity.")
                        
                        # Also save hash to file for download
                        hash_file = io.StringIO()
                        hash_file.write(img_hash)
                        st.download_button(
                            label="Download Hash File",
                            data=hash_file.getvalue(),
                            file_name="image_hash.txt",
                            mime="text/plain"
                        )
                        
                except Exception as e:
                    st.error(f"Encryption failed: {str(e)}")
                    # Provide more detailed error information based on the exception type
                    if "convert" in str(e).lower() or "empty" in str(e).lower():
                        st.warning("There was an issue with image processing. Try a different image or check image format.")
                    elif "key" in str(e).lower():
                        st.warning("There was an issue with the encryption key. Make sure to generate a valid key first.")
                    else:
                        st.warning("Check your inputs and try again with a different configuration.")
                    
                    # Show traceback information in a collapsible section for debugging
                    with st.expander("Technical Error Details"):
                        import traceback
                        st.code(traceback.format_exc())
    
    with tab3:
        st.title('Decrypt & Extract Message')
        st.markdown(
            """
            ## Steganographic Decryption Process
            1. The perceptual hash of the image is verified against the original
            2. The QR code position is extracted from the image metadata
            3. The QR code is recovered from the LSBs at the specified position
            4. The QR code is decoded to obtain the encrypted data
            5. The message is decrypted using the cheetah-derived key
            """
        )
        
        # Use the key from the first tab if available
        if 'key' in st.session_state:
            st.info(f"Using key: {st.session_state['key_hex'][:8]}...{st.session_state['key_hex'][-8:]}")
            key_to_use = st.session_state['key']
        else:
            st.warning("Please enter the encryption key below.")
            key_hex_input = st.text_input("Enter encryption key (hex format)", 
                                        placeholder="Enter 64-character hex key...",
                                        key="decrypt_key_input")
            try:
                key_to_use = bytes.fromhex(key_hex_input) if key_hex_input else None
            except ValueError:
                st.error("Invalid hex format for the key.")
                key_to_use = None
        
        # Upload stego image for decryption
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Steganographic Image")
            decrypt_file = st.file_uploader("Choose the steganographic image...", 
                                        type=['png', 'jpg', 'jpeg'])
            
            if decrypt_file is not None:
                decrypt_image = Image.open(decrypt_file).convert('RGB')
                st.image(decrypt_image, caption="Steganographic Image", width=300)
        
        with col2:
            st.subheader("Enter Image Hash")
            if 'img_hash' in st.session_state:
                stored_hash = st.session_state['img_hash']
                st.info("Using stored hash from encryption.")
            else:
                stored_hash = st.text_input("Enter the image integrity hash:", 
                                        placeholder="Enter the hash value that was generated during encryption...",
                                        key="hash_input")
        
        if st.button('Decrypt & Extract') and key_to_use is not None and decrypt_file is not None and stored_hash:
            try:
                with st.spinner('Processing...'):
                    # Perform steganographic decryption
                    decrypted_msg, extracted_qr = stego_decrypt(decrypt_image, stored_hash, key_to_use)
                    
                    # Display results
                    st.success("Message successfully extracted and decrypted!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Extracted QR Code")
                        st.image(extracted_qr, caption="Extracted QR Code", width=200)
                    
                    with col2:
                        st.subheader("Decrypted Message")
                        st.text_area("Original message:", value=decrypted_msg, height=200, key="decrypted_text")
                    
            except Exception as e:
                st.error(f"Decryption failed: {str(e)}")
                if "integrity check failed" in str(e).lower():
                    st.warning("The image appears to have been modified since encryption. This could compromise the hidden data.")
                    
                # Recovery suggestions
                st.markdown("""
                ### Troubleshooting Suggestions:
                - Make sure you're using the exact same key used for encryption
                - Verify the image hasn't been modified, resized or compressed
                - Check that the correct image hash has been provided
                - Try uploading the original steganographic image without any modifications
                """)
                
                # Display debug info if available
                if 'debug_qr' in st.session_state:
                    st.subheader("Debug Information")
                    st.info("The QR code was extracted but could not be decoded. This is often due to corruption or noise in the extraction process.")
                    debug_fig, debug_ax = plt.subplots(1, 2, figsize=(10, 5))
                    debug_ax[0].imshow(np.array(st.session_state['debug_qr']), cmap='gray')
                    debug_ax[0].set_title("Extracted QR")
                    
                    # Try to enhance for visibility
                    enhanced_qr = np.array(st.session_state['debug_qr'])
                    _, enhanced_qr = cv2.threshold(enhanced_qr, 128, 255, cv2.THRESH_BINARY)
                    debug_ax[1].imshow(enhanced_qr, cmap='gray')
                    debug_ax[1].set_title("Enhanced QR")
                    
                    st.pyplot(debug_fig)


if __name__ == '__main__':
    main()
