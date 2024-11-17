import os
import pdf2image
import pytesseract
from PIL import Image
from docx import Document
import io
import streamlit as st
import cv2
import numpy as np
from googletrans import Translator

class OCRPlatform:
    def __init__(self, output_format='docx'):
        self.output_format = output_format
        self.translator = Translator()

    def extract_text_from_pdf(self, pdf_path, page_range=None):
        """
        Extract Hindi text from a PDF, translate to English, and preserve layout.
        
        Parameters:
        pdf_path (str): Path to the input PDF file.
        page_range (tuple, optional): Tuple of start and end page numbers (inclusive) to process.
        
        Returns:
        str or bytes: Extracted and translated text in the specified output format.
        """
        # Convert PDF to images
        images = pdf2image.convert_from_path(pdf_path)
        if page_range:
            start, end = page_range
            images = images[start - 1:end]  # Adjust for zero-based indexing

        # Preprocess images to enhance OCR accuracy
        processed_images = self.preprocess_images(images)

        # Extract Hindi text from preprocessed images
        extracted_text = self.perform_ocr(processed_images)

        # Translate the Hindi text to English
        translated_text = [self.translate_text(text) for text in extracted_text]

        # Generate output in the desired format
        if self.output_format == 'docx':
            return self.generate_docx(translated_text)
        else:
            return '\n'.join(translated_text)

    def preprocess_images(self, images):
        """
        Preprocess images to improve OCR accuracy.
        
        Parameters:
        images (list): List of PIL.Image objects representing the PDF pages.
        
        Returns:
        list: List of preprocessed PIL.Image objects.
        """
        processed_images = []
        for image in images:
            # Convert to grayscale and apply thresholding
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            image_cv = cv2.threshold(image_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            # Denoise the image
            image_cv = cv2.fastNlMeansDenoising(image_cv, h=30)

            # Convert back to PIL Image
            processed_images.append(Image.fromarray(image_cv))
        return processed_images

    def perform_ocr(self, images):
        """
        Perform OCR on the preprocessed images to extract Hindi text.
        
        Parameters:
        images (list): List of preprocessed PIL.Image objects.
        
        Returns:
        list: List of extracted text for each page.
        """
        extracted_text = []
        for image in images:
            text = pytesseract.image_to_string(image, lang='hin')
            if text.strip():  # Only add text if there's content
                extracted_text.append(text.strip())
        return extracted_text

    def translate_text(self, text):
        """
        Translate Hindi text to English.
        
        Parameters:
        text (str): The Hindi text to be translated.
        
        Returns:
        str: The translated English text.
        """
        try:
            translated = self.translator.translate(text, src='hi', dest='en')
            return translated.text
        except Exception as e:
            return f"Error translating text: {str(e)}"

    def generate_docx(self, translated_text):
        """
        Generate a DOCX document from the translated text.
        
        Parameters:
        translated_text (list): List of translated text for each page.
        
        Returns:
        bytes: DOCX document as bytes
        """
        document = Document()
        for page_text in translated_text:
            document.add_paragraph(page_text)
            document.add_page_break()

        # Save the DOCX document to a byte stream
        byte_stream = io.BytesIO()
        document.save(byte_stream)
        return byte_stream.getvalue()

def main():
    st.set_page_config(page_title="Real-Time OCR Platform")
    st.title("Real-Time OCR Platform for Hindi to English Text Extraction")

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Create the "uploads" directory if it doesn't exist
        uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
        os.makedirs(uploads_dir, exist_ok=True)

        # Save the uploaded file
        pdf_path = os.path.join(uploads_dir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Page range selection
        page_range = st.sidebar.text_input("Enter page range (e.g., 1-5)", "1-")
        try:
            start_page, end_page = map(int, page_range.split("-"))
        except ValueError:
            st.error("Please enter a valid page range (e.g., 1-5)")
            return

        # Output format selection
        output_format = st.sidebar.selectbox("Select output format", ["docx", "text"])

        # Process the PDF and generate the output
        ocr_platform = OCRPlatform(output_format=output_format)
        extracted_content = ocr_platform.extract_text_from_pdf(pdf_path, (start_page, end_page))

        # Display the output
        if output_format == 'docx':
            st.download_button(
                label="Download DOCX",
                data=extracted_content,
                file_name=f"{uploaded_file.name.split('.')[0]}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        else:
            st.write(extracted_content)

if __name__ == "__main__":
    main()
