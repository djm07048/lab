import PyPDF2

class PdfOverlay:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.reader = PyPDF2.PdfReader(pdf_path)
        self.writer = PyPDF2.PdfWriter()

    def overlay_pdf(self, overlay_pdf_path, page_num):
        overlay_pdf = PyPDF2.PdfReader(overlay_pdf_path)
        overlay_page = overlay_pdf.pages[0]
        overlay_page.merge_page(self.reader.pages[page_num])
        self.writer.add_page(overlay_page)

    def save_pdf(self, output_path):
        with open(output_path, 'wb') as f:
            self.writer.write(f)

class PdfOverlay:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.reader = PyPDF2.PdfReader(pdf_path)
        self.writer = PyPDF2.PdfWriter()

    def overlay_pdf(self, overlay_pdf_path, page_num):
        overlay_pdf = PyPDF2.PdfReader(overlay_pdf_path)
        overlay_page = overlay_pdf.pages[0]
        overlay_page.merge_page(self.reader.pages[page_num])
        self.writer.add_page(overlay_page)

    def save_pdf(self, output_path):
        with open(output_path, 'wb') as f:
            self.writer.write(f)