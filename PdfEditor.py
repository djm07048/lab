import fitz  # PyMuPDF


def crop_pdf_page(src_pdf_path, src_pdf_page, src_pdf_coords, output_pdf_path):
    x_lt, x_rt, y_top, y_btm = src_pdf_coords
    src_pdf = fitz.open(src_pdf_path)
    page = src_pdf[src_pdf_page]
    rect = fitz.Rect(x_lt, y_btm, x_rt, y_top)

    # Create a new PDF with the cropped content
    writer = fitz.open()
    writer.insert_page(-1, width=rect.width, height=rect.height)
    writer[-1].show_pdf_page(fitz.Rect(0, 0, rect.width, rect.height), src_pdf, src_pdf_page, clip=rect)

    writer.save(output_pdf_path)
    writer.close()
    src_pdf.close()


def overlay_pdf_page(dst_pdf_path, dst_pdf_page, dst_pdf_coords, overlay_pdf_path, output_pdf_path):
    x_lt, x_rt, y_top, y_btm = dst_pdf_coords
    dst_pdf = fitz.open(dst_pdf_path)
    overlay_pdf = fitz.open(overlay_pdf_path)
    page = dst_pdf[dst_pdf_page]
    rect = fitz.Rect(x_lt, y_btm, x_rt, y_top)

    overlay_page = overlay_pdf[0]
    overlay_pixmap = overlay_page.get_pixmap()

    page.insert_image(rect, pixmap=overlay_pixmap)

    dst_pdf.save(output_pdf_path)
    dst_pdf.close()
    overlay_pdf.close()


# Example usage
src_pdf_path = 'stamp.pdf'
src_pdf_page = 0
src_pdf_coords = [100, 200, 500, 400]
cropped_pdf_path = 'cropped.pdf'

dst_pdf_path = 'Layout_E1_exam.pdf'
dst_pdf_page = 1
dst_pdf_coords = [300, 400, 400, 300]
output_pdf_path = 'Modified_Layout_E1_exam.pdf'

crop_pdf_page(src_pdf_path, src_pdf_page, src_pdf_coords, cropped_pdf_path)
overlay_pdf_page(dst_pdf_path, dst_pdf_page, dst_pdf_coords, cropped_pdf_path, output_pdf_path)