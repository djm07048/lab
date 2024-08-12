import fitz  # PyMuPDF

def crop_pdf_page(src_pdf_path, src_pdf_page, src_pdf_coords, output_pdf_path):
    x_lt, x_rt, y_top_from_btm, y_btm_from_btm = src_pdf_coords  # Adjusted to match PDF coordinate system
    if x_lt >= x_rt or y_top_from_btm <= y_btm_from_btm:
        raise ValueError("Invalid coordinates: get available area.")

    # Get the height of the page and inverse yaxis to fit in the PyMuPDF coordinate system
    src_pdf = fitz.open(src_pdf_path)
    page = src_pdf[src_pdf_page]
    height = page.rect.height
    y_top = height - y_top_from_btm
    y_btm = height - y_btm_from_btm

    rect = fitz.Rect(x_lt, y_top, x_rt, y_btm)

    # Create a new PDF with the cropped content
    writer = fitz.open()
    new_page = writer.new_page(width=rect.width, height=rect.height)
    new_page.show_pdf_page(fitz.Rect(0, 0, rect.width, rect.height), src_pdf, src_pdf_page, clip=rect)

    writer.save(output_pdf_path)
    writer.close()
    src_pdf.close()

# Example usage
src_pdf_path = 'E1aaaHY250023.pdf'
src_pdf_page = 0
src_pdf_coords = [100, 200, 1050, 950]  # Ensure these coordinates form a valid rectangle
cropped_pdf_path = 'cropped.pdf'

crop_pdf_page(src_pdf_path, src_pdf_page, src_pdf_coords, cropped_pdf_path)


def overlay_pdf_page(cropped_src_pdf_path, dst_pdf_path, dst_pdf_page, dst_pdf_coords, output_pdf_path):
    x_lt, x_rt, y_top_from_btm, y_btm_from_btm = dst_pdf_coords  # Adjusted to match PDF coordinate system
    if x_lt >= x_rt or y_top_from_btm <= y_btm_from_btm:
        raise ValueError("Invalid coordinates: get available area.")

    # Get the height of the page and inverse yaxis to fit in the PyMuPDF coordinate system
    dst_pdf = fitz.open(dst_pdf_path)
    overlay_pdf = fitz.open(cropped_src_pdf_path)
    dst_page = dst_pdf[dst_pdf_page]
    height = dst_page.rect.height
    y_top = height - y_top_from_btm
    y_btm = height - y_btm_from_btm

    rect = fitz.Rect(x_lt, y_top, x_rt, y_btm)

    # Overlay the cropped PDF onto the destination PDF
    overlay_page = overlay_pdf[0]
    dst_page.show_pdf_page(rect, overlay_pdf, 0)

    dst_pdf.save(output_pdf_path)
    dst_pdf.close()
    overlay_pdf.close()

# Example usage
cropped_src_pdf_path = 'cropped.pdf'
dst_pdf_path = 'Layout.pdf'
dst_pdf_page = 0
dst_pdf_coords = [300, 400, 400, 300]
output_pdf_path = 'Modified_Layout_E1_exam.pdf'

overlay_pdf_page(cropped_src_pdf_path, dst_pdf_path, dst_pdf_page, dst_pdf_coords, output_pdf_path)

'''
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
'''