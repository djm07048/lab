from TestProcessor import Book

class DstCoordsCalculator:
    def __init__(self, book_code, cont_version_dict, dict_para_coords, dict_para_alignment, para):
        self.book_para_cont_df = Book(book_code, cont_version_dict).load_book_para_cont_df
        self.book_code = book_code
        self.dict_para_coords = dict_para_coords
        self.book_type = book_code.split('_')[1]
        self.test_type = book_code.split('_')[0]
        self.is_para_predefined = {
            ('ex', 'pbm'): True,
            ('ex', 'sol'): False,
            ('wk', 'pbm'): False,
            ('wk', 'sol'): False
        }
        self.dict_para_alignment = dict_para_alignment
        self.para = para

    def align_by_para(self):
        if self.is_para_predefined[(self.test_type, self.book_type)]:
            self.para_group = self.book_para_cont_df.groupby('dst_pdf_para')
        else:
            current_para = self.para[0]
            current_y = self.dict_para_coords[current_para][2]
            for index, row in self.book_para_cont_df.iterrows():
                height = row['src_pdf_height']
                current_y -= height
                if current_y < self.dict_para_coords[current_para][4]:
                    current_para = self.para[self.para.index(current_para) + 1]
                    current_y = self.dict_para_coords[current_para][2]
                else:
                    row['dst_pdf_para'] = current_para
                row['dst_pdf_page'] = self.dict_para_coords[current_para][5]
                row['dst_pdf_path'] = self.dict_para_coords[current_para][6]
            self.para_group = self.book_para_cont_df.groupby('dst_pdf_para')
        self.book_para_cont_df = self.book_para_cont_df[self.book_para_cont_df['cont_type'] != 'blank']
        for para, group in self.para_group:
            alignment, spacing = self.dict_para_alignment[para]
            if alignment == 'justify_align':
                self.justify_align(group, spacing)
            elif alignment == 'justify_with_last_blank':
                self.justify_with_last_blank(group, spacing)
            elif alignment == 'top_align':
                self.top_align(group, spacing)
        return self.book_para_cont_df

    def justify_align(self, group, spacing):
        para = group['dst_pdf_para'].iloc[0]
        current_y = self.dict_para_coords[para][2]
        total_height = group['src_pdf_height'].sum()
        item_codes = group['item_code']
        item_change = (item_codes != item_codes.shift()).sum() - 1
        spacing_btw_item = (self.dict_para_coords[para][2] - self.dict_para_coords[para][
            3] - total_height) / item_change if item_change != 0 else 0
        for index, row in group.iterrows():
            row['dst_pdf_coords'][2] = current_y
            current_y -= row['src_pdf_height']
            row['dst_pdf_coords'][0] = self.dict_para_coords[para][0]
            row['dst_pdf_coords'][1] = self.dict_para_coords[para][1]
            row['dst_pdf_coords'][3] = current_y
            if index < len(group) - 1 and row['item_code'] != group.iloc[index + 1]['item_code']:
                current_y -= spacing_btw_item

    def justify_with_last_blank(self, group, spacing):
        para = group['dst_pdf_para'].iloc[0]
        current_y = self.dict_para_coords[para][2]
        total_height = group['src_pdf_height'].sum()
        item_codes = group['item_code']
        item_change = (item_codes != item_codes.shift()).sum() - 1
        spacing_btw_item = (self.dict_para_coords[para][2] - self.dict_para_coords[para][3] - total_height) / (
                    item_change + 1) if item_change != 0 else 0
        for index, row in group.iterrows():
            row['dst_pdf_coords'][2] = current_y
            current_y -= row['src_pdf_height']
            row['dst_pdf_coords'][0] = self.dict_para_coords[para][0]
            row['dst_pdf_coords'][1] = self.dict_para_coords[para][1]
            row['dst_pdf_coords'][3] = current_y
            if index < len(group) - 1 and row['item_code'] != group.iloc[index + 1]['item_code']:
                current_y -= spacing_btw_item
        current_y -= spacing_btw_item

    def top_align(self, group, spacing):
        para = group['dst_pdf_para'].iloc[0]
        current_y = self.dict_para_coords[para][2]
        for index, row in group.iterrows():
            row['dst_pdf_coords'][2] = current_y
            current_y -= row['src_pdf_height']
            row['dst_pdf_coords'][0] = self.dict_para_coords[para][0]
            row['dst_pdf_coords'][1] = self.dict_para_coords[para][1]
            row['dst_pdf_coords'][3] = current_y
            if index < len(group) - 1 and row['item_code'] != group.iloc[index + 1]['item_code']:
                current_y -= spacing

book_code = 'E1wk25주간01_pbm_00'
cont_version_dict = {'pbm': 'original'}
dist_para_coords = {
    '1a': [87.87, 406.74, 932.26, 123.37, 'basefile.pdf', 0],
    '1b': [433.64, 752.51, 932.26, 123.37, 'basefile.pdf', 0],
    '2a': [87.87, 406.74, 1031.53, 123.37, 'basefile.pdf', 0],
    '2b': [433.64, 752.51, 1031.53, 123.37, 'basefile.pdf', 0],
    '3a': [87.87, 406.74, 1031.53, 123.37, 'basefile.pdf', 0],
    '3b': [433.64, 752.51, 1031.53, 123.37, 'basefile.pdf', 0],
    '4a': [87.87, 406.74, 1031.53, 123.37, 'basefile.pdf', 0],
    '4b': [433.64, 752.51, 1031.53, 144.01, 'basefile.pdf', 0]
}
dict_para_alignment = {
    '1a': ['justify_align', 0],
    '1b': ['justify_align', 0],
    '2a': ['justify_align', 0],
    '2b': ['justify_align', 0],
    '3a': ['justify_align', 0],
    '3b': ['justify_align', 0],
    '4a': ['justify_align', 0],
    '4b': ['justify_with_last_blank', 0]
}
para = ['1a', '1a', '1a', '1b', '1b',
        '2a', '2a', '2b', '2b', '2b',
        '3a', '3a', '3a', '3b', '3b',
        '4a', '4a', '4a', '4b', '4b']

book_para_cont_df = DstCoordsCalculator(book_code, cont_version_dict, dist_para_coords, dict_para_alignment, para).align_by_para()
print(book_para_cont_df)