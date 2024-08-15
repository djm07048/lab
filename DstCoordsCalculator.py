import os


{'E1aaaHY250021_pbm_pbm_01': [0, 1, 2, 4, 3, 'pbm', 'pbm', '01']}
#key = cont_code : value = [x_lt, x_rt, y_top, y_btm, height, cont_type, cont_subtype, cont_num]
for cont_code, coords in row['item_cont_info'].items():
    cont_type, cont_sub_type, cont_num = cont_code.split('_')[1:]
    item_cont_example ={
        'cont_code': cont_code,
        'item_code': item_code,
        'item_serial_num': item_serial_num,
        'cont_type': cont_type,
        'cont_sub_type': cont_sub_type,
        'cont_num': cont_num,
        'src_pdf_path': item_pdf_path,
        'src_pdf_page': 0,
        'src_pdf_coords': coords[0:3],
        'src_pdf_height': coords[4],
        'dst_pdf_path': None,
        'dst_pdf_page': None,
        'dst_pdf_para': None,
        'dst_pdf_coords': None
    }

    book_cont_example ={
        'cont_code': None,
        'item_code': None,
        'item_serial_num': None,
        'cont_type': cont_type,
        'cont_sub_type': cont_sub_type,
        'cont_num': cont_num,
        'src_pdf_path': 'input',
        'src_pdf_page': 'input',
        'src_pdf_coords': 'input',
        'src_pdf_height': 'input',
        'dst_pdf_path': None,
        'dst_pdf_page': None,
        'dst_pdf_para': None,
        'dst_pdf_coords': None
    }


    dict_para_coords = {
        '1a' : [x_lt, x_rt, y_upper, y_lower, height, dst_pdf_file, dst_pdf_page],
        '1b' : [x_lt, x_rt, y_upper, y_lower, height, dst_pdf_file, dst_pdf_page],
        '2a' : [x_lt, x_rt, y_upper, y_lower, height, dst_pdf_file, dst_pdf_page],
    }


    # value = [alignment, spacing]
    # alignment = 'justity_alingn', 'justify_with_last_blank', 'top_align'
    # spacing = int, 50 for top_align
# 1) merged_cont_df를 받아오는 코드

# 시작하기 전에
# para, dict_para_coords, dict_para_alignment를 정의해야 함
# 이때 dict들은 재사용을 전제로 구성하지 말고, 모든 page에 대해 모두 정의해주어야 함.
# if 2단으로 100page -> 길이가 200인 para, dict를 만들어야 함
# 물론, 이때 반복문을 활용하여 할 수도 있겠다.

# book_code는 입력하면 됨
# merged_cont_df는 열심히.. 만든 것을 받아오면 됨

class DstCoordsCalculator:
    def __init__(self, book_code, merged_cont_df):
        self.merged_cont_df = merged_cont_df
        self.book_code = book_code
        self.book_type = book_code.split('_')[1]
        self.test_type = book_code.split('_')[0]

        self.is_para_predefined = {
            ['ex', 'pbm']: True,
            ['ex', 'sol']: False,
            ['wk', 'pbm']: False,
            ['wk', 'sol']: False
        }
        self.dict_para_alignment = {
            '1a' : ['justify_align', 0],
            '1b' : ['justify_align', 0],
            '2a' : ['justify_align', 0],
            '2b' : ['justify_align', 0],
            '3a' : ['justify_align', 0],
            '3b' : ['justify_align', 0],
            '4a' : ['justify_align', 0],
            '4b' : ['justify_with_last_blank', 0]

        }

        self.para = ['1a', '1b', '2a', '2b', '3a', '3b', '4a', '4b']

    def align_by_para(self):
        if self.is_para_predefined[self.test_type, self.book_type]:
            # merged_cont_df에 predefined para의 정보를 넣어 주어야 한다.
            self.para_group = self.merged_cont_df.groupby('dst_pdf_para')
        else:
            current_para = self.para[0]
            current_y = dict_para_coords[current_para][2]
            for index, row in self.merged_cont_df.iterrows():
                height = row['src_pdf_height']
                current_y -= height
                if current_y < dict_para_coords[current_para][4]:
                    current_para = self.para[self.para.index(current_para) + 1]
                    current_y = dict_para_coords[current_para][2]
                else:
                    row['dst_pdf_para'] = current_para

                row['dst_pdf_page'] = dict_para_coords[current_para][5]
                row['dst_pdf_path'] = dict_para_coords[current_para][6]

            self.para_group = self.merged_cont_df.groupby('dst_pdf_para')

        # Check if the row's 'cont_type' is 'blank' and remove the row if true
        self.merged_cont_df = self.merged_cont_df[self.merged_cont_df['cont_type'] != 'blank']

        # align conts in para
        for para, group in self.para_group:
            alignment, spacing = self.dict_para_alignment[para]
            if alignment == 'justify_align':
                self.justify_align(group, spacing)
            elif alignment == 'justify_with_last_blank':
                self.justify_with_last_blank(group, spacing)
            elif alignment == 'top_align':
                self.top_align(group, spacing)

        return self.merged_cont_df

    def justify_align(self, group, spacing):
        para = group['dst_pdf_para'].iloc[0]
        current_y = dict_para_coords[para][2]

        # Calculate total height of all rows in the group
        total_height = group['src_pdf_height'].sum()

        # Count the number of item_code changes
        item_codes = group['item_code']
        item_change = (item_codes != item_codes.shift()).sum() - 1

        # Calculate spacing between items
        if item_change != 0:
            spacing_btw_item = (dict_para_coords[para][2] - dict_para_coords[para][3] - total_height) / item_change
        else:
            spacing_btw_item = 0

        # Iterate over each row in the group
        for index, row in group.iterrows():
            row['dst_pdf_coords'][2] = current_y
            current_y -= row['src_pdf_height']
            row['dst_pdf_coords'][0] = dict_para_coords[para][0]
            row['dst_pdf_coords'][1] = dict_para_coords[para][1]
            row['dst_pdf_coords'][3] = current_y

            # Check if the current row's item_code is different from the next row's item_code
            if index < len(group) - 1 and row['item_code'] != group.iloc[index + 1]['item_code']:
                current_y -= spacing_btw_item

    def justify_with_last_blank(self, group, spacing):
        para = group['dst_pdf_para'].iloc[0]
        current_y = dict_para_coords[para][2]

        # Calculate total height of all rows in the group
        total_height = group['src_pdf_height'].sum()

        # Count the number of item_code changes
        item_codes = group['item_code']
        item_change = (item_codes != item_codes.shift()).sum() - 1

        # Calculate spacing between items
        if item_change != 0:
            spacing_btw_item = (dict_para_coords[para][2] - dict_para_coords[para][3] - total_height) / (item_change+1)
        else:
            spacing_btw_item = 0

        # Iterate over each row in the group
        for index, row in group.iterrows():
            row['dst_pdf_coords'][2] = current_y
            current_y -= row['src_pdf_height']
            row['dst_pdf_coords'][0] = dict_para_coords[para][0]
            row['dst_pdf_coords'][1] = dict_para_coords[para][1]
            row['dst_pdf_coords'][3] = current_y

            # Check if the current row's item_code is different from the next row's item_code
            if index < len(group) - 1 and row['item_code'] != group.iloc[index + 1]['item_code']:
                current_y -= spacing_btw_item

        # Add spacing at the end of the group
        current_y -= spacing_btw_item

    def top_align(self, group, spacing):
        para = group['dst_pdf_para'].iloc[0]
        current_y = dict_para_coords[para][2]

        # Iterate over each row in the group
        for index, row in group.iterrows():
            row['dst_pdf_coords'][2] = current_y
            current_y -= row['src_pdf_height']
            row['dst_pdf_coords'][0] = dict_para_coords[para][0]
            row['dst_pdf_coords'][1] = dict_para_coords[para][1]
            row['dst_pdf_coords'][3] = current_y

            # Check if the current row's item_code is different from the next row's item_code
            if index < len(group) - 1 and row['item_code'] != group.iloc[index + 1]['item_code']:
                current_y -= spacing


