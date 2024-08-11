import pandas as pd

item_df_data = {
    'item_code': ['E1aaaHY250021', 'E1aaaHY250022'],
    'item_pdf_path': ['path1', 'path2'],
    'reference': [{'25': 'E1wk25주간01_001', '26': 'E1wk26주간01_002'}, {'25': 'E1wk25주간01_003', '26': 'E1wk26주간01_004'}],
    'domain': ['domain1', 'domain2'],
    'item_cont_info': [
        {'E1aaaHY250021_pbm_pbm_01': [0, 1, 2, 3, 4, 'pbm', 'pbm', '01'],
         'E1aaaHY250021_sol_opA_01': [4, 5, 6, 7, 4, 'sol', 'opA', '01'],
         'E1aaaHY250021_sol_opB_02': [4, 5, 6, 7, 4, 'sol', 'opB', '02']},
        {'E1aaaHY250022_pbm_pbm_01': [8, 9, 10, 11, 4, 'pbm', 'pbm', '01'],
         'E1aaaHY250022_sol_opA_01': [12, 13, 14, 15],
         'E1aaaHY250022_sol_opA_02': [12, 13, 14, 15],
         'E1aaaHY250022_sol_opB_03': [12, 13, 14, 15]}
    ]
}

'''reference
f'{test_code}_{item_num}' : item_code'''

'''item_cont_info
f'{item_code}_{cont_type}_{cont_num}' : [x_lt, x_rt, y_top, y_btm, height, cont_type, cont_subtype, cont_num]'''
item_df = pd.DataFrame(item_df_data)

# item_df로부터 test_df 생성
test_code = ['E1']
test_df = item_df[item_df['item_code'].isin(test_code)]


# test_df로부터 item_cont_df 생성
item_cont_data = []
for _, row in test_df.iterrows():
    item_code = row['item_code']
    item_pdf_path = row['item_pdf_path']
    for cont_code, coords in row['item_cont_info'].items():
        cont_type, cont_num = cont_code.split('_')[1:]
        item_cont_data.append({
            'cont_code': cont_code,
            'item_code': item_code,
            'cont_type': cont_type,
            'cont_num': cont_num,
            'src_pdf_path': item_pdf_path,
            'src_pdf_page': 0,
            'src_pdf_coords': coords,
            'dst_pdf_path': None,
            'dst_pdf_page': None,
            'dst_pdf_para': None,
            'dst_pdf_coords': None
        })

item_cont_df = pd.DataFrame(item_cont_data)


#book_cont_df 가져오기
book_cont_data = [
    {
        'cont_code': 'item1_cont1',
        'item_code': 'item1',
        'cont_type': 'cont',
        'cont_num': '1',
        'src_pdf_path': 'path1',
        'src_pdf_coords': [0, 1, 2, 3],
        'src_pdf_page': 0,
        'dst_pdf_path': 'book_path1',
        'dst_pdf_page': 1,
        'dst_pdf_para': 1,
        'dst_pdf_coords': [0, 1, 2, 3]
    },
    {
        'cont_code': 'item2_cont1',
        'item_code': 'item2',
        'cont_type': 'cont',
        'cont_num': '1',
        'src_pdf_path': 'path2',
        'src_pdf_coords': [8, 9, 10, 11],
        'src_pdf_page': 0,
        'dst_pdf_path': 'book_path2',
        'dst_pdf_page': 2,
        'dst_pdf_para': 2,
        'dst_pdf_coords': [8, 9, 10, 11]
    }
]

book_cont_df = pd.DataFrame(book_cont_data)


#cont_df 생성하기
cont_df = pd.concat([item_cont_df, book_cont_df], ignore_index=True)
