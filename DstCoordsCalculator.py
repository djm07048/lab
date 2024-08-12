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
        'dst_pdf_path': None,
        'dst_pdf_page': None,
        'dst_pdf_para': None,
        'dst_pdf_coords': None
    }

# merged_cont_df를 받아온다.
# test_type and book_type에 따라 서로 다른 합체 방식을 구현
# 비어있는 dst_pdf_path, dst_page, dst_coords를 채워야 함
# src_pdf_page, src_pdf_coords, src_pdf_path는 주어져 있음

# 1. 모의고사 문제지) test_type = 'ex', book_type = 'pbm'
    '''
    1. filter cont_type = pbm
    
    2. book_cont = Null
    
    3. dst_pdf_path
        test_code로부터 받아온 것에 맞는 pdf file을 선택하면 됨
        
    4. dst_pdf_page & dst_pdf_para
        input으로 받아야 함
        
    5. dst_pdf_coords
        pre-defined para
        for 1a ~ 4a
            varied-spacing btw each cont
            justify-align
        for 4b
            varied-spacing btw each cont
            top-align
            
    6. overlay_multiple_pdf_from_merged_cont_df(merged_cont_df)
    '''

# 2. 모의고사 해설지) test_type = 'ex', book_type = 'sol'

    '''
    1. filter cont_type = sol
    
    2. book_cont = header
    
    3. dst_pdf_path
        test_code로부터 받아온 것에 맞는 pdf file을 선택하면 됨
        유일함
    
    4. dst_pdf_page & dst_pdf_para
        -> calculated
        
    5. dst_pdf_coords
        un-defined para
        fixed-min-spacing btw each cont
        justify-align
    
    '''

# 3. 주간지 문제 파트) test_type = 'wk', book_type = 'pbm'

# 4. 주간지 해설 파트) test_type = 'wk', book_type = 'sol'
