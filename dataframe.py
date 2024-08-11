'''
item_df
row = item
coverage = whole items
col = item_code, item_pdf_path, reference, domain, item_cont_info

test_df
mother = item_df
filtered by the items matches test_code
row = item
coverage = items in test
col = item_code, item_pdf_path, item_cont_info

* item_cont_info
dictionary

cont_code =  f'{item_code}_{cont_type}_{cont_num}'
key = cont_code
value = [x_lt, x_rt, y_top, y_btm]

item_cont_df
mother = test_df

row = cont
coverage = conts in items which are in book

for item_cont_info in test_df['item_cont_info']:
    for keys in item_cont_info:
        each key is row.
col = cont_code(from test_df),
item_code(from test_df),
cont_type(from test_df),
cont_num(from test_df),
src_pdf_path(=item_pdf_path from test_df),
src_pdf_coords(=value of item_cont_info),


cont_df
mother = item_cont_df + book_cont_df
row = cont
coverage = item_conts from item_cont_df and book_conts from book_cont_df
col = cont_code, item_code, cont_type, cont_num, src_pdf_path, src_pdf_coords,
dst_pdf_path, dst_pdf_page, dst_pdf_para, dst_pdf_coords (all from book_info)
'''