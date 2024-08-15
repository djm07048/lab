from ItemProcessor import DB, Ratio, Template

'''
item_cont_df -> book_item_cont_df
bonus_cont_df -> book_bonus_cont_df

book_item_cont_df + book_bonus_cont_df 중에서, para cont -> book_cont_df

'''

# 다시 머리가 꼬이는 중.
# 1. cont_df 없으면 만들도록 코드 수정
# 2. Book에 대하여 지정해야 할 수많은 정보들을 어떻게 정리할지 결정


class Test:
    def __init__(self, test_code):
        self.test_code = test_code
        self.test_subject = test_code[0:2]
        self.test_type = test_code[2:4]
        self.test_year = test_code[4:6]
        self.test_name = test_code[6:8]
        self.test_num = test_code[8:10]
        # E1wk25주간01

        self.item_df = DB().load_item_df()

    def load_test_item_df_from_item_df(self):
        # Filter rows where the reference value matches the test_code
        test_item_df = self.item_df[self.item_df.apply(
            lambda row: row['references'].get(f'ref_{self.test_year}', '')[:10] == self.test_code, axis=1
        )]

        # Add item_serial_num column
        test_item_df['item_serial_num'] = test_item_df.apply(
            lambda row: int(row['references'].get(f'ref_{self.test_year}', '')[10:]), axis=1
        )

        return test_item_df
class Book(Test):
    def __init__(self, book_code, cont_version_dict):
        test_code = book_code.split('_')[0]
        super().__init__(test_code)
        self.book_code = book_code
        self.book_type = book_code.split('_')[1]
        self.book_num = book_code.split('_')[2]
        self.cont_version_dict = cont_version_dict

    def load_book_item_cont_df(self):
        item_cont_df = DB().load_item_cont_df()
        test_item_df = self.load_test_item_df_from_item_df()
        book_item_cont_df = item_cont_df[item_cont_df['item_code'].isin(test_item_df['item_code'])]
        book_item_cont_df = book_item_cont_df[book_item_cont_df['cont_type'].isin(self.cont_version_dict.keys())]
        book_item_cont_df = book_item_cont_df[book_item_cont_df.apply(
            lambda row: self.cont_version_dict[row['cont_type']] == row['cont_version'], axis=1
        )]
        for index, row in book_item_cont_df.iterrows():
            item_code = row['item_code']
            test_row = test_item_df[test_item_df['item_code'] == item_code].iloc[0]
            item_serial_num = int(test_row['references'][f'ref_{self.test_year}'][10:])
            book_item_cont_df.at[index, 'item_serial_num'] = item_serial_num
        return book_item_cont_df

    def load_book_para_cont_df(self):
        book_bonus_cont_df = DB.load_bonus_cont_df(self)
        book_bonus_cont_df = book_bonus_cont_df[book_bonus_cont_df['cont_type'].isin(self.cont_version_dict.keys())]
        book_bonus_cont_df = book_bonus_cont_df[book_bonus_cont_df.apply(
            lambda row: self.cont_version_dict[row['cont_type']] == row['cont_version'], axis=1
        )]
        book_para_cont_df = book_bonus_cont_df.append(self.load_book_item_cont_df(), ignore_index=True)
        return book_para_cont_df




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