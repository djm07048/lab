import os
import shutil
import datetime
import PyPDF2
import fitz
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd

class Ratio:
    @staticmethod
    def mm_to_png_px(mm, dpi_png):
        return mm * dpi_png / 25.4

    @staticmethod
    def png_px_to_mm(png_px, dpi_png):
        return png_px * 25.4 / dpi_png

    @staticmethod
    def pdf_pt_to_mm(pdf_pt):
        return pdf_pt * (25.4 / 72)

    @staticmethod
    def mm_to_pdf_pt(mm):
        return mm * (72 / 25.4)

    @classmethod
    def png_px_to_pdf_pt(cls, png_px, dpi_png):
        return cls.mm_to_pdf_pt(cls.png_px_to_mm(png_px, dpi_png))

    @classmethod
    def pdf_pt_to_png_px(cls, pdf_pt, dpi_png):
        return cls.mm_to_png_px(cls.pdf_pt_to_mm(pdf_pt), dpi_png)

    @staticmethod
    def get_height_pdf_pt(pdf_path):
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            page = pdf_reader.pages[0]
            media_box = page.mediabox
            return float(media_box.height)

class Template:
    # Template for 문항 양식
    @staticmethod
    def pbm_area():
        # x_lt, x_rt, y_top_from_top, y_btm_from_top, y_top_from_btm, y_btm_from_btm
        return 23, 135.5, 13, 420, 407, 0

    @staticmethod
    def sol_area():
        # x_lt, x_rt, y_top_from_top, y_btm_from_top, y_top_from_btm, y_btm_from_btm
        return 174, 274, 24, 420, 396, 0

class DB:
    db_dir = Path(r'C:\Users\LEE YONGJOO\PycharmProjects\lab\DB')
    item_db_dir = db_dir / 'itemDB'
    test_db_dir = db_dir / 'testDB'
    book_db_dir = db_dir / 'bookDB'
    item_df_path = item_db_dir / 'item_df.csv'
    
    @staticmethod
    def get_item_code(src_pdf_path):
        filename_with_ext = os.path.basename(src_pdf_path)
        filename_without_ext = os.path.splitext(filename_with_ext)[0]
        return filename_without_ext

    @staticmethod
    def get_pdf_creation_date(pdf_path):
        creation_time = os.path.getctime(pdf_path)
        date_obj = datetime.datetime.fromtimestamp(creation_time)
        return date_obj.strftime('%y%m%d_%H%M%S')

    def load_item_df(self):
        if not self.item_df_path.exists():
            item_df = pd.DataFrame(columns=['item_code', 'item_pdf_path', 'reference', 'domain', 'item_cont_list'])
        else:
            pass
        item_df = pd.read_csv(self.item_df_path)
        return item_df
    
    def save_item_df(self, item_df):
        item_df.to_csv(self.item_df_path, index=False)

    def load_book_df(self):
        # book_cont_df는 row=book, col=book_code, range=whole book인 df임, 이때 book은 test 하나당 1개 이상 존재함
        # col : test_code, book_code, book_type
        book_df_path = self.book_db_dir / 'book_df.csv'
        book_df = pd.read_csv(book_df_path)
        return book_df

    def load_book_cont_all_df(self):
        book_cont_all_df_path = self.book_db_dir / 'book_cont_all_df.csv'
        book_cont_all_df = pd.read_csv(book_cont_all_df_path)
        return book_cont_all_df


class Item(DB):
    #Item-specific DB 관리
    def __init__(self, item_code):
        super().__init__()
        self.item_code = item_code

        self.item_subject = self.item_code[:2]
        self.item_topic = self.item_code[2:5]
        self.item_author = self.item_code[5:7]
        self.item_year = self.item_code[7:9]
        self.item_num = self.item_code[9:13]

        self.item_subject_dir = self.item_db_dir / self.item_subject
        self.item_topic_dir = self.item_subject_dir / self.item_topic
        self.item_dir = self.item_topic_dir / self.item_code
        self.item_meta_dir = self.item_dir / 'metadata'
        self.item_legacy_dir = self.item_dir / 'legacy'
        self.item_pdf_path = self.item_dir / f'{self.item_code}.pdf'


    def make_item_dir(self):
        self.item_dir.mkdir(parents=True, exist_ok=True)
        self.item_legacy_dir.mkdir(exist_ok=True)
        self.item_meta_dir.mkdir(exist_ok=True)
        
    def reset_item_meta_dir(self):
        if self.item_meta_dir.exists():
            shutil.rmtree(self.item_meta_dir)
        self.item_meta_dir.mkdir(exist_ok=True)


class ItemDf(Item):
    # Item 하나의 row를 ItemDF에서 관리한다.
    '''
    row = item
    coverage = whole items
    col = item_code, item_pdf_path, ref_25 ~ ref_44, domain, item_cont_list
    '''
    
    def __init__(self, item_code):
        super().__init__(item_code)
        self.item_df = self.load_item_df()

    def add_empty_row_to_item_df(self):
        self.item_df.append({'item_code': self.item_code,
                        'item_pdf_path': self.item_pdf_path,
                        'reference': None,
                        'domain': None,
                        'item_cont_list': None #item_cont_list는 item_cont들이 담겨있는 리스트임
                        }, ignore_index=True)

    def update_item_cont_list(self, item_cont_list):
        self.item_df.loc[self.item_df['item_code'] == self.item_code, 'item_cont_list'] = item_cont_list

    def update_item_references(self, references):
        self.item_df.loc[self.item_df['item_code'] == self.item_code, 'references'] = references

    def update_item_domain(self, domain):
        self.item_df.loc[self.item_df['item_code'] == self.item_code, 'domain'] = domain


# ItemProcessor 부분이 잘 동작하지 않음
class ItemProcessor(ItemDf):
    def __init__(self, item_code):
        super().__init__(item_code)

    def upload_item_by_pdf(self, pdf_src_path):
        self.make_item_dir()
        self.reset_item_meta_dir()
        if self.item_pdf_path.exists():
            creation_date = self.get_pdf_creation_date(self.item_pdf_path)
            new_filename = f"{self.item_code}_{creation_date}{self.item_pdf_path.suffix}"
            shutil.move(self.item_pdf_path, self.item_legacy_dir / new_filename)
        shutil.copy(pdf_src_path, self.item_pdf_path)

        # get item_cont_list
        item_png_extractor = ItemPngExtractor(self.item_code, 508)

        # Extract PNG information
        item_cont_list = item_png_extractor.extract_png()

        # Load existing DataFrame
        item_df = self.load_item_df()

        # Update DataFrame
        if not self.item_code in item_df['item_code'].values:
            item_df = self.add_empty_row_to_item_df(item_df, self.item_code)
        self.update_item_cont_list(item_df, self.item_code, item_cont_list)

        # Save updated DataFrame
        self.save_item_df(item_df)


class ItemPngExtractor(ItemDf):
    def __init__(self, item_code, dpi_png):
        super().__init__(item_code)
        self.dpi_png = dpi_png

    def pdf_to_naive_png(self, pdf_path):
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(self.dpi_png / 72, self.dpi_png / 72))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return np.array(img)

    @staticmethod
    def save_png(naive_png, filename):
        img = Image.fromarray(naive_png)
        img.save(filename)

    def item_pbm_png_extractor(self):
        item_png = self.pdf_to_naive_png(self.item_pdf_path)
        height, _, _ = item_png.shape

        pdf_mm_x_lt, pdf_mm_x_rt, pdf_mm_y_top_from_top, _, pdf_mm_y_top_from_btm, _ = Template.pbm_area()
        png_px_x_lt = Ratio.mm_to_png_px(pdf_mm_x_lt, self.dpi_png)
        png_px_x_rt = Ratio.mm_to_png_px(pdf_mm_x_rt, self.dpi_png)
        png_px_y_top = Ratio.mm_to_png_px(pdf_mm_y_top_from_top, self.dpi_png)
        png_px_y_btm_default = height

        cropped = item_png[int(png_px_y_top):png_px_y_btm_default, int(png_px_x_lt):int(png_px_x_rt)]

        red_pixels = np.where((cropped == [255, 0, 0]).all(axis=2))
        px_match = list(zip(red_pixels[1] + int(png_px_x_lt), red_pixels[0] + int(png_px_y_top)))
        png_px_y_btm = min((y for _, y in px_match), default=png_px_y_btm_default) - 20
        png_px_y_btm_from_btm = height - png_px_y_btm
        pbm_png = item_png[int(png_px_y_top):int(png_px_y_btm), int(png_px_x_lt):int(png_px_x_rt)]
        pdf_pt_x_lt = Ratio.mm_to_pdf_pt(pdf_mm_x_lt)
        pdf_pt_x_rt = Ratio.mm_to_pdf_pt(pdf_mm_x_rt)
        pdf_pt_y_top = Ratio.mm_to_pdf_pt(pdf_mm_y_top_from_btm)
        pdf_pt_y_btm = Ratio.png_px_to_pdf_pt(png_px_y_btm_from_btm, self.dpi_png)

        pbm_item_cont_list = [[pdf_pt_x_lt, pdf_pt_x_rt, pdf_pt_y_top, pdf_pt_y_btm, pdf_pt_y_top - pdf_pt_y_btm, 'pbm', 'pbm', '00']]

        return pbm_item_cont_list, pbm_png

    def item_sol_png_extractor(self):
        item_png = self.pdf_to_naive_png(self.item_pdf_path)
        height, _, _ = item_png.shape
        pdf_mm_x_lt, pdf_mm_x_rt, pdf_mm_y_top_from_top, _, pdf_mm_y_top_from_btm, _ = Template.sol_area()
        png_px_x_lt_scan = int(Ratio.mm_to_png_px(141, self.dpi_png))
        png_px_x_rt_scan = png_px_x_lt_scan + 1
        png_px_y_top = int(Ratio.mm_to_png_px(pdf_mm_y_top_from_top, self.dpi_png))
        png_px_y_btm_default = height
        cropped = item_png[png_px_y_top:png_px_y_btm_default, png_px_x_lt_scan:png_px_x_rt_scan]

        changes = []
        prev_color = np.array([255, 255, 255])
        for y, color in enumerate(cropped[:, 0, :3]):
            if not np.array_equal(color, prev_color):
                if np.array_equal(prev_color, [255, 255, 255]):
                    changes.append((y, 'start'))
                elif np.array_equal(color, [255, 255, 255]):
                    changes.append((y, 'end'))
            prev_color = color
        sol_item_cont_list = []
        sol_pngs = []
        sol_serial_num = 1
        for i in range(0, len(changes) - 1, 2):
            if changes[i][1] == 'start' and changes[i + 1][1] == 'end':
                start_y, end_y = changes[i][0], changes[i + 1][0]
                range_start = max(start_y + 5, 0)
                range_end = min(end_y - 5, cropped.shape[0] - 1)
                if range_start < range_end:
                    g_values = cropped[range_start:range_end, 0, 1]
                    avg_g_value = int(np.mean(g_values))
                else:
                    avg_g_value = cropped[start_y, 0, 1]
                sol_subtype_dict = ['ans', 'opA', 'opB', 'opC', 'arw', 'clp', 'chk', 'wrn', 'etc']
                sol_subtype_num = min(range(len(sol_subtype_dict)), key=lambda i: abs(255 - i * 30 - avg_g_value))
                sol_png = item_png[start_y + png_px_y_top:end_y + png_px_y_top,
                          int(Ratio.mm_to_png_px(pdf_mm_x_lt, self.dpi_png)):int(
                              Ratio.mm_to_png_px(pdf_mm_x_rt, self.dpi_png))]
                sol_item_cont_list.append([
                    Ratio.mm_to_pdf_pt(pdf_mm_x_lt),
                    Ratio.mm_to_pdf_pt(pdf_mm_x_rt),
                    Ratio.mm_to_pdf_pt(
                        pdf_mm_y_top_from_btm - Ratio.png_px_to_mm(start_y + png_px_y_top, self.dpi_png)),
                    Ratio.mm_to_pdf_pt(
                        pdf_mm_y_top_from_btm - Ratio.png_px_to_mm(end_y + png_px_y_top, self.dpi_png)),
                    Ratio.mm_to_pdf_pt(Ratio.png_px_to_mm(end_y - start_y, self.dpi_png)),
                    'sol',
                    sol_subtype_dict[sol_subtype_num],
                    sol_serial_num
                ])
                sol_pngs.append(sol_png)
                sol_serial_num += 1

        return sol_item_cont_list, sol_pngs

    def extract_png(self):
        self.make_item_dir()
        item_meta_dir = Path(self.item_meta_dir)
        pbm_item_cont_list, pbm_png = self.item_pbm_png_extractor()
        sol_item_cont_list, sol_pngs = self.item_sol_png_extractor()

        item_cont_list_keys = []
        item_cont_list_values = pbm_item_cont_list + sol_item_cont_list

        for i, png in enumerate([pbm_png] + sol_pngs):
            item_cont_list_key = f'{self.item_code}_{item_cont_list_values[i][5]}_{item_cont_list_values[i][6]}_{int(item_cont_list_values[i][7]):02d}'
            item_cont_list_keys.append(item_cont_list_key)
            self.save_png(png, item_meta_dir / f'{item_cont_list_key}.png')

        return item_cont_list_values

ItemPngExtractor('E1aaaHY250023', 508).extract_png()


# 여기부터 '수정' 필요~
class Test(DB):
    def __init__(self, test_code):
        self.test_code = test_code
        self.test_subject = self.test_code[0:2]
        self.test_type = self.test_code[2:4]
        self.test_year = self.test_code[4:6]
        self.test_name = self.test_code[6:8]
        self.test_num = self.test_code[8:10]
        '''
        Test 아래에
        ex: 모의고사
        mn: 월간지
        wk: 주간지        
        '''
        self.test_subject_dir = self.test_db_dir / self.test_subject
        self.test_type_dir = self.test_subject_dir / f'{self.test_type}'
        self.test_year_dir = self.test_type_dir / f'{self.test_year}'
        self.test_dir = self.test_year_dir / self.test_code
        self.test_item_dir = self.test_dir / 'item'

    def make_test_dir(self):
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.test_item_dir.mkdir(exist_ok=True)
    def get_test_df(self):
        item_df = DB.load_item_df()
        test_df = item_df[item_df['reference'].apply(lambda ref: self.test_year in ref and ref[self.test_year][:10] == self.test_code)]
        test_df['item_serial_num'] = test_df['reference'].apply(lambda ref: int(ref[self.test_year][11:14]))

        # Sort test_df by item_serial_num
        test_df = test_df.sort_values(by='item_serial_num')

        return test_df

    # test_df의 하나의 row를 update하는 코드도 필요

    def get_item_cont_df(self):
        test_df = self.get_test_df(self.test_code)
        item_cont_data = []
        for _, row in test_df.iterrows():
            item_code = row['item_code']
            item_pdf_path = row['item_pdf_path']
            item_serial_num = row['item_serial_num']
            for cont_code, coords in row['item_cont_list'].items():
                cont_type, cont_sub_type, cont_num = cont_code.split('_')[1:]
                item_cont_data.append({
                    'cont_code': cont_code,
                    'item_code': item_code,
                    'item_serial_num': item_serial_num,
                    'cont_type': cont_type,
                    'cont_sub_type': cont_sub_type,
                    'cont_num': cont_num,       #item 내에서 모든 cont에 대한 순번임을 유의
                    'src_pdf_path': item_pdf_path,
                    'src_pdf_page': 0,
                    'src_pdf_coords': coords[0:3],
                    'src_pdf_height': coords[4],
                    'dst_pdf_path': None,
                    'dst_pdf_page': None,
                    'dst_pdf_para': None,
                    'dst_pdf_coords': None
                })

        item_cont_df = pd.DataFrame(item_cont_data)
        return item_cont_df

class Book(DB):
    def __init__(self, book_code):
        self.book_code = book_code
        # book_type: pbm, sol, ans, ...
        # book_num: 01, 02, 03, ...
        self.test_code, self.book_type, self.book_num\
            = self.book_code.split('_')[0], self.book_code.split('_')[1], self.book_code.split('_')[2]
        self.book_cont_all_df = self.open_book_cont_all_df_from_csv()
    def get_book_cont_df(self, book_code):
        #어떤 규칙으로 book_cont_all_df의 일부에서 book_cont_df를 가져올지 생각해야함
        book_cont_df = self.book_cont_all_df[self.book_cont_all_df['book_code'] == book_code]
        return book_cont_df
class MergedContDf(Book, ItemDf):
    def __init__(self):
        super().__init__()

    def merge_and_fill_merged_cont_df(self, merged_cont_df):
        # item_cont_df와 book_cont_df를 가져온다.
        item_cont_df = self.get_item_cont_df(self.test_code)
        book_cont_df = self.get_book_cont_df(self.book_code)
        '''  col: item_cont_df, book_cont_df
            'cont_code': cont_code,
            'item_code': item_code,
            'item_serial_num': item_serial_num,
            'cont_type': cont_type,
            'cont_sub_type': cont_sub_type,
            'cont_num': cont_num,
            'src_pdf_path': item_pdf_path,
            'src_pdf_page': 0,
            'src_pdf_coords': coords,
            'dst_pdf_path': None,
            'dst_pdf_page': None,
            'dst_pdf_para': None,
            'dst_pdf_coords': None
        '''
        # item_cont_df의 빈 값들을 채운다.

        # book_cont_df의 빈 값들을 채운다.

        # item_cont_df와 book_cont_df를 병합한다

        merged_cont_df = pd.merge(item_cont_df, book_cont_df, on='cont_code')
        return merged_cont_df

    dict_cont_type_serial_num = {
        'header': 0,
        'pbm_pbm': 10,
        'sol_ans': 20,
        'sol_aaa': 21,
        'sol_bbb': 22,
        'sol_ccc': 23,
        'sol_arw': 24,
        'sol_clp': 25,
        'sol_chk': 26,
        'sol_wrn': 27,
        'sol_etc': 28,
        'footer': 30,
        'blank': 40
    }

    def sort_merged_cont_df(self, merged_cont_df):
        # Map cont_type to cont_type_serial_num
        merged_cont_df['cont_type_serial_num'] = merged_cont_df['cont_type'].map(self.dict_cont_type_serial_num)

        # Sort by item_serial_num, cont_type_serial_num, and cont_num
        merged_cont_df = merged_cont_df.sort_values(by=['item_serial_num', 'cont_type_serial_num', 'cont_num'])

        return merged_cont_df

    # 이렇게 얻어진 merged_cont_df는 src와 dst가 모두 명확하게 기재된 cont_df가 된다.
    # para가 predefined인지... 아니면 사용자가 지정해야 하는지에 따라서, dst_pdf_para를 채워주는 코드가 필요하다.


# 여기서부터는 처음부터 다시 작성해야 함

class TestParaManager(ItemContDf):
    def __init__(self, test_code):
        super().__init__(test_code)
        self.test_info_df = self.get_test_info_df(self.open_test_df_from_csv(), self.test_code)

    def pbm_varied_spacing_calculator(self):
        item_by_para = self.group_item_by_para()
        test_info_df = self.test_info_df
        dict_para_pdf_pt = self.test_para_info()

        for para, items in item_by_para:
            x_left, x_right, y_upper, y_lower, subtract_one = dict_para_pdf_pt[para]
            item_ht_list = [test_info_df['item_ht_list'][test_info_df.index(item)] for item in items]
            header_height = self.header()
            footer_height = self.footer()
            total_item_height = sum(item_ht_list)
            total_height = y_upper - y_lower
            blank_num = len(item_ht_list) - 1 if subtract_one else len(item_ht_list)
            blank_avg_height = (total_height - total_item_height - header_height - footer_height) / blank_num

            y_coord = y_upper
            for i, item_height in enumerate(item_ht_list):
                header_y_upper = y_coord
                y_coord -= header_height
                item_y_upper = y_coord
                y_coord -= item_height
                footer_y_upper = y_coord
                y_coord -= footer_height
                if i < blank_num:
                    y_coord -= blank_avg_height

                self.para_item_indices[items[i]] = {
                    'header_y_upper': header_y_upper,
                    'item_y_upper': item_y_upper,
                    'footer_y_upper': footer_y_upper,
                }

        self.test_df['pbm_y_coords'] = self.test_df['item_code_list'].apply(lambda x: self.para_item_indices.get(x, {}))

    def sol_brutal_spacing_calculator(self, y_upper, y_lower, item_ht_list, subtract_one):
        None
    '''
    여기서 보충이 필요함
    item_df에 sol_ht_list를 추가해야함 png 추출한 직후에 바로 업데이트 되도록 해야 함
    pbm도 동일한 원리로 구현해야 함
    각각의 pbm_ht_list와 sol_ht_list로부터, 요소를 가져와서 배치하도록.
    이 함수는 sol을 가져다가 배치하는 것으로 구현
    문항- 문항 간 spacing은 brutal하게 배치된 후, 위아래로 뻗치는 방식(서바 해설지->varied_spacing)으로 구현
    or 위로 쭉 매달지(->fixed_spacing)로 구현
    '''

    def sol_fixed_spacing_calculator(self, y_upper, y_lower, item_ht_list, subtract_one):
        None

    def pbm_fixed_spacing_calculator(self, y_upper, y_lower, item_ht_list, subtract_one):
        None


class ExCollocator(TestParaManager):
    def __init__(self, test_code):
        super().__init__(test_code)
    '''
    pbm은 varied_spacing으로 배치
    sol은 brutal_spacing으로 배치
    '''
class WkCollocator(TestParaManager):
    def __init__(self, test_code):
        super().__init__(test_code)
    '''
    pbm은 varied spacing으로 배치
    sol은 brutal_spacing으로 배치
    '''

class Book(Test):
    def __init__(self, test_code, book_type, book_num):
        # book_type: pbm, sol, ans, ...
        # book_num: 01, 02, 03, ...
        super().__init__(test_code)
        self.test_db_dir, _, _ = self.define_dir()
        self.test_base_dir = self.test_db_dir / 'base'
        self.test_subject, self.test_type, self.test_year, self.test_name, self.test_num = self.define_test()
        book_code = f'{test_code}_{book_type}_{book_num:02d}'
        book_type = f'{self.test_type}_{book_type}'

    def test_base_pbm_pdf(self):
        # get the base pdf file for the test
        #E1wk25주간01
        base_pdf_path = self.test_base_dir / f'{self.test_code}_pbm.pdf'
        return base_pdf_path

    def test_base_sol_pdf(self):
        # get the base pdf file for the test
        #E1wk25주간01
        base_pdf_path = self.test_base_dir / f'{self.test_code}_sol.pdf'
        return base_pdf_path

    def test_para_info(self):
        _, test_type, _, _, _ = self.define_test()
        if test_type == 'ex':
            dict_para_pdf_pt = {
                # x_left, x_right, y_upper, y_lower, subtract_one
                '1a': (87.87, 406.74, 932.26, 123.37, True),
                '1b': (433.64, 752.51, 932.26, 123.37, True),
                '2a': (87.87, 406.74, 1031.53, 123.37, True),
                '2b': (433.64, 752.51, 1031.53, 123.37, True),
                '3a': (87.87, 406.74, 1031.53, 123.37, True),
                '3b': (433.64, 752.51, 1031.53, 123.37, True),
                '4a': (87.87, 406.74, 1031.53, 123.37, True),
                '4b': (433.64, 752.51, 1031.53, 144.01, False)
            }
        elif test_type == 'mn':
            dict_para_pdf_pt = {
                '1a': (87.87, 406.74, 932.26, 123.37, True)
            }
            # Implement more test_para_pdf_pt for mnly
        elif test_type == 'wk':
            dict_para_pdf_pt = {
                '1a': (87.87, 406.74, 932.26, 123.37, True)
            }
            # Implement more test_para_pdf_pt for wkly
        else:
            dict_para_pdf_pt = None
        return dict_para_pdf_pt

class MakeBook(Book):
    def __init__(self, test_code, book_type, book_num):
        super().__init__(test_code, book_type, book_num)
        self.test_db_dir, _, _ = self.define_dir()
        self.test_base_dir = self.test_db_dir / 'base'
        self.test_subject, self.test_type, self.test_year, self.test_name, self.test_num = self.define_test()
        book_code = f'{test_code}_{book_type}_{book_num:02d}'
        book_type = f'{self.test_type}_{book_type}'

#여기부터 MakeBook은 교재별로 다르게 만들어서 수행
class MakeExPbl(MakeBook):
    None

class MakeExSol(MakeBook):
    None

class MakeExAns(MakeBook):
    None

class MakeWkPbl(MakeBook):
    None

if __name__ == "__main__":
    item_code = "E1aaaHY250023"
    pdf_src_path = Path(r"E1aaaHY250023.pdf")
    processor = ItemProcessor(item_code)
    processor.upload_item_by_pdf(pdf_src_path)
    print(f"Item {item_code} processed successfully.")