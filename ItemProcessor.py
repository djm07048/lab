import os
import shutil
import datetime
import PyPDF2
import fitz
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import json
from glob import glob

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
    cont_db_dir = db_dir / 'contDB'
    item_df_path = item_db_dir / 'item_df.json'
    item_cont_df_path = cont_db_dir / 'item_cont_df.json'
    bonus_cont_df_path = cont_db_dir / 'bonus_cont_df.json'
    extra_cont_df_path = cont_db_dir / 'extra_cont_df.json'

    # cont 중에서, item에 의해 생성되는 것은 item_cont,
    # item과 같이 para에 배열되지만, book_type에 따라 다르게 생성되는 것은 bonus_cont,
    # item_cont, bonus_cont와 별개로 배열되는 것은 extra_cont => item_num에 종속적일수도 있고, 아닐수도 있음.
    # item_cont_df는 개별 cont를 각각 별개로 저장함
    # bonus_cont, extra_cont는 합집합적으로 저장함

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
            item_df = pd.DataFrame(columns=['item_code', 'item_pdf_path', 'references', 'domain'])
            self.save_item_df(item_df)
        else:
            with open(self.item_df_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                item_df = pd.DataFrame(data)
                # Ensure the DataFrame has the required columns
                for col in ['item_code', 'item_pdf_path', 'references', 'domain']:
                    if col not in item_df.columns:
                        item_df[col] = None
        return item_df
    def save_item_df(self, item_df):
        with open(self.item_df_path, 'w', encoding='utf-8') as f:
            json.dump(item_df.to_dict(orient='records'), f, ensure_ascii=False, indent=2)

    def load_item_cont_df(self):
        if not self.item_cont_df_path.exists():
            # Create an empty DataFrame with the required columns
            item_cont_df = pd.DataFrame(
                columns=['cont_code', 'item_code', 'item_serial_num', 'cont_type', 'cont_version',
                         'cont_num', 'src_pdf_path', 'src_pdf_page', 'src_pdf_coords', 'src_pdf_height',
                         'src_pdf_width', 'dst_pdf_path', 'dst_pdf_page', 'dst_pdf_para', 'dst_pdf_coords'])
            # Save the empty DataFrame as a JSON file
            self.save_item_cont_df(item_cont_df)
        else:
            with open(self.item_cont_df_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                item_cont_df = pd.DataFrame(data)
                for col in ['cont_code', 'item_code', 'item_serial_num', 'cont_type', 'cont_version', 'cont_num',
                            'src_pdf_path', 'src_pdf_page', 'src_pdf_coords', 'src_pdf_height', 'src_pdf_width',
                            'dst_pdf_path', 'dst_pdf_page', 'dst_pdf_para', 'dst_pdf_coords']:
                    if col not in item_cont_df.columns:
                        item_cont_df[col] = None
        return item_cont_df
    def update_item_cont_df(self, item_code, new_item_cont):
        # 파일이 존재하지 않으면 새로 생성
        if not os.path.exists(self.item_cont_df_path):
            with open(self.item_cont_df_path, 'w', encoding='utf-8') as f:
                json.dump([], f)

        # 기존 데이터 읽기
        with open(self.item_cont_df_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)

        # 기존 데이터에서 새 데이터의 item_code와 일치하지 않는 항목만 유지
        updated_data = [item for item in existing_data if item['item_code'] not in item_code]

        # 새 데이터 추가
        updated_data.extend(new_item_cont)

        # 업데이트된 데이터를 파일에 쓰기
        with open(self.item_cont_df_path, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)

    def save_item_cont_df(self, item_cont_df):
        with open(self.item_cont_df_path, 'w', encoding='utf-8') as f:
            json.dump(item_cont_df.to_dict(orient='records'), f, ensure_ascii=False, indent=2)
    def load_bonus_cont_df(self):
        if not self.bonus_cont_df_path.exists():
            book_cont_df = pd.DataFrame(columns=['cont_code', 'item_code', 'item_serial_num', 'cont_type', 'cont_version',
                                                 'cont_num', 'src_pdf_path', 'src_pdf_page', 'src_pdf_coords', 'src_pdf_height',
                                                 'src_pdf_width', 'dst_pdf_path', 'dst_pdf_page', 'dst_pdf_para', 'dst_pdf_coords'])
        else:
            with open(self.bonus_cont_df_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                book_cont_df = pd.DataFrame(data)
                for col in ['cont_code', 'item_code', 'item_serial_num', 'cont_type', 'cont_version', 'cont_num',
                            'src_pdf_path', 'src_pdf_page', 'src_pdf_coords', 'src_pdf_height', 'src_pdf_width',
                            'dst_pdf_path', 'dst_pdf_page', 'dst_pdf_para', 'dst_pdf_coords']:
                    if col not in book_cont_df.columns:
                        book_cont_df[col] = None
        return book_cont_df
    def update_bonus_cont_df(self, new_bonus_cont):
        # 파일이 존재하지 않으면 새로 생성
        if not os.path.exists(self.bonus_cont_df_path):
            with open(self.bonus_cont_df_path, 'w', encoding='utf-8') as f:
                json.dump([], f)

        # 기존 데이터 읽기
        with open(self.bonus_cont_df_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)

        # 새 데이터 추가
        existing_data.extend(new_bonus_cont)

        # 업데이트된 데이터를 파일에 쓰기
        with open(self.bonus_cont_df_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

    def load_extra_cont_df(self):
        if not self.extra_cont_df_path.exists():
            book_cont_df = pd.DataFrame(columns=['cont_code', 'item_code', 'item_serial_num', 'cont_type', 'cont_version',
                                                 'cont_num', 'src_pdf_path', 'src_pdf_page', 'src_pdf_coords', 'src_pdf_height',
                                                 'src_pdf_width', 'dst_pdf_path', 'dst_pdf_page', 'dst_pdf_para', 'dst_pdf_coords'])
        else:
            with open(self.extra_cont_df_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                book_cont_df = pd.DataFrame(data)
                for col in ['cont_code', 'item_code', 'item_serial_num', 'cont_type', 'cont_version', 'cont_num',
                            'src_pdf_path', 'src_pdf_page', 'src_pdf_coords', 'src_pdf_height', 'src_pdf_width',
                            'dst_pdf_path', 'dst_pdf_page', 'dst_pdf_para', 'dst_pdf_coords']:
                    if col not in book_cont_df.columns:
                        book_cont_df[col] = None
        return book_cont_df

    def update_extra_cont_df(self, new_extra_cont):
        # 파일이 존재하지 않으면 새로 생성
        if not os.path.exists(self.extra_cont_df_path):
            with open(self.extra_cont_df_path, 'w', encoding='utf-8') as f:
                json.dump([], f)

        # 기존 데이터 읽기
        with open(self.extra_cont_df_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)

        # 새 데이터 추가
        existing_data.extend(new_extra_cont)

        # 업데이트된 데이터를 파일에 쓰기
        with open(self.extra_cont_df_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)


class Item(DB):
    # Item-specific DB 관리
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

    def add_item_to_item_df(self):
        if self.item_code not in self.item_df["item_code"].values:
            new_row = {
                "item_code": self.item_code,
                "item_pdf_path": str(self.item_pdf_path),
                "references": {f'ref_{yr}': "" for yr in range(25, 30)},
                "domain": ""
            }
            item_df = self.item_df._append(new_row, ignore_index=True)
            self.save_item_df(item_df)

    def update_item_referecnes(self, references):
        self.item_df.loc[self.item_df['item_code'] == self.item_code, 'references'] = references
    def update_item_domain(self, domain):
        self.item_df.loc[self.item_df['item_code'] == self.item_code, 'domain'] = domain


# ItemProcessor 부분이 이제 잘 동작함
# 각각의 정보를 item_df로 update해주는 부분이 필요함
class ItemProcessor(ItemDf):
    def __init__(self, item_code):
        super().__init__(item_code=item_code)
        self.item_pdf_path = self.item_dir / f'{self.item_code}.pdf'

    def update_item_by_pdf(self, pdf_src_path):
        self.item_code = self.get_item_code(pdf_src_path)
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

        # Add item to item_df
        self.add_item_to_item_df()

        # Update item_cont_df
        self.update_item_cont_df(self.item_code, item_cont_list)
    def update_item_by_pdf_within_folder(self):
        self.make_item_dir()
        self.reset_item_meta_dir()

        # get item_cont_list
        item_png_extractor = ItemPngExtractor(self.item_code, 508)

        # Extract PNG information
        item_cont_list = item_png_extractor.extract_png()

        # Add item to item_df
        self.add_item_to_item_df()

        # Update DataFrame
        self.update_item_cont_df(self.item_code, item_cont_list)

    # folder upload logic 검토 필요
    def upload_item_folder(self, folder_src_path):
        folder_src_path = Path(folder_src_path)
        self.item_code = folder_src_path.name
        pdf_src_path = folder_src_path / f'{self.item_code}.pdf'
        self.update_item_by_pdf(pdf_src_path)
        for file in folder_src_path.iterdir():
            if file.name != f'{self.item_code}.pdf':
                shutil.copy2(file, self.item_dir)
    def upload_mulitple_item_folder(self, folder_src_path):
        for item_folder in folder_src_path.iterdir():
            self.upload_item_folder(item_folder)


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
        src_pdf_path = str(self.item_pdf_path)

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

        cont_code = f'{self.item_code}_00_pbm_pbm'

        pbm_item_cont_list = [{
            "cont_code" : cont_code,
            "item_code" : self.item_code,
            "item_serial_num" : 0,
            "cont_type" : "pbm",
            "cont_version" : "original",
            "cont_num" : 0,
            "src_pdf_path" :src_pdf_path,
            "src_pdf_page" : 0,
            "src_pdf_coords" : [
                pdf_pt_x_lt,
                pdf_pt_x_rt,
                pdf_pt_y_top,
                pdf_pt_y_btm
            ],
            "src_pdf_height" : pdf_pt_y_top - pdf_pt_y_btm,
            "src_pdf_width": (pdf_pt_x_rt-pdf_pt_x_lt),
            "dst_pdf_path" : "dst_pdf_path",
            "dst_pdf_page" : 0,
            "dst_pdf_para" : "dst_pdf_para",
            "dst_pdf_coords" : [0, 0, 0, 0]
        }]
        return pbm_item_cont_list, pbm_png

    def item_sol_png_extractor(self):
        item_png = self.pdf_to_naive_png(self.item_pdf_path)
        src_pdf_path = str(self.item_pdf_path)
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
        item_serial_num = 1
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
                sol_type_dict = ['solAnswer', 'solA', 'solB', 'solC', 'solArrow', 'solClip', 'solClip', 'solWarning', 'solEtc']
                sol_type_num = min(range(len(sol_type_dict)), key=lambda i: abs(255 - i * 30 - avg_g_value))
                sol_png = item_png[start_y + png_px_y_top:end_y + png_px_y_top,
                          int(Ratio.mm_to_png_px(pdf_mm_x_lt, self.dpi_png)):int(
                              Ratio.mm_to_png_px(pdf_mm_x_rt, self.dpi_png))]


                cont_type = sol_type_dict[sol_type_num]

                cont_version = "original"
                cont_code = f'{self.item_code}_{int(item_serial_num):02d}_{cont_type}_{cont_version}'

                sol_item_cont_list.append({
                    "cont_code" : cont_code,
                    "item_code" : self.item_code,
                    "item_serial_num" : 0,
                    "cont_type" : cont_type,
                    "cont_version" : cont_version,
                    "cont_num" : item_serial_num,
                    "src_pdf_path" : src_pdf_path,
                    "src_pdf_page" : 0,
                    "src_pdf_coords" : [
                        Ratio.mm_to_pdf_pt(pdf_mm_x_lt),
                        Ratio.mm_to_pdf_pt(pdf_mm_x_rt),
                        Ratio.mm_to_pdf_pt(pdf_mm_y_top_from_btm - Ratio.png_px_to_mm(start_y + png_px_y_top, self.dpi_png)),
                        Ratio.mm_to_pdf_pt(pdf_mm_y_top_from_btm - Ratio.png_px_to_mm(end_y + png_px_y_top, self.dpi_png))
                    ],
                    "src_pdf_height" : Ratio.mm_to_pdf_pt(Ratio.png_px_to_mm(end_y - start_y, self.dpi_png)),
                    "src_pdf_width": Ratio.mm_to_pdf_pt(pdf_mm_x_rt-pdf_mm_x_lt),
                    "dst_pdf_path" : "dst_pdf_path",
                    "dst_pdf_page" : 0,
                    "dst_pdf_para" : "dst_pdf_para",
                    "dst_pdf_coords" : [0, 0, 0, 0]
                })
                sol_pngs.append(sol_png)

                item_serial_num += 1
        return sol_item_cont_list, sol_pngs


    def extract_png(self):
        self.make_item_dir()
        item_meta_dir = Path(self.item_meta_dir)
        pbm_item_cont_list, pbm_png = self.item_pbm_png_extractor()
        sol_item_cont_list, sol_pngs = self.item_sol_png_extractor()

        item_cont_list_keys = []
        item_cont_list_values = pbm_item_cont_list + sol_item_cont_list

        for i, png in enumerate([pbm_png] + sol_pngs):
            item_cont_list_key = (f'{self.item_code}_{int(item_cont_list_values[i]["cont_num"]):02d}_{item_cont_list_values[i]["cont_type"]}'
                                  f'_{item_cont_list_values[i]["cont_version"]}')
            item_cont_list_keys.append(item_cont_list_key)
            self.save_png(png, item_meta_dir / f'{item_cont_list_key}.png')

        #save item_cont_list_values to item_cont_df
        self.update_item_cont_df(self.item_code, item_cont_list_values)

        return item_cont_list_values


def update_multiple_items(folder_path):
    folder_path = Path(folder_path)
    pdf_files = folder_path.glob('*.pdf')

    for pdf_file in pdf_files:
        pdf_src_path = str(pdf_file)
        item_code = pdf_file.stem
        item_processor = ItemProcessor(item_code)
        item_processor.update_item_by_pdf(pdf_src_path)

DB().load_item_cont_df()
folder_path = r'C:\Users\LEE YONGJOO\PycharmProjects\lab\item_folder'
update_multiple_items(folder_path)