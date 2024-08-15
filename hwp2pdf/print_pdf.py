import sys
import os
from glob import glob
import subprocess
from tqdm import tqdm
from pyhwpx import Hwp

# 시작 전에, ezPDF Builder Supreme 프린터에서 reference_sizes에 해당하는 용지 크기를 추가하고, '자주사용하는용지'로 check해 두어야 함

def change_extension(file_path, new_extension):
    base = os.path.splitext(file_path)[0]
    return f"{base}.{new_extension}"


def identify_item(width, height, tolerance=2):
    size_set = {width, height}

    reference_sizes = {
        (262, 358): "RFAM",
        (263, 371): "CURATION",
        (235, 322): "CONNECTOME",
        (301, 390): "ONTOLOGY",
        (210, 297): "A4",
        (297, 420): "A3"
    }

    for ref_size, item_text in reference_sizes.items():
        if all(any(abs(dim - ref_dim) <= tolerance for ref_dim in ref_size) for dim in size_set):
            return item_text

    return "Unknown size"


def print_pdf(file):
    hwp.open(file)
    sec_def = hwp.HParameterSet.HSecDef
    hwp.HAction.GetDefault("PageSetup", sec_def.HSet)
    width = sec_def.PageDef.PaperWidth
    height = sec_def.PageDef.PaperHeight
    item_text = identify_item(hwp.hwp_unit_to_mili(width), hwp.hwp_unit_to_mili(height))

    prop_window = subprocess.Popen(["rundll32", "printui.dll,PrintUIEntry", "/e", "/n", "ezPDF Builder Supreme"])
    subprocess.run([sys.executable, "ezpdf_select_paper.py", item_text])
    prop_window.wait()

    act = hwp.CreateAction("PrintToPDFEx")
    pset = act.CreateSet()
    act.GetDefault(pset)
    pset.SetItem("FileName", change_extension(f, "pdf"))
    pset.SetItem("Range", hwp.PrintRange("All"))
    pset.SetItem("Collate", 1)
    pset.SetItem("UserOrder", 0)
    pset.SetItem("NumCopy", 1)
    pset.SetItem("PrintToFile", 0)
    pset.SetItem("PrinterName", "ezPDF Builder Supreme")
    pset.SetItem("PrinterPaperSize", item_text)
    pset.SetItem("PrinterPaperWidth", width)
    pset.SetItem("PrinterPaperHeight", height)
    pset.SetItem("UsingPagenum", 1)
    pset.SetItem("ReverseOrder", 0)
    pset.SetItem("Pause", 0)
    pset.SetItem("PrintImage", 1)
    pset.SetItem("PrintDrawObj", 1)
    pset.SetItem("PrintClickHere", 0)
    pset.SetItem("PrintAutoFootnoteLtext", "^f")
    pset.SetItem("PrintAutoFootnoteCtext", "^t")
    pset.SetItem("PrintAutoFootnoteRtext", "^P쪽 중 ^p쪽")
    pset.SetItem("PrintAutoHeadnoteLtext", "^c")
    pset.SetItem("PrintAutoHeadnoteCtext", "^n")
    pset.SetItem("PrintAutoHeadnoteRtext", "^p")
    pset.SetItem("PrintFormObj", 1)
    pset.SetItem("PrintMarkPen", 0)
    pset.SetItem("PrintMemo", 0)
    pset.SetItem("PrintMemoContents", 0)
    pset.SetItem("PrintRevision", 1)
    pset.SetItem("PrintBarcode", 1)
    pset.SetItem("PrintPronounce", 0)
    pset.SetItem("Device", 5)
    pset.SetItem("PrintMethod", 1)

    act.Execute(pset)


if __name__ == '__main__':
    hwp = Hwp(visible=False)
    hwp_list = glob(r"C:\Users\LEE YONGJOO\PycharmProjects\lab\hwp2pdf_0815\*.hwp")
    for f in tqdm(hwp_list):
        print_pdf(f)
    hwp.quit()
