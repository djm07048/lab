import sys
import os
from glob import glob
import subprocess

from pyhwpx import Hwp

hwp = Hwp(visible=False)


def print_pdf(hwp_list):
    PRINTER_NAME = "ezPDF Builder Supreme"
    DEFAULT = ("210", "297")

    for file in hwp_list:
        hwp.open(file)
        sec_def = hwp.HParameterSet.HSecDef
        hwp.HAction.GetDefault("PageSetup", sec_def.HSet)
        width = hwp.hwp_unit_to_mili(sec_def.PageDef.PaperWidth)
        height = hwp.hwp_unit_to_mili(sec_def.PageDef.PaperHeight)


        pset = hwp.HParameterSet.HPrint
        hwp.HAction.GetDefault("Print", pset.HSet)
        pset.Collate = 1
        pset.UserOrder = 1
        pset.PrintToFile = 0
        pset.PrinterName = PRINTER_NAME
        pset.UsingPagenum = 0
        pset.ReverseOrder = 0
        pset.Pause = 0
        pset.PrintImage = 1
        pset.PrintDrawObj = 1
        pset.PrintClickHere = 0
        pset.PrintAutoFootnoteLtext = "^f"
        pset.PrintAutoFootnoteCtext = "^t"
        pset.PrintAutoFootnoteRtext = "^P쪽 중 ^p쪽"
        pset.PrintAutoHeadnoteLtext = "^c"
        pset.PrintAutoHeadnoteCtext = "^n"
        pset.PrintAutoHeadnoteRtext = "^p"
        pset.PrintFormObj = 1
        pset.PrintMarkPen = 0
        pset.PrintMemo = 0
        pset.PrintMemoContents = 0
        pset.PrintRevision = 1
        pset.PrintBarcode = 1
        pset.Flags = 8192
        pset.Device = hwp.PrintDevice("Printer")
        pset.PrintPronounce = 0

        # subprocess.Popen을 사용해 별도 프로세스로 ezpdf.py 실행
        process1 = subprocess.Popen(["python", "ezpdf.py",
                                     hwp.Path.rsplit(".", maxsplit=1)[0] + ".pdf",
                                     str(width), str(height)])

        # 인쇄 기본 설정 창 미리 열기
        process2 = subprocess.Popen(["rundll32", "printui.dll,PrintUIEntry", "/p", f"/n{PRINTER_NAME}"])

        # ezpdf.py가 백그라운드에서 실행되는 동안 인쇄 작업을 계속 진행
        hwp.HAction.Execute("Print", pset.HSet)
        process1.wait()
        process2.wait()
        print(os.path.basename(hwp.Path))
        hwp.FileClose()
    process3 = subprocess.Popen(["rundll32", "printui.dll,PrintUIEntry", "/p", f"/n{PRINTER_NAME}"])
    subprocess.run([sys.executable, "ezpdf_setprop.py", *DEFAULT])
    process3.wait()


if __name__ == '__main__':
    hwp_list = glob(r"C:\Users\user\Desktop\통상 업무\noname\ex.hwp")
    print_pdf(hwp_list)
    hwp.quit()
