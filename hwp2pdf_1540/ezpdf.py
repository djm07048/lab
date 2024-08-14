import subprocess

import win32gui, win32con
import time
import sys


def save_pdf_with_text(file_path="output.pdf", title="생성될 PDF 파일의 이름을 입력 해주세요", interval=0.1, timeout=20):
    start_time = time.time()

    while True:
        dialog_hwnd = win32gui.FindWindow(None, title)

        if dialog_hwnd:
            win32gui.ShowWindow(dialog_hwnd, win32con.SW_HIDE)
            child_hwnds = []
            win32gui.EnumChildWindows(dialog_hwnd, lambda hwnd, param: param.append(hwnd), child_hwnds)

            edit_found = False
            for hwnd in child_hwnds:
                class_name = win32gui.GetClassName(hwnd)
                if class_name == "Edit":
                    win32gui.SendMessage(hwnd, win32con.WM_SETTEXT, None, file_path)
                    edit_found = True
                    break

            if not edit_found:
                print("Edit 컨트롤을 찾을 수 없습니다.")
                return

            for hwnd in child_hwnds:
                text = win32gui.GetWindowText(hwnd)
                if text == "저장(&S)":
                    win32gui.PostMessage(hwnd, win32con.BM_CLICK, 0, 0)
                    break

            break
        else:
            if time.time() - start_time > timeout:
                print(f"타임아웃: 창 '{title}'을(를) 찾을 수 없습니다.")
                return
            time.sleep(interval)


if __name__ == '__main__':
    process0 = subprocess.Popen([sys.executable, "ezpdf_setprop.py", sys.argv[2], sys.argv[3]])
    process1 = subprocess.Popen([sys.executable, "overwrite_y.py"])
    save_pdf_with_text(file_path=sys.argv[1])
    process0.wait()
    process1.wait()
