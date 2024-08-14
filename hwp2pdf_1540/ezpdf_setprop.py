import subprocess
import sys
import time
import win32gui
import win32con


def printer_prop(button):
    title = "ezPDF Builder Supreme 속성"

    dialog_hwnd = None
    max_wait_time = 20  # 최대 10회 시도 (1초 대기)

    for _ in range(max_wait_time):
        dialog_hwnd = win32gui.FindWindow(None, title)
        if dialog_hwnd:
            win32gui.ShowWindow(dialog_hwnd, win32con.SW_HIDE)
            break
        # time.sleep(0.1)  # 0.1초 대기

    if dialog_hwnd:
        child_hwnds = []
        win32gui.EnumChildWindows(dialog_hwnd, lambda hwnd, param: param.append(hwnd), child_hwnds)

        for hwnd in child_hwnds:
            text = win32gui.GetWindowText(hwnd)
            # print(text, hwnd)
            if text.startswith(button):
                win32gui.PostMessage(hwnd, win32con.BM_CLICK, 0, 0)
                break
    else:
        print(f"Window with title '{title}' not found.")

    return dialog_hwnd


if __name__ == '__main__':
    process = subprocess.Popen([sys.executable, "ezpdf_select_paper.py", sys.argv[1], sys.argv[2]])
    hwnd = printer_prop("기본 설정(&E)")
    process.wait()
    printer_prop("확인")
