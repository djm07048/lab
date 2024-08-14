import win32gui, win32con
import time


def overwrite_y(title="다른 이름으로 저장 확인", interval=0.1, timeout=2):
    start_time = time.time()
    while True:
        dialog_hwnd = win32gui.FindWindow(None, title)

        if dialog_hwnd:
            win32gui.ShowWindow(dialog_hwnd, win32con.SW_HIDE)
            child_hwnds = []
            win32gui.EnumChildWindows(dialog_hwnd, lambda hwnd, param: param.append(hwnd), child_hwnds)

            for hwnd in child_hwnds:
                text = win32gui.GetWindowText(hwnd)
                if text == "예(&Y)":
                    win32gui.PostMessage(hwnd, win32con.BM_CLICK, 0, 0)
                    break

            break
        else:
            if time.time() - start_time > timeout:
                # print(f"타임아웃: 창 '{title}'을(를) 찾을 수 없습니다.")
                return
            time.sleep(interval)


if __name__ == '__main__':
    overwrite_y()