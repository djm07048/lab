import sys
import time
import win32gui
import win32con
import win32api


def select_combobox_and_click_button():
    item_text = sys.argv[1]  # item_text

    combo_box_index = 0
    window_title = "ezPDF Builder Supreme 인쇄 기본 설정"
    button_text = "확인"

    hwnd = None
    max_wait_time = 5  # 최대 50회 시도 (5초 대기)

    for _ in range(max_wait_time):
        hwnd = win32gui.FindWindow(None, window_title)
        if hwnd != 0:
            win32gui.ShowWindow(hwnd, win32con.SW_HIDE)
            time.sleep(1)
            break
        time.sleep(1)  # 0.1초 대기

    if hwnd == 0:
        print(f"Window with title '{window_title}' not found.")
        return

    # 2. 첫 번째 ComboBox의 핸들을 찾기 위해 EnumChildWindows 사용
    def enum_child_windows_callback(hwnd, hwnd_list):
        class_name = win32gui.GetClassName(hwnd)
        if class_name == "ComboBox":
            hwnd_list.append((hwnd, class_name))
        elif class_name == "Button":
            hwnd_list.append((hwnd, class_name))

    child_windows = []
    win32gui.EnumChildWindows(hwnd, enum_child_windows_callback, child_windows)

    # 3. 특정 인덱스에 있는 ComboBox의 핸들을 가져옴
    combo_boxes = [hwnd for hwnd, class_name in child_windows if class_name == "ComboBox"]

    if len(combo_boxes) <= combo_box_index:
        print(f"ComboBox with index {combo_box_index} not found.")
        return

    combo_hwnd = combo_boxes[combo_box_index]

    win32gui.SendMessage(combo_hwnd, win32con.CB_SELECTSTRING, -1, item_text)

    for hwnd, class_name in child_windows:
        if class_name == "Button":
            button_text_current = win32gui.GetWindowText(hwnd)
            if button_text_current == button_text:
                win32api.PostMessage(hwnd, win32con.BM_CLICK, 0, 0)
                break
    else:
        pass


if __name__ == '__main__':
    select_combobox_and_click_button()
