import pyautogui, keyboard, time
import pygame
import datetime, pickle

try:
    with open('date.pkl', 'rb') as f:
        saved_date = pickle.load(f)
except FileNotFoundError:
    saved_date = None

if saved_date == datetime.date.today():
    if input("Coins has been collected for today. Type 'continue' to continue anyways.\n").lower() != "continue":
        raise Exception("Coins has been collected for today")
else:
    with open('date.pkl', 'wb') as f:
        pickle.dump(datetime.date.today(), f)

paused = False


def on_key_press(e):
    global paused
    paused = not paused


keyboard.on_press_key('esc', on_key_press)


def play_partial_sound(sound_file_path, duration_seconds=5):
    pygame.mixer.init()
    sound = pygame.mixer.Sound(sound_file_path)

    # Get the length of the sound in milliseconds
    sound_length = int(sound.get_length() * 1000)

    # Ensure that the specified duration does not exceed the total length of the sound
    duration_ms = min(duration_seconds * 1000, sound_length)

    sound.play(maxtime=duration_ms)

    # You can add a delay if needed
    pygame.time.delay(duration_ms)


captcha_pixel = (34, 34, 34)
bg_color = (31, 33, 41)

captcha_count = 0

while captcha_count <= 2:
    if not paused:
        pyautogui.moveTo(x=1102, y=521)
        if pyautogui.pixel(1102, 521) == captcha_pixel:
            pyautogui.click()
            play_partial_sound('mixkit-spaceship-alarm-998.wav')
            print("captcha detected")
            captcha_count += 1
            time.sleep(3)
            while pyautogui.pixel(1325, 474) != bg_color:
                time.sleep(1)
                print("waiting for captcha")
            print("captcha resolved")
            pyautogui.click(1049, 607)

        pyautogui.click()
        time.sleep(1)
        pyautogui.moveTo(x=1559, y=620)
        pyautogui.click()
        time.sleep(1)
        if pyautogui.pixel(1165, 570) == (28, 206, 1):
            play_partial_sound('mixkit-spaceship-alarm-998.wav')
            print("On cooldown. Press any key to continue")
            break
