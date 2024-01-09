import pyautogui, time
import pygame

import pygame

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



play_partial_sound('mixkit-spaceship-alarm-998.wav')
input()

while True:
    pyautogui.moveTo(x=1102, y=521)
    if pyautogui.pixel(1102, 521) == (34, 34, 34):
        play_partial_sound('mixkit-spaceship-alarm-998.wav')
        input("Press any key to continue")
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(x=1559, y=628)
    pyautogui.click()
    time.sleep(1)
    if pyautogui.pixel(1165, 570) == (28, 206, 1):
        play_partial_sound('mixkit-spaceship-alarm-998.wav')
        input("On cooldown. Press any key to continue")
        input()
        break
