#!/usr/bin/env python

import pygame as pg

kb_file = open("my_keyboard.kb", 'w')

pg.init()
screen = pg.display.set_mode((200, 200))
print("Press the keys in the right order. Press Escape to finish.")
while True:
    event = pg.event.wait()
    #print(event.type)
    if event.type == pg.KEYDOWN:
        if event.key == pg.K_ESCAPE:
            break
        else:
            name = pg.key.name(event.key)
            print("Last key pressed: %s" % name)
            kb_file.write(name + '\n')

    # keypressed = pg.key.get_pressed()
    # if keypressed[pg.K_w]:
    #     print("it worked")
    # else:
    #     print("It didn't work")

    # pg.event.pump()

kb_file.close()
print("Done. you have a new keyboard configuration file: %s" % (kb_file.name))
pg.quit()







'''
pygame.mixer.init(fps, -16, 1, 512) # so flexible ;)
screen = pygame.display.set_mode((640,480)) # for the focus

# Get a list of the order of the keys of the keyboard in right order.
# ``keys`` is like ['Q','W','E','R' ...] 
keys = open('typewriter.kb').read().split('\n')
print(keys)

sounds = map(pygame.sndarray.make_sound, transposed)
key_sound = dict(zip(keys, sounds))
is_playing = {k: False for k in keys}

while True:
    pygame.event.pump()
    keys = pygame.key.get_pressed()
    event =  pygame.event.wait()
    s = pygame.event.get()
    print(s)

    if event.type in (pygame.KEYDOWN, pygame.KEYUP):
        key = pygame.key.name(event.key)


    if event.type == pygame.KEYDOWN:

        if (key in key_sound.keys()) and (not is_playing[key]):
            key_sound[key].play(fade_ms=50)
            is_playing[key] = True

        elif event.key == pygame.K_ESCAPE:
            pygame.quit()
            raise KeyboardInterrupt

    elif event.type == pygame.KEYUP and key in key_sound.keys():

        key_sound[key].fadeout(50) # stops with 50ms fadeout
        is_playing[key] = False
'''