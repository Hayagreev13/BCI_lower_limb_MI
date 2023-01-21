#!/usr/bin/env python
# -*- coding: utf-8 -*-

from psychopy import visual, core, event
import numpy as np


# time for the trial
calTime = 10.0
restTime = 3.0
cueTime = 2.0
focusTime = 10.0
blkTime = 5.0


# --------- Preparing Ready Window --------
win = visual.Window(
    size=(1440,900), fullscr=True, screen=0, #change this for dual screen
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color='black', colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')


# -----------Initializing stimuli to be shown -------
# Initialize components for Routine "10 sec calibration"
ten_sec = visual.ShapeStim(
    win=win, name='ten_sec',color = 'black',
    size=(0.044, 0.044), vertices='circle', # change size to 0.045,0.045 if zoomed in
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='darkgrey', fillColor=(0.3255,0.3255,0.3255),
    opacity=None, depth=-2.0, interpolate=True)

# Initialize components for Routine "trial"
restCross = visual.ImageStim(
    win=win, name='RestCross',
    image='./images/VC_Cross.jpg', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=None,
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
Cue = visual.ImageStim(
    win=win, name='Cue',
    image='./images/VC_Legs.jpg', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=None,
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
focus = visual.ShapeStim(
    win=win, name='focus',color = 'black',
    size=(0.044, 0.044), vertices='circle', # change size to 0.045,0.045 if zoomed in
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='darkgrey', fillColor=(0.3255,0.3255,0.3255),
    opacity=None, depth=-2.0, interpolate=True)
Blank = visual.ImageStim(
    win=win, name='BlankScreen',
    image='./images/VC_Blank.jpg', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=None,
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)

# Create a stimulus for a certain window
readyText = visual.TextStim(win, "Sample Ready?", color=(1,1,1))
readyText.draw()
#present ready text on the screen 
win.flip()
#wait for user to press return key 
event.waitKeys(keyList=['return'])

# creating cue list
img_list = [('./images/VC_Relax.jpg',0),('./images/VC_Legs.jpg',3)]*2
np.random.shuffle(img_list)

ten_sec.draw()
win.flip()
core.wait(calTime)
    
for cue in img_list:

    Cue.image = cue[0]
    cue_cls = cue[1]

    restCross.draw()
    win.flip()
    core.wait(restTime)
    #win.flip()

    Cue.draw()
    win.flip()
    core.wait(cueTime)
    #win.flip()

    focus.draw()
    win.flip()
    core.wait(focusTime)

    Blank.draw()
    win.flip()
    core.wait(blkTime)
                
message = visual.TextStim(win, text="Trial Done")
# Draw the stimulus to the window. We always draw at the back buffer of the window.
message.draw()
# Flip back buffer and front  buffer of the window.
win.flip()
# Pause 5 s, so you get a chance to see it!
core.wait(5.0)
# Close the window
win.close()