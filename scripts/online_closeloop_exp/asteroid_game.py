import pygame
import time
import random
import os

pygame.init()
asteroid_img = 'images/asteroid.png'
asteroid_imgb = 'images/astb.png'
background_img = 'images/background.png'
rocket_img = 'images/rocket.png'
game_images = {}


# colors
blue = (0,0,255)
green = (0,255,0)
red = (255,0,0)
white = (255, 255, 255)
black = (0, 0, 0)
bright_red = (95,0,0)
bright_green = (0,95,0)
gray  = (128, 128, 128)#(0,0,255)

# display coordinates
display_width = 1440
display_height = 900


os.environ['SDL_VIDEO_WINDOW_POS'] = "1440,0"

# game display
pygame.display.set_caption('Space Explorer')
gameDisplay = pygame.display.set_mode((display_width, display_height),pygame.FULLSCREEN)
clock = pygame.time.Clock()

game_images['rocket'] = pygame.image.load(rocket_img).convert_alpha()
game_images['asteroid'] = pygame.image.load(asteroid_img).convert_alpha()
game_images['asteroidb'] = pygame.image.load(asteroid_imgb).convert_alpha()
game_images['background'] = pygame.image.load(background_img).convert_alpha()

game_images['rocket'] = pygame.transform.scale(game_images['rocket'],(150, 62)) # w/h = 2.43618
game_images['asteroid'] = pygame.transform.scale(game_images['asteroid'],(65, 70)) #w/h = 0.9253
game_images['asteroidb'] = pygame.transform.scale(game_images['asteroidb'],(65, 65)) #w/h = 1
game_images['background'] = pygame.transform.scale(game_images['background'],(1440,900))
# rocket
rocket_width = game_images['rocket'].get_width()

# rocket method for the position of the rocket
def rocket(object_x, object_y):
    gameDisplay.blit(game_images['rocket'], (object_x, object_y))
    #pygame.draw.rect(gameDisplay, color , [object_x, object_y, object_width, object_height])

# bot image method
def objects(object_x, object_y):
    gameDisplay.blit(game_images['asteroid'], (object_x, object_y))
    #pygame.draw.rect(gameDisplay, color , [object_x, object_y, object_width, object_height])

def objects2(object_x, object_y):
    gameDisplay.blit(game_images['asteroidb'], (object_x, object_y))
    #pygame.draw.rect(gameDisplay, color , [object_x, object_y, object_width, object_height])

def objects3(object_x, object_y):
    gameDisplay.blit(game_images['asteroid'], (object_x, object_y))
    #pygame.draw.rect(gameDisplay, color , [object_x, object_y, object_width, object_height])

#Score Function
def objects_dodged(count):
    font = pygame.font.SysFont(None,40)
    text = font.render("Score : "+str(count),True,white)
    gameDisplay.blit(text,(0, 0))

# Game Menu
def game_intro():
    intro = True
    while intro :
        for event in pygame.event.get():
            #print(event)
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                
        gameDisplay.fill(black)
        largeText = pygame.font.Font('freesansbold.ttf',30)
        TextSurf,TextRect = text_objects("Ready to start?",largeText)
        TextRect.center =((display_width/2),(display_height/2))
        gameDisplay.blit(TextSurf,TextRect)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            game_loop()

        pygame.display.update()
        clock.tick(20)

# game loop
def game_loop():
    # position of rocket
    x = (display_width * 0.25)
    y = (display_height * 0.4)

    y_up = 0
    y_down = 0
    
    # obects parameters
    object_speed = -85
    object_height1 = game_images['asteroid'].get_height()
    object_height2 = game_images['asteroidb'].get_height()
    object_height3 = game_images['asteroidb'].get_height()
    object_width = game_images['asteroid'].get_width()
    rocket_height = game_images['rocket'].get_height()
    rocket_width = game_images['rocket'].get_width()
    object_start_y = random.randrange(object_height1//2, display_height - (object_height1//2))
    object_start_x = display_width + 1440
    object2_start_y = random.randrange(object_height2//2, display_height - (object_height2//2))
    object2_start_x = display_width + 1440 + 1440 + 720
    object3_start_y = random.randrange(object_height3//2, display_height - (object_height3//2))
    object3_start_x = display_width + 1400 + 1440 + 1440 + 360

    #Objects Dodged
    dodged = 0

    # Game Logic
    game_exit = False

    while not game_exit:
        for event in pygame.event.get():                            # For closing the game
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:                        # when a key is pressed down
                if event.key == pygame.K_DOWN:                      # left arrow key
                    y_down  = 40
                    y_up = 0
                if event.key == pygame.K_UP:                     # right arrow key
                    y_up = -40
                    y_down = 0
                
        # change the position of the rocket
        y += y_down
        y += y_up

        if x+rocket_width > (object_start_x + object_width + object_speed): # object crash logic
            if (y+rocket_height > object_start_y and y < object_start_y + object_height1) or (y+rocket_height > object_start_y and y + rocket_height < object_start_y+object_height1):
                crashed(dodged)
                game_quit(dodged)
        elif x+rocket_width > (object2_start_x + object_width + object_speed): # object crash logic
            if  (y+rocket_height > object2_start_y and y < object2_start_y + object_height2) or (y+rocket_height > object2_start_y and y + rocket_height < object2_start_y+object_height2):
                crashed(dodged)
                game_quit(dodged)  
        elif x+rocket_width > (object3_start_x + object_width + object_speed): # object crash logic
            if  (y+rocket_height > object3_start_y and y < object3_start_y + object_height3) or (y+rocket_height > object3_start_y and y + rocket_height < object3_start_y+object_height3):
                crashed(dodged)
                game_quit(dodged)

        # black background
        gameDisplay.blit(game_images['background'], (0, 0))
        

        # objects location
        objects(object_start_x, object_start_y)
        objects2(object2_start_x, object2_start_y)
        objects3(object3_start_x, object3_start_y)
        object_start_x += object_speed
        object2_start_x += object_speed
        object3_start_x += object_speed

        rocket(x, y)
        objects_dodged(dodged)
        
        if y > (display_height - rocket_height):                # if the rocket goes outside the boundary
            y = 15
            rocket(x, y)
            #print('Lower bounds')
        
        elif y < (- rocket_height - 20):
            y = 875
            rocket(x, y)
            #print('Upper bounds')
            #crashed(dodged)
            #game_quit(dodged)

        if object_start_x + object_width < x-50:                               # object repeats itself
            object_start_x = display_width + object_width
            object_start_y = random.randrange(object_height1//2, display_height- (object_height1)//2)
            dodged += 1
            object_height1 += 2

        if object2_start_x + object_width < x-50:                               # object repeats itself
            object2_start_x = display_width + object_width
            object2_start_y = random.randrange(object_height2//2, display_height- (object_height2)//2)
            dodged += 1
            object_height2 += 2
        
        if object3_start_x + object_width < x-50:                               # object repeats itself
            object3_start_x = display_width + object_width
            object3_start_y = random.randrange(object_height3//2, display_height- (object_height3)//2)
            dodged += 1
            object_height3 += 2

        pygame.display.update()
        clock.tick(2)

# Quit Game
def game_quit(dodged) :
    print(f'Total score: {dodged}')
    #TODO put dodged amount in csv of txt file
    pygame.quit()
    quit()


# crashed method
def crashed(dodged):
   crashed_message(f'Crashed! Total score: {dodged}')

# text objects
def text_objects(message, font):
    text = font.render(message, True, gray)
    return text, text.get_rect()


# crashed message
def crashed_message(message):
    large_text = pygame.font.Font('freesansbold.ttf', 75)
    text_surface, text_rectangle = text_objects(message, large_text)
    text_rectangle.center = ((display_width/2), (display_height/2))
    gameDisplay.blit(text_surface, text_rectangle)
    pygame.display.update()
    time.sleep(2)


game_intro()
game_loop()