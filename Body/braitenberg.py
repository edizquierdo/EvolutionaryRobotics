import numpy as np

# def trig(xpos, radius, orientation, angleoffset):
#     return xpos + radius * np.cos(orientation + angleoffset)

class Vehicle:

    def __init__(self):
        self.xpos = 0.0                                       # agent's x position, starts in middle of world
        self.ypos = 0.0                                       # agent's y position, starts in middle of world
        self.orientation = np.random.random()*2*np.pi         # agent's orientation, starts at random
        self.velocity = 0.0                                   # agent's velocity, starts at 0
        self.radius = 1.0                                     # the size/radius of the vehicle
        self.leftsensor = 0.0                                 # left sensor value
        self.rightsensor = 0.0                                # right sensor value

        # Attributes to determine the placement of the sensors
        self.angleoffset = np.pi/2                                                 # left/right sensor angle offset
        self.rs_xpos = self.radius * np.cos(self.orientation + self.angleoffset)   # right sensor x position
        self.rs_ypos = self.radius * np.sin(self.orientation + self.angleoffset)   # right sensor y position
        self.ls_xpos = self.radius * np.cos(self.orientation - self.angleoffset)   # left sensor x position
        self.ls_ypos = self.radius * np.sin(self.orientation - self.angleoffset)   # left sensor y position

    def sense(self,light):
        # Calculate the distance of the light for each of the sensors
        self.leftsensor = 1 - np.sqrt((self.ls_xpos-light.x)**2 + (self.ls_ypos-light.y)**2)/10
        self.leftsensor = np.clip(self.leftsensor,0,1)
        self.rightsensor = 1 - np.sqrt((self.rs_xpos-light.x)**2 + (self.rs_ypos-light.y)**2)/10
        self.rightsensor = np.clip(self.rightsensor,0,1)

    def think(self):
        self.rightmotor = self.leftsensor
        self.leftmotor = self.rightsensor

    def move(self):
        # Update the orientation and velocity of the vehicle based on the left and right motors
        self.rightmotor = np.clip(self.rightmotor,0,1)
        self.leftmotor  = np.clip(self.leftmotor,0,1)
        self.orientation += ((self.leftmotor - self.rightmotor)/10) + np.random.normal(0,0.1)
        self.velocity = 0.01 #((self.rightmotor + self.leftmotor)/2)/10

        # Update position of the agent
        self.xpos += self.velocity * np.cos(self.orientation)
        self.ypos += self.velocity * np.sin(self.orientation)

        # Update position of the sensors
        self.rs_xpos = self.xpos + self.radius * np.cos(self.orientation + self.angleoffset)
        self.rs_ypos = self.ypos + self.radius * np.sin(self.orientation + self.angleoffset)
        self.ls_xpos = self.xpos + self.radius * np.cos(self.orientation - self.angleoffset)
        self.ls_ypos = self.ypos + self.radius * np.sin(self.orientation - self.angleoffset)

    def distance(self,light):
        return np.sqrt((self.x-light.x)**2 + (self.y-light.y)**2)

class Light:

    def __init__(self):
        angle = np.random.random()*2*np.pi
        self.x = 10.0 * np.cos(angle)
        self.y = 10.0 * np.sin(angle)


