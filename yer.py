import torch
import numpy as np
from numpy import interp as interp
import itertools  
import sys
from direct.showbase.ShowBase import ShowBase

from direct.showbase.DirectObject import DirectObject
from direct.showbase.InputStateGlobal import inputState

from panda3d.core import AmbientLight
from panda3d.core import DirectionalLight
from panda3d.core import Vec3
from panda3d.core import Vec4
from panda3d.core import Point3
from panda3d.core import TransformState
from panda3d.core import BitMask32
from panda3d.core import Filename
from panda3d.core import PNMImage
from panda3d.core import GeoMipTerrain
from panda3d.core import getModelPath
from panda3d.core import WindowProperties
from panda3d.core import GraphicsOutput
from panda3d.core import Texture
from panda3d.core import Camera
from panda3d.core import PerspectiveLens
from panda3d.core import Material
from direct.gui.DirectGui import *
import gltf
# from panda3d.core import PandaNode

from panda3d.core import FrameBufferProperties
from panda3d.core import GraphicsBuffer
from panda3d.core import GraphicsPipe
from panda3d.bullet import BulletGhostNode
from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletCapsuleShape
from panda3d.bullet import BulletSphereShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletDebugNode
from panda3d.bullet import BulletHeightfieldShape
from panda3d.bullet import ZUp
from panda3d.bullet import BulletCharacterControllerNode
from panda3d.core import GraphicsEngine



import small_net
import first_seed
import time
from time import perf_counter
from opensimplex import OpenSimplex


base = ShowBase()



class Yer(DirectObject):

    def __init__(self):
        self.loop_counter = 0
        #initial values for creation and removal for first agent (manual controlled agent)
        self.remove_this = 'agent1'
        self.agent_name = 'agent1'        
        self.agent_number = 1
        # list of food pieces, dictionary {"food id" ,[food object, number of time eaten]}
        self.food_piece_np={}
        # list of agents, dictionary {"agent_name" ,[agent object, agent_age, agent energy]}
        self.population = []
        self.num_of_individuals=1
        

        
        self.best_brains=[]
        #gltf loader instead of native loader (must be installed first)
        gltf.patch_loader(base.loader)
        # create a rendering window
        wp = WindowProperties()
        wp.setSize(1000, 1000)
        # somehow this is IMPORTANT (requestProperties)
        base.win.requestProperties(wp)
        base.setBackgroundColor(0.1, 0.1, 0.8, 1)
        base.setFrameRateMeter(True)
        base.cam.setPos(0, -80, 50)
        base.cam.lookAt(0, 0, 0)
        base.disableMouse()
        base.useTrackball()

        # Light      ####################################################################################

        alight = AmbientLight('ambientLight')
        alight.setColor(Vec4(0.5, 0.5, 0.5, 1))
        alightNP = render.attachNewNode(alight)
        dlight = DirectionalLight('directionalLight')
        dlight.setDirection(Vec3(1, 1, -1))
        dlight.setColor(Vec4(0.8, 0.8, 0.8, 1))
        dlightNP = render.attachNewNode(dlight)
        render.clearLight()
        render.setLight(alightNP)
        render.setLight(dlightNP)
        render.setShaderAuto()

        self.landscape()

        taskMgr.add(self.update, 'updateWorld')
        
        
        
        taskMgr.doMethodLater(2, self.checker, 'checker')

        self.accept('f3', self.toggleDebug)

        # remove a agent
        #self.accept('k', self.remove_agent, [self.remove_this])
        # add an agent

        # manual agent control ----------------------
        inputState.watchWithModifiers('forward', 'w')
        inputState.watchWithModifiers('left', 'a')
        inputState.watchWithModifiers('reverse', 's')
        inputState.watchWithModifiers('right', 'd')
        inputState.watchWithModifiers('turnLeft', 'q')
        inputState.watchWithModifiers('turnRight', 'e')
        inputState.watchWithModifiers('pulse', 'p')
        inputState.watchWithModifiers('jump', 'j')

        # create a new duck
        self.accept('z', self.agent_zero, [LilliesManual])
        self.accept('n', self.agent_factory, [Lillies])

        #initial food supply
        self.food_maker()

    
    
    
    # check every pieces of food in every frame and checks every agents
    # TODO this could be a yielding fuction, each time we can run it only one of  the piece
    # I guess it is enough.

    def eats(self):
        if self.food_piece_np:
            for key in self.food_piece_np:
                food =self.food_piece_np[key][0].node()
                # iterates over all overlapping nodes with that piece which is one most of the time
                for agent_node in food.getOverlappingNodes():
                    #prints node (which is agent) and food piece node
                    # print(agent_node,self.food_piece_np[key][0].node())
                    # print(self.food_piece_np[key][1])
                    #gets the name of the agent for updating life in population dictionary
                    # print(agent_node)
                    agent_name=agent_node.getName()
                    # updates the life
                    for agent in self.population:

                        # print(agent[2])
                        if self.individual_name(agent)==agent_name:
                            # print("life+")
                            agent[3]+=10

                    # self.population[agent_name][1]+=10
                    # updates the number of bites taken from a particular food piece
                    self.food_piece_np[key][1] +=1


    def individual_name(self,individual):
        """ brings individual's name from population item"""
        return individual[0]


    def individual_obj(self,individual):
        """ brings individual's object from population item"""
        return individual[1]
    

    def individual_b_day(self,individual):
        """ brings individual's born date from population item"""
        return individual[2]


    def individual_energy(self,individual):
        """ brings individual'energy level population item"""
        return individual[3]


    def individual_burn_energy(self,individual):
        """ decreases individual's energy level"""
        individual[3] -= 2

    
    def individual_age(self,individual):
        return time.perf_counter() - self.individual_b_day(individual)


    def individual_z(self,individual):
        return self.individual_obj(individual).my_z



    


    def food_maker(self):
        
        dx = .5
        dy = .5
        dz = .5
        food = OpenSimplex(seed=1)
        visualNPList={}

        myMaterial = Material()
        myMaterial.setShininess(5.0) #Make this material shiny
        myMaterial.setAmbient((1, 0, 1, 1)) #Make this material blue

        
        
        # based on approxiamate width of the landscape (hard-coded)

        for i in range(-60, 61, 6):
            for j in range(-60, 61, 6):
                # 2d noise for scaling
                food_scale = food.noise2d(i/50, j/50)
                # interpolated between (0-1)
                food_scale = interp(food_scale, (-1, 1), (0, 1))
                chance = food_scale * np.random.randint(0, 100)
                # made it string to keep the consisteny with population dictionary (name,[object,data])
                food_id="Box"+str(i)+str(j)
                if chance > 50:
                    shape = BulletBoxShape(Vec3(dx*food_scale , dy*food_scale , dz*food_scale))
                    """this row creates a GhostNode and adds '0' as number of bite eaten by agent to a list,
                     then put this list in a food_piece_np dictionary."""
                    self.food_piece_np[food_id] = [self.worldNP.attachNewNode(BulletGhostNode(food_id)),0]
                    # initial position for ghost node
                    self.food_piece_np[food_id][0].setPos(i, j, 4)                    
                    self.food_piece_np[food_id][0].node().setFriction(1)
                    self.food_piece_np[food_id][0].node().addShape(shape) 
                    # set the mask to one to prevent contact between terrain and food
                    self.food_piece_np[food_id][0].setCollideMask(BitMask32.bit(1))
                    #add node to the world
                    self.world.attachGhost(self.food_piece_np[food_id][0].node())
                    # get current position of the food (it is in sky)
                    z = self.food_piece_np[food_id][0].getZ()
                    # find terrain surface
                    pFrom = Point3(i,j,z)
                    pTo = Point3(i,j,z*-50)
                    # this is the hit object, from food piece(cube) to ground
                    result = self.world.rayTestClosest(pFrom, pTo)
                    # print(result.hasHit())
                    # print(result.getHitPos())
                    # print(result.getHitPos()[0])
                    # print(result.getHitNormal())
                    # print(result.getHitFraction())
                    # print(result.getNode())                    
                    # (dy*food_scale/2) this is for keeping food over the surfaces a bit.
                    self.food_piece_np[food_id][0].setPos(i, j, result.getHitPos()[2]+dy*food_scale/2)
                    # this is the visual for the cube                     
                    visualNPList[food_id] = loader.loadModel('models/cube.gltf')
                    visualNPList[food_id].set_scale(food_scale)

                    visualNPList[food_id].setMaterial(myMaterial)

                    visualNPList[food_id].reparentTo(self.food_piece_np[food_id][0])
                

    def toggleDebug(self):
        if self.debugNP.isHidden():
            self.debugNP.show()
        else:
            self.debugNP.hide()


    def landscape(self):
        ## Bullet World #################################################################################
        self.worldNP = render.attachNewNode('World')

        self.debugNP = self.worldNP.attachNewNode(BulletDebugNode('Debug'))
        self.debugNP.show()
        self.debugNP.node().showNormals(True)

        self.world = BulletWorld()
        self.world.setDebugNode(self.debugNP.node())
        self.world.setGravity(Vec3(0, 0, -9.81))

        # Heightfield Surface ##########################################################################
        height = 6.0
        img = PNMImage()
        # couldn't read the files at fist and asked help from the forum. That's why it looks weird.
        assert img.read(getModelPath().findFile(
            'models/elevation2.png')), "Failed to read!"
        shape = BulletHeightfieldShape(img, height, ZUp)
        shape.setUseDiamondSubdivision(True)
        np = self.worldNP.attachNewNode(BulletRigidBodyNode('Heightfield'))
        np.node().addShape(shape)
        np.setPos(0, 0, 0)
        np.set_scale(2)
        np.node().setFriction(.5)
        np.setCollideMask(BitMask32.bit(2))
        self.world.attach(np.node())
        self.hf = np.node()  # To enable/disable debug visualisation
        self.terrain = GeoMipTerrain('terrain')
        self.terrain.setHeightfield(img)
        self.terrain.setBlockSize(32)
        # I don't want any optimization that's why I commented that
        # self.terrain.setNear(50)
        # self.terrain.setFar(100)
        # self.terrain.setFocalPoint(base.camera)
        rootNP = self.terrain.getRoot()
        rootNP.reparentTo(render)
        rootNP.setSz(height * 2)
        rootNP.setSx(2)
        rootNP.setSy(2)
        offset = img.getXSize() / 2.0 - 0.5
        rootNP.setPos(-offset * 2, -offset * 2, -height)
        self.terrain.generate()


    def brains(self):
        ''' returns the name of the oldest individual'''
        # it should be revised to return best brain model not name
        now = perf_counter()
        best_brains= sorted(self.population.items(), key = (lambda item: now - (item[1][1])),reverse=True)
        # name of the oldest individual
        return best_brains[0][0]


    def food_checker(self):
        for p,r in self.food_piece_np.items():
            # remove piece after 3rd bite
            if r[1]>3:                
                self.remove_agent(p,self.food_piece_np)
                break
            #print (type(p),p,r) 
                   
    def check_health(self,individual):
        
        if self.individual_age(individual)>1550 or self.individual_z(individual)<-10 or self.individual_energy(individual)<0 :
            self.remove_individual(individual,self.population)


    def pick_from_population(self,frame):
        # circular call for each item
        idx= frame % len(self.population)       
        individual=self.population[idx]
        return individual


    def energy_burner(self):
        for agent in self.population:        
            self.individual_burn_energy(agent)  
             

    def checker(self, task):
        """ updates every 2 seconds """
        # print(f'{task.frame} frame')
        # print("checker")
        self.energy_burner()
        self.food_checker()
        self.eats()  
        return task.again


    def update(self, task):
        """ updates every frame """
        dt = globalClock.getDt()
        self.world.doPhysics(dt)
        # next agent
        individual=self.pick_from_population(task.frame)
        # heartbeat of next agent
        self.individual_obj(individual).heart()
        self.check_health(individual)
        self.num_of_individuals=len(self.population)
        # Manual agent
        agent0.heart()
        return task.cont


    def remove_individual(self, individual,fromhere):
        """ this fuction removes agents and foods from environment
        also removes relevant nodes and nodepatths(not sure about notePaths))"""

        name=self.individual_name(individual)

        np=self.worldNP.find(name)
        #print(np,type(np))
        print(f'removed agent: {name}')

        if not self.worldNP.find(name).isEmpty():
            # this is pyhon Lilly Class instance in the population dictionary
            self.population.remove(individual)
            
            # this is removes PyBullet Node
            self.world.remove(np.node())
            # this removes nodePath.....check print statement
            np.removeNode()
            #print(np)

    def remove_agent(self, remove_this,fromhere):
        """ this fuction removes agents and foods from environment
        also removes relevant nodes and nodepatths(not sure about notePaths))"""

        np=self.worldNP.find(remove_this)
        #print(np,type(np))
        print(f'removed agent: {self.worldNP.find(remove_this)}')

        if not self.worldNP.find(remove_this).isEmpty():
            # this is pyhon Lilly Class instance in the population dictionary
            del fromhere[remove_this]
            # this is removes PyBullet Node
            self.world.remove(np.node())
            # this removes nodePath.....check print statement
            np.removeNode()
            #print(np)


    def agent_zero(self, fn):

        # add this instance to population dictionary in the yer
        agent = fn(self.agent_name)
        return agent


    # def agent_zero(self, fn):

    #     # add this instance to population dictionary in the yer
    #     agent = fn(self.agent_name)
    #     full_stomach=50
    #     self.population.append([self.agent_name,agent, time.perf_counter(), full_stomach])        
    #     self.agent_number += 1
    #     self.agent_name = 'agent'+str(self.agent_number)
    #     print(self.population)        

    def agent_factory(self, fn):
        brains=first_seed.model_loader()
        for brain in brains:
            agent = fn(self.agent_name,brain)
            full_stomach=50
            self.population.append([self.agent_name,agent, time.perf_counter(), full_stomach])        
            self.agent_number += 1
            self.agent_name = 'agent'+str(self.agent_number)
        print(self.population)
        

    def make_body(self, agent_name):
        """creates agents for the world"""
        shape = BulletCapsuleShape(.35, 1, ZUp)
        shape2 = BulletCapsuleShape(.35, 1, ZUp)
        head = BulletSphereShape(.3)
        # nodepath---------------------------
        body_node_path = self.worldNP.attachNewNode(
            BulletRigidBodyNode(agent_name))
        # print(body_node_path)
        body_node_path.setPos(0, 0, 5)
        # body_node_path.set_scale(3)
        body_node_path.setCollideMask(BitMask32.allOn())
        # node-------------------------------
        body_node = body_node_path.node()
        body_node.setMass(5)
        body_node.addShape(shape, TransformState.makePosHpr(
            Point3(-.35, 0, 0), Point3(90, 0, 90)))
        body_node.addShape(shape2, TransformState.makePosHpr(
            Point3(.35, 0, 0), Point3(90, 0, 90)))
        body_node.addShape(head, TransformState.makePos(Point3(0, .5, 1)))
        body_node.setFriction(0.5)
        # visual representation--------------
        visualNP = loader.loadModel('models/lilly.gltf')
        visualNP.set_scale(.5)
        visualNP.setPos(0, 0, 0)
        visualNP.setHpr(180, 270, 0)
        materials = visualNP.findAllMaterials()
        materials[0].clearBaseColor()

        # BU NE?
        visualNP.clearModelNodes()
        # BU NE?
        visualNP.reparentTo(body_node_path)
        return agent_name, body_node_path, body_node



class Lillies(Yer):

    def __init__(self, agent_name,brain):
        # not used can be deleted
        self.name=agent_name
        self.hearttime = 0
        # bullet notePath 'z' value
        self.my_z = 0
        # self.lilly_11=lilly_11
        # self.brain = small_net
        self.brain = brain
        #print(self.brain.model)
        # print(self.lilly_11[0].values)
        self.x_Force = 0
        self.y_Force = 0
        self.z_Force = 0
        self.z_Torque = 0

        fb_prop = FrameBufferProperties()
        # Request 8 RGB bits, no alpha bits, and a depth buffer.
        fb_prop.setRgbColor(True)
        # fb_prop.setSrgbColor(True)**** Dosn't work with this in this file???? ************
        fb_prop.setRgbaBits(8, 8, 8, 0)
        fb_prop.setDepthBits(16)
        # Create a WindowProperties object set to 256x256 size.
        win_prop = WindowProperties.size(224, 224)
        flags = GraphicsPipe.BF_refuse_window
        # flags = GraphicsPipe.BF_require_window

        lens = PerspectiveLens()
        self.my_buff = base.graphicsEngine.make_output(
            base.pipe, yer.agent_name+"_buffer", -100, fb_prop, win_prop, flags, base.win.getGsg(), base.win)
        my_cam = base.makeCamera(self.my_buff, sort=6, displayRegion=(
            0.0, 1, 0, 1), camName=yer.agent_name+"_cam")
        my_cam.setHpr(0, 0, 0)
        my_cam.setPos(0, 0, 1)
        my_cam.node().setLens(lens)
        lens.setFov(100)

        # make body of the agent

        agent_name, body_node_path, body_node = yer.make_body(agent_name)
        self.my_path = body_node_path
        self.body_node = body_node
        my_cam.reparentTo(body_node_path)
        body_node_path.setPos(np.random.randint(-60, 60),
                              np.random.randint(-60, 60), np.random.randint(2, 5))
        yer.world.attach(body_node)

        # removing except statement from  heart() and adding "renderFrame" may lead better performance TRY IT
        base.graphicsEngine.renderFrame()

    def heart(self):

        # now = perf_counter()
        self.my_z = self.my_path.getZ()
        my_output = self.my_buff.getActiveDisplayRegion(0).getScreenshot()
            # for feeding neural net
        numpy_image_data = np.array(my_output.getRamImageAs("RGB"), np.float32)
        
        # print('heartbeat')
        # I removed this try/except part because it slows down all process (pyhon try/except is very slow)
        # instead I add renderFrame to _init_ to solve the error (if you try to get screenShot before creating cam)
        
        # try:
        #     my_output = self.my_buff.getActiveDisplayRegion(
        #         0).getScreenshot()
        #     # for feeding neural net
        #     numpy_image_data = np.array(
        #         my_output.getRamImageAs("RGB"), np.float32)
        # except:
        #     base.graphicsEngine.renderFrame()
        #     print("except")
        #     my_output = self.my_buff.getActiveDisplayRegion(
        #         0).getScreenshot()
        #     numpy_image_data = np.array(
        #         my_output.getRamImageAs("RGB"), np.float32)
        # # output neural net
        prediction = first_seed.predict(numpy_image_data,self.brain)

        x_Force = prediction[0][0]
        y_Force = prediction[0][1]
        z_Force = prediction[0][2]
        z_Torque = prediction[0][3]

        force = Vec3(x_Force, y_Force, z_Force)*5*yer.num_of_individuals
        torque = Vec3(0, 0, z_Torque)*1*yer.num_of_individuals

        force = yer.worldNP.getRelativeVector(self.my_path, force)
        torque = yer.worldNP.getRelativeVector(self.my_path, torque)
        self.body_node.setActive(True)
        self.body_node.applyCentralForce(force)
        self.body_node.applyTorque(torque)
        # can be deleted(test it)
        # self.hearttime = perf_counter()


class LilliesManual(Yer):

    def __init__(self, agent_name):

        self.hearttime = 0
        # bullet notePath 'z' value
        self.my_z = 0
        # self.lilly_11=lilly_11
        # self.home_bred = home_bred
        # print(self.lilly_11[0].values)
        # self.x_Force=0
        # self.y_Force=0
        # self.z_Force=0
        # self.z_Torque=0

        fb_prop = FrameBufferProperties()
        # Request 8 RGB bits, no alpha bits, and a depth buffer.
        fb_prop.setRgbColor(True)
        # fb_prop.setSrgbColor(True)**** Dosn't work with this in this file???? ************
        fb_prop.setRgbaBits(8, 8, 8, 0)
        fb_prop.setDepthBits(16)
        # Create a WindowProperties object set to 256x256 size.
        win_prop = WindowProperties.size(128, 128)
        # flags = GraphicsPipe.BF_refuse_window
        flags = GraphicsPipe.BF_require_window

        lens = PerspectiveLens()
        self.my_buff = base.graphicsEngine.make_output(
            base.pipe, yer.agent_name+"_buffer", -100, fb_prop, win_prop, flags, base.win.getGsg(), base.win)
        my_cam = base.makeCamera(self.my_buff, sort=6, displayRegion=(
            0.0, 1, 0, 1), camName=yer.agent_name+"_cam")
        my_cam.setHpr(0, 0, 0)
        my_cam.setPos(0, 0, 1)
        my_cam.node().setLens(lens)
        lens.setFov(100)

        # make body of the agent

        agent_name, body_node_path, body_node = yer.make_body(agent_name)
        self.my_path = body_node_path
        self.body_node = body_node
        my_cam.reparentTo(body_node_path)
        body_node_path.setPos(np.random.randint(-60, 60),
                              np.random.randint(-60, 60), np.random.randint(2, 5))
        yer.world.attach(body_node)

        # removing except statement from  heart() and adding "renderFrame" may lead better performance TRY IT
        # base.graphicsEngine.renderFrame()

    def heart(self):

        self.my_z = self.my_path.getZ()
        # print('heartbeat')
        try:
            my_output = self.my_buff.getActiveDisplayRegion(0).getScreenshot()
            # for feeding neural net

        except:
            base.graphicsEngine.renderFrame()
            print("except")
            my_output = self.my_buff.getActiveDisplayRegion(0).getScreenshot()

        force = Vec3(0, 0, 0)
        torque = Vec3(0, 0, 0)
        # Manual Lillie------------------

        if inputState.isSet('forward'):
            force.setY(1.0)
        if inputState.isSet('reverse'):
            force.setY(-1.0)
        if inputState.isSet('left'):
            force.setX(-1.0)
        if inputState.isSet('right'):
            force.setX(1.0)

        if inputState.isSet('turnLeft'):
            torque.setZ(1.0)
        if inputState.isSet('turnRight'):
            torque.setZ(-1.0)
        # Manual Lillie------------------

        force *= 10*yer.num_of_individuals
        torque *= 2*yer.num_of_individuals
        # force=Vec3(x_Force,y_Force,0)*150
        # torque=Vec3(0,0,z_Torque)*40

        force = yer.worldNP.getRelativeVector(self.my_path, force)
        torque = yer.worldNP.getRelativeVector(self.my_path, torque)
        self.body_node.setActive(True)
        self.body_node.applyCentralForce(force)
        self.body_node.applyTorque(torque)
        self.hearttime = perf_counter()


yer = Yer()
# this is manual controlled agent for debugging

agent0 = yer.agent_zero(LilliesManual)

# first load after startup
brains=first_seed.model_loader()
for brain in brains:
    
    agent=Lillies(yer.agent_name,brain)
    full_stomach=50
    yer.population.append([yer.agent_name,agent, time.perf_counter(), full_stomach])        
    yer.agent_number += 1
    yer.agent_name = 'agent'+str(yer.agent_number)
print(yer.population)




#f = yer.food_maker()


"""
print("WorldNP ALL Children")
print(yer.worldNP.getChildren())

print(render.getChildren())

print("Camera List--------")
print(base.camList)

print("WORLD NP")
print(yer.worldNP.findAllMatches("*"))

print("A.J.A.N")
print(yer.worldNP.find("*").node())"""


bk_text = "This is my Demo"
textObject = OnscreenText(text=bk_text, pos=(0.95,-0.95), scale=0.07,
                    fg=(1, 0.5, 0.5, 1), 
                    mayChange=1)
def setText():
        bk_text = "Button Clicked"
        textObject.setText(bk_text)

# Add button
b = DirectButton(text=("OK", "click!", "rolling over", "disabled"),
                 scale=.1, command=setText)
base.run()
