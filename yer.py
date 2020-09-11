import torch

import numpy as np
from numpy import interp as interp

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

# import lilly_11
import home_bred
# import puddle
# import ernie
import time
from time import perf_counter
from opensimplex import OpenSimplex


base = ShowBase()


class Yer(DirectObject):

    def __init__(self):
        self.loop_counter = 0
        self.remove_this = 'agent0'
        self.agent_name = 'agent0'
        
        self.agent_number = 0
        # list of food pieces
        self.food_piece_np={}
        # list of agents
        self.population = {}
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

        self.accept('f3', self.toggleDebug)

        # remove a agent
        #self.accept('k', self.remove_agent, [self.remove_this])
        # add an agent
        self.accept('n', self.agent_factory, [Lillies])

        # manual agent control ----------------------
        inputState.watchWithModifiers('forward', 'w')
        inputState.watchWithModifiers('left', 'a')
        inputState.watchWithModifiers('reverse', 's')
        inputState.watchWithModifiers('right', 'd')
        inputState.watchWithModifiers('turnLeft', 'q')
        inputState.watchWithModifiers('turnRight', 'e')
        inputState.watchWithModifiers('pulse', 'p')
        inputState.watchWithModifiers('jump', 'j')

        #initial food supply
        self.food_maker()

    
    
    
    # check every pieces of food in every frame and checks every agents agains it
    def eats(self):
        if self.food_piece_np:
            for key in self.food_piece_np:
                food =self.food_piece_np[key][0].node()

                # iterates over all overlapping nodes with that piece which is one most of the time
                for agent_node in food.getOverlappingNodes():
                    #prints node (which is agent) and food piece node
                    print(agent_node,self.food_piece_np[key][0].node())
                    print(self.food_piece_np[key][1])

                    #gets the name of the agent for updating life in population dictionary
                    agent_name=agent_node.getName()
                    # updates the life
                    self.population[agent_name][1]+=10
                    self.food_piece_np[key][1] +=1


    def food_maker(self):
        # dictionary {"food id" ,[food object, number of time eaten]}
        dx = .5
        dy = .5
        dz = .5
        #shape = BulletBoxShape(Vec3(dx, dy, dz))
        food = OpenSimplex(seed=1)
        visualNPList={}
        
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
                    self.food_piece_np[food_id] = [self.worldNP.attachNewNode(BulletGhostNode('Box'+str(i)+str(j))),0]
                    self.food_piece_np[food_id][0].setPos(i, j, 4)                    
                    self.food_piece_np[food_id][0].node().setFriction(1)
                    #self.food_piece_np[food_id].node().setMass(5)
                    self.food_piece_np[food_id][0].node().addShape(shape) 
                    self.food_piece_np[food_id][0].setCollideMask(BitMask32.bit(1))
                    self.world.attachGhost(self.food_piece_np[food_id][0].node())
                    z = self.food_piece_np[food_id][0].getZ()
                    pFrom = Point3(i,j,z)
                    pTo = Point3(i,j,z*-50)
                    # this is the hit object, from food piece(cube) to ground
                    result = self.world.rayTestClosest(pFrom, pTo)
                    # print(result.hasHit())
                    # print(result.getHitPos())
                    # print(result.getHitPos()[0])
                    # print(result.getHitNormal())
                    # print(result.getHitFraction())
                    print(result.getNode())
                    # (dy*food_scale/2) this is for keeping food over the surfaces a bit.
                    self.food_piece_np[food_id][0].setPos(i, j, result.getHitPos()[2]+dy*food_scale/2)
                    # self.food_piece_np[food_id].setPos(i, j, result.getHitPos()[2])
                    visualNPList[food_id] = loader.loadModel('models/cube.gltf')
                    visualNPList[food_id].set_scale(food_scale)
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

    def life_checker(self):
        now = perf_counter()
        # TODO: the lifechecker checks and make agent's heart beat in every frame in below loop
        # but it is not efficient, maybe only one of them is enough.
        # it is possible with generator or something

        # self.loop_counter=now
        for i,_ in self.population.items():
            # print('heartbeat')
            # heartbeat of the agent            #
            #print(type(i),i)
            self.population[i][0].heart()
            # current height
            my_z = self.population[i][0].my_z
            # 'elapsed'current age for the agent
            elapsed = now-self.population[i][1]

            if (elapsed > 10) or (my_z < -10):
                
                self.remove_agent(i,self.population)
                # very nice break, just breaks the loop after removing agent so no dictionary error
                break
        #checks how many times eaten an removes after one time

        for p,r in self.food_piece_np.items():
            if r[1]>0:                
                self.remove_agent(p,self.food_piece_np)
                break
            #print (type(p),p,r)


    def update(self, task):
        dt = globalClock.getDt()
        self.world.doPhysics(dt)
        self.life_checker()
        self.eats()

        return task.cont

    def remove_agent(self, remove_this,fromhere):

        print(f'removed agent: {self.worldNP.find(remove_this)}')

        if not self.worldNP.find(remove_this).isEmpty():
            # this is pyhon Lilly Class instance in the population dictionary
            del fromhere[remove_this]
            # this is PyBullet Node
            self.world.remove(self.worldNP.find(remove_this).node())
            # this is PyBullet NodePath
            self.worldNP.find(remove_this).detachNode()

    def agent_factory(self, fn):

        # add this instance to population dictionary in the yer
        agent = fn(self.agent_name)

        self.population[self.agent_name] = [agent, time.perf_counter()]        
        self.agent_number += 1
        self.agent_name = 'agent'+str(self.agent_number)
        print(self.population.items())
        # return fn(self.agent_name)

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

    def __init__(self, agent_name):

        self.hearttime = 0
        # bullet notePath 'z' value
        self.my_z = 0
        # self.lilly_11=lilly_11
        self.home_bred = home_bred
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
        win_prop = WindowProperties.size(240, 240)
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
        # base.graphicsEngine.renderFrame()

    def heart(self):

        now = perf_counter()

        if now-self.hearttime > .5:

            self.my_z = self.my_path.getZ()
            # print('heartbeat')
            try:
                my_output = self.my_buff.getActiveDisplayRegion(
                    0).getScreenshot()
                # for feeding neural net
                numpy_image_data = np.array(
                    my_output.getRamImageAs("RGB"), np.float32)
            except:
                base.graphicsEngine.renderFrame()
                print("except")
                my_output = self.my_buff.getActiveDisplayRegion(
                    0).getScreenshot()
                numpy_image_data = np.array(
                    my_output.getRamImageAs("RGB"), np.float32)
            # output neural net
            prediction = self.home_bred.home_bred_predict(numpy_image_data)

            x_Force = prediction[0][0]
            y_Force = prediction[0][1]
            z_Force = prediction[0][2]
            z_Torque = prediction[0][3]

            force = Vec3(x_Force, y_Force, z_Force)*100
            torque = Vec3(0, 0, z_Torque)*25

            force = yer.worldNP.getRelativeVector(self.my_path, force)
            torque = yer.worldNP.getRelativeVector(self.my_path, torque)
            self.body_node.setActive(True)
            self.body_node.applyCentralForce(force)
            self.body_node.applyTorque(torque)
            self.hearttime = perf_counter()


class LilliesManual(Yer):

    def __init__(self, agent_name):

        self.hearttime = 0
        # bullet notePath 'z' value
        self.my_z = 0
        # self.lilly_11=lilly_11
        self.home_bred = home_bred
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
        win_prop = WindowProperties.size(240, 240)
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

        force *= 90.0
        torque *= 30.0
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

agent0 = yer.agent_factory(LilliesManual)
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

base.run()
