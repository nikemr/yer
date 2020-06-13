import torch

import numpy as np

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
import lilly_11
# import puddle
# import ernie
import time

base = ShowBase()


class Yer(DirectObject):

    def __init__(self):   

        self.agent_name='agent0'
        self.agent_number=0
        gltf.patch_loader(base.loader)
        # create a rendering window
        wp = WindowProperties()
        wp.setSize(1000, 1000)
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
        self.accept('k',self.agent_trig,[self.remove_agent])
        # add an agent
        self.accept('n', self.agent_trig,[self.make_agent])

        # inputState.watchWithModifiers('forward', 'w')
        # inputState.watchWithModifiers('left', 'a')
        # inputState.watchWithModifiers('reverse', 's')
        # inputState.watchWithModifiers('right', 'd')
        # inputState.watchWithModifiers('turnLeft', 'q')
        # inputState.watchWithModifiers('turnRight', 'e')
        # inputState.watchWithModifiers('pulse', 'p')
        # inputState.watchWithModifiers('jump', 'j')
        # inputState.watchWithModifiers('removeNode', 'k')



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
        assert img.read(getModelPath().findFile('models/elevation2.png')), "Failed to read!"
        shape = BulletHeightfieldShape(img, height, ZUp)
        shape.setUseDiamondSubdivision(True)
        np = self.worldNP.attachNewNode(BulletRigidBodyNode('Heightfield'))
        np.node().addShape(shape)
        np.setPos(0, 0, 0)
        np.set_scale(2)
        np.node().setFriction(.5)
        np.setCollideMask(BitMask32.allOn())
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

    def update(self, task):
        dt = globalClock.getDt()
        self.world.doPhysics(dt)
        return task.cont

    def remove_agent(self,remove_this):

        print(self.worldNP.find(remove_this))
        
        if not self.worldNP.find(remove_this).isEmpty():
            # print(f'Node type:{type(self.worldNP.find(remove_this).node())}')
            # print(f'Pybullet Node:{self.worldNP.find(remove_this).node()}')
            # print(f'Pybullet World{self.world}')
            self.world.remove(self.worldNP.find(remove_this).node())
            self.worldNP.find(remove_this).detachNode()




   
    def agent_trig(self,fn):
        return fn(self.agent_name)


    def make_agent(self,agent_name):
        """creates agents for the world"""
        shape = BulletCapsuleShape(.35, 1, ZUp)
        shape2 = BulletCapsuleShape(.35, 1, ZUp)
        head = BulletSphereShape(.3)
        # nodepath---------------------------
        agent_node_path = self.worldNP.attachNewNode(BulletRigidBodyNode(agent_name))
        print(agent_node_path)
        agent_node_path.setPos(0, 0, 10)
        # agent_node_path.set_scale(3)
        agent_node_path.setCollideMask(BitMask32.allOn())
        # node-------------------------------
        agent_node = agent_node_path.node()
        agent_node.setMass(5)
        agent_node.addShape(shape, TransformState.makePosHpr(Point3(-.35, 0, 0), Point3(90, 0, 90)))
        agent_node.addShape(shape2, TransformState.makePosHpr(Point3(.35, 0, 0), Point3(90, 0, 90)))
        agent_node.addShape(head, TransformState.makePos(Point3(0, .5, 1)))
        agent_node.setFriction(0.5)
        # visual representation--------------
        visualNP = loader.loadModel('models/lilly.gltf')
        materials = visualNP.findAllMaterials()
        materials[0].clearBaseColor()
        visualNP.reparentTo(agent_node_path)

        self.agent_number += 1
        self.agent_name='agent'+str(self.agent_number)

        return self.add_to_world(agent_node)

    # def total_agents(self):

    def food_maker(self):
        """creates food for the world"""
        pass

    def add_to_world(self, add_this):
        """adds object to the world"""

        return self.world.attach(add_this)


yer = Yer()

ajan = yer.make_agent("agent0")

print("WorldNP ALL Children")
print(yer.worldNP.getChildren())

print(render.getChildren())

print("Camera List--------")
print(base.camList)

print("WORLD NP")
print(yer.worldNP.findAllMatches("*"))

print("A.J.A.N")
print(yer.worldNP.find("*").node())

base.run()
