
#from pandac.PandaModules import loadPrcFileData
#loadPrcFileData('', 'load-display tinydisplay')
import numpy as np
import sys
from direct.showbase.ShowBase import ShowBase

#import direct.directbase.DirectStart

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

from panda3d.core import FrameBufferProperties
from panda3d.core import GraphicsBuffer
from panda3d.core import GraphicsPipe


from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletDebugNode
from panda3d.bullet import BulletHeightfieldShape
from panda3d.bullet import ZUp

base=ShowBase()
class Yer(DirectObject):
    def __init__(self):
        # create a rendering window
        wp=WindowProperties()
        wp.setSize(1000,1000)
        base.win.requestProperties(wp)        
        self.setup()
        taskMgr.add(self.update, 'updateWorld')
    

    def setup(self):
        # create a world nodepath
        self.worldNP = render.attachNewNode('World')

        # World 
        self.debugNP = self.worldNP.attachNewNode(BulletDebugNode('Debug'))
        self.debugNP.show()
        self.debugNP.node().showNormals(True)

        self.world = BulletWorld()        
        self.world.setGravity(Vec3(0, 0, -9.81))        
        self.world.setDebugNode(self.debugNP.node())

        
        # Heightfield (static)
        height = 12.0
        img = PNMImage()
        # couldn't read the files at fist and asked help from the forum. That's why it looks weird.
        assert img.read(getModelPath().findFile('models/elevation2.png')), "Failed to read!"
        shape = BulletHeightfieldShape(img, height, ZUp)
        shape.setUseDiamondSubdivision(True)
        np = self.worldNP.attachNewNode(BulletRigidBodyNode('Heightfield'))
        np.node().addShape(shape)
        np.setPos(0, 0, 0)

        #I put this
        np.node().setFriction(1)

        np.setCollideMask(BitMask32.allOn())

        self.world.attachRigidBody(np.node())

        self.hf = np.node() # To enable/disable debug visualisation

        self.terrain = GeoMipTerrain('terrain')
        self.terrain.setHeightfield(img)
    
        self.terrain.setBlockSize(32)
        #I don't want any optimization that's why I commented that
        #self.terrain.setNear(50)
        #self.terrain.setFar(100)
        #self.terrain.setFocalPoint(base.camera)
    
        rootNP = self.terrain.getRoot()
        rootNP.reparentTo(render)
        rootNP.setSz(height)

        offset = img.getXSize() / 2.0 - 0.5
        rootNP.setPos(-offset, -offset, -height / 2.0)
    
        self.terrain.generate()
        # Box (dynamic) 

        for r in range(12):

            shape = BulletBoxShape(Vec3(0.5, 0.5, 0.5))
            np = self.worldNP.attachNewNode(BulletRigidBodyNode('Box'))
            np.node().setMass(5)
            np.node().addShape(shape)
            np.setPos(r/2, r,3*r)
            np.set_scale(1)
            np.setCollideMask(BitMask32.allOn())
            self.world.attachRigidBody(np.node())
            #Friction for the Box
            np.node().setFriction(0.5)
            self.boxNP = np # For applying force & torque
            visualNP = loader.loadModel('models/mox.egg')
            visualNP.clearModelNodes()
            visualNP.reparentTo(self.boxNP)   
    def update(self, task):
        dt = globalClock.getDt()
        #self.processInput(dt)
        self.world.doPhysics(dt)
        return task.cont
    
yer = Yer()
base.run()           