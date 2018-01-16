import pybullet as pb
import numpy as np
import time,os,math,inspect,re,errno
import random,glob,math
from shutil import copyfile
import sensenet
from sensenet.error import Error

class SurviveEnv(sensenet.SenseEnv):

    def __init__(self,options={}):
        self.options = options
        self.steps = 0
        self.physics = pb

        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        parentdir = os.path.dirname(os.path.dirname(currentdir))
        os.sys.path.insert(0,parentdir)
        #TODO check if options is a string, so we know which environment to load
        if 'render' in self.options and self.options['render'] == True:
            pb.connect(pb.GUI)
        else:
            pb.connect(pb.DIRECT)
        #pb.setGravity(10,10,10)
        pb.setGravity(0,0,-10)
        pb.setRealTimeSimulation(0)
        self.move = 0.01 # 0.01
        self.touched_steps = 0 #how many steps have we touched agent
        self.touch_max_steps_until_death = 10

    def load_enemies(self):

        #self.enemy = pb.loadURDF(urdf_path,(obj_x,obj_y,obj_z),useFixedBase=1)
        #pb.changeVisualShape(self.enemy,-1,rgbaColor=[0,0,1,1])
        #self.enemy = pb.createCollisionShape(pb.GEOM_BOX,radius=0.5)
        self.enemy = pb.createCollisionShape(pb.GEOM_SPHERE,halfExtents=[0.5,0.5,0.5])
        pb.changeVisualShape(self.enemy,-1,rgbaColor=[1,1,0,1])
        boxUid = pb.createMultiBody(1,self.enemy,-1)
        self.ecid = pb.createConstraint(self.enemy,-1,-1,-1,pb.JOINT_FIXED,[0,0,0],[0,0,0],[0,0,1])
    def action_space(self):
        return range(4)

    def action_space_n(self):
        return 5
    def load_agent(self):
        obj_x = 0 #random.randint(0,10)
        obj_y = 0 #random.randint(0,10)
        obj_z = 0
        """
        path = os.path.dirname(inspect.getfile(inspect.currentframe()))
        print("path",path)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        stlfile = path+"/../../../tests/data/sphere.stl"
        copyfile(stlfile, dir_path+"/data/file.stl")
        urdf_path = dir_path+"/data/loader.stl.urdf"
        print("urdfpath",urdf_path)
        self.agent = pb.loadURDF(urdf_path,(obj_x,obj_y,obj_z),useFixedBase=1)
        pb.changeVisualShape(self.agent,-1,rgbaColor=[1,0,0,1])
        """

        self.agent = pb.createCollisionShape(pb.GEOM_SPHERE,halfExtents=[0.5,0.5,0.5])
        pb.changeVisualShape(self.agent,-1,rgbaColor=[1,0,0,1])
        boxUid = pb.createMultiBody(1,self.agent,-1)
        self.acid = pb.createConstraint(self.agent,-1,-1,-1,pb.JOINT_FIXED,[0,0,0],[0,0,0],[0,0,1])


    def reset(self):
        self.ax,self.ay,self.az = 0,0,0
        planeId = pb.createCollisionShape(pb.GEOM_PLANE)
        planeUid = pb.createMultiBody(0,planeId,0)
        pb.changeVisualShape(planeUid,-1,rgbaColor=[1,0.6,0.4,1])
        self.load_agent()
        self.load_enemies()

    def new_step():
        self.done = False
        # agent moves in xyz
        for mn in self.agent_motor_neurons:
            #fire(mn)
            pass
        x,y,z = get_xyz(self.agent)
        for enemy in self.enemies:
            new_x,new_y,new_z = new_distance(enemy, x,y,z)
            self.set_xyz(enemy,new_x,new_y,new_z)
            if False and is_touching_enemy(self.agent,self.enemy):
                done = True


        reward += 1
        # return observation,reward,done,info,homeostatis
        return observation,reward,done,info

    def _step(self,action):
        done = False
        self.steps +=1
        reward =1
        info=[42]
        if action == 0: #up
            self.ax += self.move
        elif action == 1: #down
            self.ax -= self.move
        elif action == 2: #left
            self.ay -= self.move
        elif action == 3: #right
            self.ay += self.move
        elif action == 4: #stop
            pass
        pivot=[self.ax,self.ay,self.az]
        orn = pb.getQuaternionFromEuler([0,0,0])
        pb.changeConstraint(self.acid,pivot,jointChildFrameOrientation=orn, maxForce=50)
        #pb.changeConstraint(self.acid,pivot,jointChildFrameOrientation=orn, maxForce=50)
        pb.stepSimulation()
        (x,y,z),_ = pb.getBasePositionAndOrientation(self.agent)
        enem_p = pb.getBasePositionAndOrientation(self.enemy)
        newx = enem_p[0][0]+((enem_p[0][0]+x)/2)
        newy = enem_p[0][1]+((enem_p[0][1]+y)/2)
        newz = enem_p[0][2]+((enem_p[0][2]+z)/2)
        pivot = [newx,newy,newz]
        orn = pb.getQuaternionFromEuler([0,0,0])
        #pb.changeConstraint(self.ecid,pivot,jointChildFrameOrientation=orn, maxForce=50)


        default = np.zeros((1))
        self.current_observation = default
        return self.current_observation,reward,done,info
