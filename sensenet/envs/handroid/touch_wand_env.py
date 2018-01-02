import pybullet as pb
import numpy as np
import time,os,math,inspect,re,errno
import random,glob,math
from shutil import copyfile
import sensenet
from sensenet.error import Error
from sensenet.utils import mkdir_p
from sensenet.envs.handroid.hand_env import HandEnv

class TouchWandEnv(HandEnv):
#class TouchWandEnv(sensenet.SenseEnv):

    def __init__(self,options={}):
        self.options = options
        self.steps = 0

        #TODO check if options is a string, so we know which environment to load
        if 'render' in self.options and self.options['render'] == True:
            pb.connect(pb.GUI)
        else:
            pb.connect(pb.DIRECT)
        pb.setGravity(0,0,-0.001)
        pb.setRealTimeSimulation(0)
        self.move = 0.05 # 0.01
        self.prev_distance = 10000000


    def load_agent(self):
        xyz=[1,-1,0]
        orientation = [0,1,0,1]
        self.agent = pb.createCollisionShape(pb.GEOM_CYLINDER,height=0.7,radius=0.05)
        self.agent_mb= pb.createMultiBody(1,self.agent,-1,basePosition=xyz,baseOrientation=orientation)
        pb.changeVisualShape(self.agent,-1,rgbaColor=[1,1,0,1])
        self.agent_cid = pb.createConstraint(self.agent,-1,-1,-1,pb.JOINT_FIXED,[0,0,0],[0,0,0],xyz,childFrameOrientation=orientation)

    def observation_space(self):
        #TODO return Box/Discrete
        return 40000  #200x200

    def action_space(self):
        base = [x for x in range(5)]
        return base

    def action_space_n(self):
        return len(self.action_space())

    def ahead_view(self):
        self.offset = 0.02
        link_state = pb.getBasePositionAndOrientation(self.agent)
        link_p = link_state[0]
        link_o = link_state[1]
        handmat = pb.getMatrixFromQuaternion(link_o)

        axisX = [handmat[0],handmat[3],handmat[6]]
        axisY = [-handmat[1],-handmat[4],-handmat[7]] # Negative Y axis
        axisZ = [handmat[2],handmat[5],handmat[8]]

        eye_pos    = [link_p[0]+self.offset*axisY[0],link_p[1]+self.offset*axisY[1],link_p[2]+self.offset*axisY[2]]
        target_pos = [link_p[0]+axisY[0],link_p[1]+axisY[1],link_p[2]+axisY[2]] # target position based by axisY, not X
        up = axisZ # Up is Z axis
        viewMatrix = pb.computeViewMatrix(eye_pos,target_pos,up)

        if 'render' in self.options and self.options['render'] == True:
            pb.addUserDebugLine(link_p,[link_p[0]-35.1,link_p[1]+3.1*axisY[1],link_p[2]+0.1*axisY[2]],[1,0,0],2,0.2)
            #pb.addUserDebugLine(link_p,[link_p[0]+0.1*axisY[0],link_p[1]+3.1*axisY[1],link_p[2]+0.1*axisY[2]],[1,0,0],2,0.2)
#            pb.addUserDebugLine(link_p,[link_p[0]+0.1*axisY[0],link_p[1]+0.1*axisY[1],link_p[2]+0.1*axisY[2]],[1,0,0],2,0.2)
            pass

        return viewMatrix


    def _step(self,action):
        done = False

        aspect = 1
        camTargetPos = [0,0,0]
        yaw = 40
        pitch = 10.0
        roll=0
        upAxisIndex = 2
        camDistance = 4
        pixelWidth = 320
        pixelHeight = 240
        nearPlane = 0.0001
        farPlane = 0.022
        lightDirection = [0,1,0]
        lightColor = [1,1,1]#optional
        fov = 50  #10 or 50
        agent_po = pb.getBasePositionAndOrientation(self.agent_mb)
        x = agent_po[0][0]
        y = agent_po[0][1]
        z = agent_po[0][2]
       
        #print("xyz",x,y,z)
        if action == 0: #down
          x += self.move
        elif action == 1: #up            
          x -= self.move
        elif action == 2: #left            
          y += self.move
        elif action == 3: #right            
          y -= self.move
        elif action == 4: #<        
          z += self.move
        elif action == 5: #>            
          z -= self.move
        pivot = [x,y,z]
        #print("newxyz",x,y,z)
        orn = pb.getQuaternionFromEuler([0,0,0])
        pb.changeConstraint(self.agent_cid,pivot,jointChildFrameOrientation=[0,1,0,1],maxForce=50)
        viewMatrix = self.ahead_view()
        projectionMatrix = pb.computeProjectionMatrixFOV(fov,aspect,nearPlane,farPlane)
        w,h,img_arr,depths,mask = pb.getCameraImage(200,200, viewMatrix,projectionMatrix, lightDirection,lightColor,renderer=pb.ER_TINY_RENDERER)
        new_obs = np.absolute(depths-1.0)
        new_obs[new_obs > 0] =1
        self.current_observation = new_obs.flatten()
        self.current_observation = [42]
        info = [42] #answer to life,TODO use real values
        pb.stepSimulation()
        self.steps +=1
        #reward if moving towards the object or touching the object
        reward = 0
        max_steps = 1000
        if self.is_touching():
            touch_reward = 10
        else:
            touch_reward = 0
        #obj_po = pb.getBasePositionAndOrientation(self.obj_to_classify)
        #distance = math.sqrt(sum([(xi-yi)**2 for xi,yi in zip(obj_po[0],agent_po[0])])) #TODO faster euclidean
        #distance =  np.linalg.norm(obj_po[0],hand_po[0])
        distance = 999
        #print("distance:",distance)
        if distance < self.prev_distance:
            reward += 1 * (max_steps - self.steps)
        elif distance > self.prev_distance:
            reward -= 10
        reward -= distance
        reward += touch_reward
        self.prev_distance = distance
        if self.steps >= max_steps or self.is_touching():
            done = True
        return self.current_observation,reward,done,info

    def is_touching(self):
        points = pb.getContactPoints(self.agent,self.obj_to_classify)
        return len(points) > 0 and np.amax(self.current_observation) > 0

    def _reset(self):
        # load a new object to classify
        # move hand to 0,0,0
        pb.resetSimulation()
        self.load_agent()
        self.load_object()
        default = np.zeros((self.observation_space()))
        self.steps = 0
        self.current_observation = default
        return default
