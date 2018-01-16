import pybullet as pb
import numpy as np
import time,os,math,inspect,re,errno
import random,glob,math
from shutil import copyfile

import sensenet
from sensenet.error import Error
from sensenet.envs.handroid.hand_env import HandEnv

class IndexFingerHandEnv(HandEnv):

    def __init__(self,options={}):
        self.options = options
        #print("options",options)
        self.steps = 0

        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        parentdir = os.path.dirname(os.path.dirname(currentdir))
        os.sys.path.insert(0,parentdir)
        #TODO check if options is a string, so we know which environment to load
        if 'render' in self.options and self.options['render'] == True:
            pb.connect(pb.GUI)
        else:
            pb.connect(pb.DIRECT)
        self.timeStep = 0.6 # 0.04 by default
        pb.setPhysicsEngineParameter(fixedTimeStep=self.timeStep)
        pb.setGravity(0,0,0)
        pb.setRealTimeSimulation(0)
        self.move = 0.1 #0.05 # 0.01
        self.pi = 3.1415926535
        self.tinyForce = 5*10e-6
        self.load_object()
        self.load_agent()

        self.pinkId = 0
        self.middleId = 1
        self.indexId = 2
        self.thumbId = 3
        self.ring_id = 4
        self.indexEndID = 21 # Need get position and orientation from index finger parts
        self.offset = 0.02 # Offset from basic position
        self.downCameraOn = False
        self.prev_distance = 10000000

    def load_agent(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        agent_path = dir_path + "/data/MPL/MPL.xml"
        objects = pb.loadMJCF(agent_path,flags=0)
        self.agent=objects[0]  #1 total
        #if self.obj_to_classify:
        obj_po = pb.getBasePositionAndOrientation(self.obj_to_classify)
        self.hand_cid = pb.createConstraint(self.agent,-1,-1,-1,pb.JOINT_FIXED,[0,0,0],[0,0,0],[0,0,0])
        #hand_po = pb.getBasePositionAndOrientation(self.agent)
        #distance = math.sqrt(sum([(xi-yi)**2 for xi,yi in zip(obj_po[0],hand_po[0])])) #TODO faster euclidean
        pb.resetBasePositionAndOrientation(self.agent,(obj_po[0][0],obj_po[0][1]+0.5,obj_po[0][2]),obj_po[1])

        hand_po = pb.getBasePositionAndOrientation(self.agent)

        ho = pb.getQuaternionFromEuler([0.0, 0.0, 0.0])
        pb.changeConstraint(self.hand_cid,(hand_po[0][0],hand_po[0][1],hand_po[0][2]),ho, maxForce=200)
        
        joint7Pos = pb.getJointState(self.agent, 7)[0]
        joint24Pos = pb.getJointState(self.agent, 24)[0]
        joint32Pos = pb.getJointState(self.agent, 32)[0]
        joint40Pos = pb.getJointState(self.agent, 40)[0]
        
        # these are pretty close
        joint7tgt = self.pi/4 # thumb
        joint9tgt = self.pi/2
        joint11tgt = self.pi/2
        joint13tgt = self.pi/2
        # pretty close
        joint24tgt = self.pi/2.25 # middle finger
        joint26tgt = self.pi/2
        joint28tgt = self.pi/2
        # working
        joint32tgt = self.pi/2.5 # ring finger
        joint34tgt = self.pi/2
        joint36tgt = self.pi/2
        # pretty close
        joint40tgt = self.pi/3 # pinky
        joint42tgt = self.pi/2
        joint44tgt = self.pi/2

        while (joint7Pos < joint7tgt) and (joint24Pos < joint24tgt) and (joint32Pos < joint32tgt) and (joint40Pos < joint40tgt):

            pb.setJointMotorControl2(self.agent,7,pb.POSITION_CONTROL,joint7tgt) 
            pb.setJointMotorControl2(self.agent,9,pb.POSITION_CONTROL,joint9tgt)
            pb.setJointMotorControl2(self.agent,11,pb.POSITION_CONTROL,joint11tgt)
            pb.setJointMotorControl2(self.agent,13,pb.POSITION_CONTROL,joint13tgt)

            pb.setJointMotorControl2(self.agent,24,pb.POSITION_CONTROL,joint24tgt)
            pb.setJointMotorControl2(self.agent,26,pb.POSITION_CONTROL,joint26tgt)
            pb.setJointMotorControl2(self.agent,28,pb.POSITION_CONTROL,joint28tgt)

            pb.setJointMotorControl2(self.agent,32,pb.POSITION_CONTROL,joint32tgt)
            pb.setJointMotorControl2(self.agent,34,pb.POSITION_CONTROL,joint34tgt)
            pb.setJointMotorControl2(self.agent,36,pb.POSITION_CONTROL,joint36tgt)
            
            pb.setJointMotorControl2(self.agent,40,pb.POSITION_CONTROL,joint40tgt)
            pb.setJointMotorControl2(self.agent,42,pb.POSITION_CONTROL,joint42tgt)
            pb.setJointMotorControl2(self.agent,44,pb.POSITION_CONTROL,joint44tgt)
 
            pb.stepSimulation()

            joint7Pos = pb.getJointState(self.agent, 7)[0]
            joint24Pos = pb.getJointState(self.agent, 24)[0]
            joint32Pos = pb.getJointState(self.agent, 32)[0]
            joint40Pos = pb.getJointState(self.agent, 40)[0]
            #return random.random()
    
    def _step(self,action):
        done = False
        #reward (float): amount of reward achieved by the previous action. The scale varies between environments, but the goal is always to increase your total reward.
        #done (boolean): whether it's time to reset the environment again. Most (but not all) tasks are divided up into well-defined episodes, and done being True indicates the episode has terminated. (For example, perhaps the pole tipped too far, or you lost your last life.)
        #observation (object): an environment-specific object representing your observation of the environment. For example, pixel data from a camera, joint angles and joint velocities of a robot, or the board state in a board game.
        #info (dict): diagnostic information useful for debugging. It can sometimes be useful for learning (for example, it might contain the raw probabilities behind the environment's last state change). However, official evaluations of your agent are not allowed to use this for learning.
        def convertSensor(finger_index):
            if finger_index == self.indexId:
                return random.uniform(-1,1)
                #return 0
            else:
                #return random.uniform(-1,1)
                return 0
        def convertAction(action):
            #converted = (action-30)/10
            #converted = (action-16)/10
            if action == 6:
                converted = -1
            elif action == 25:
                converted = 1
            #print("action ",action)
            #print("converted ",converted)
            return converted

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
        hand_po = pb.getBasePositionAndOrientation(self.agent)
        ho = pb.getQuaternionFromEuler([0.0, 0.0, 0.0])        
       
        # So when trying to modify the physics of the engine, we run into some problems. If we leave
        # angular damping at default (0.04) then the hand rotates when moving up and dow, due to torque.
        # If we set angularDamping to 100.0 then the hand will bounce off into the background due to 
        # all the stored energy, when it makes contact with the object. The below set of parameters seem
        # to have a reasonably consistent performance in keeping the hand level and not inducing unwanted
        # behavior during contact. 

        pb.changeDynamics(self.agent, linkIndex=-1, angularDamping=0.9999)

        if action == 65298 or action == 0: #down
            pb.changeConstraint(self.hand_cid,(hand_po[0][0]+self.move,hand_po[0][1],hand_po[0][2]),ho, maxForce=self.tinyForce)
        elif action == 65297 or action == 1: #up            
            pb.changeConstraint(self.hand_cid,(hand_po[0][0]-self.move,hand_po[0][1],hand_po[0][2]),ho, maxForce=self.tinyForce)
        elif action == 65295 or action == 2: #left            
            pb.changeConstraint(self.hand_cid,(hand_po[0][0],hand_po[0][1]+self.move,hand_po[0][2]),ho, maxForce=self.tinyForce)
        elif action== 65296 or action == 3: #right            
            pb.changeConstraint(self.hand_cid,(hand_po[0][0],hand_po[0][1]-self.move,hand_po[0][2]),ho, maxForce=self.tinyForce)
        elif action == 44 or action == 4: #<        
            pb.changeConstraint(self.hand_cid,(hand_po[0][0],hand_po[0][1],hand_po[0][2]+self.move),ho, maxForce=self.tinyForce)            
        elif action == 46 or action == 5: #>            
            pb.changeConstraint(self.hand_cid,(hand_po[0][0],hand_po[0][1],hand_po[0][2]-self.move),ho, maxForce=self.tinyForce)
        elif action >= 6 and action <= 7:
            # keeps the hand from moving towards origin
            pb.changeConstraint(self.hand_cid,(hand_po[0][0],hand_po[0][1],hand_po[0][2]),ho, maxForce=200)
            if action == 7:
                action = 25 #bad kludge redo all this code

            index = convertAction(action) # action = 6 or 25 due to kludge -> return -1 or 1                        

            # Getting positions of the index joints to use for moving to a relative position
            joint17Pos = pb.getJointState(self.agent, 17)[0]
            joint19Pos = pb.getJointState(self.agent, 19)[0]
            joint21Pos = pb.getJointState(self.agent, 21)[0]
            # need to keep the multiplier relatively small otherwise the joint will continue to move
            # when you take other actions
            finger_jump = 0.1
            newJoint17Pos = joint17Pos + index*finger_jump
            newJoint19Pos = joint19Pos + index*finger_jump
            newJoint21Pos = joint21Pos + index*finger_jump
            
            # following values found by experimentation
            if newJoint17Pos <= -0.7:
                newJoint17Pos = -0.7
            elif newJoint17Pos >= 0.57:
                newJoint17Pos = 0.57
            if newJoint19Pos <= 0.13:
                newJoint19Pos = 0.13
            elif newJoint19Pos >= 0.42:
                newJoint19Pos = 0.42            
            if newJoint21Pos <= -0.8:
                newJoint21Pos = -0.8
            elif newJoint21Pos >= 0.58:
                newJoint21Pos = 0.58

            pb.setJointMotorControl2(self.agent,17,pb.POSITION_CONTROL,newJoint17Pos)
            pb.setJointMotorControl2(self.agent,19,pb.POSITION_CONTROL,newJoint19Pos)
            pb.setJointMotorControl2(self.agent,21,pb.POSITION_CONTROL,newJoint21Pos)

        if self.downCameraOn: viewMatrix = down_view()
        else: viewMatrix = self.ahead_view()
        projectionMatrix = pb.computeProjectionMatrixFOV(fov,aspect,nearPlane,farPlane)
        w,h,img_arr,depths,mask = pb.getCameraImage(200,200, viewMatrix,projectionMatrix, lightDirection,lightColor,renderer=pb.ER_TINY_RENDERER)
        #w,h,img_arr,depths,mask = pb.getCameraImage(200,200, viewMatrix,projectionMatrix, lightDirection,lightColor,renderer=pb.ER_BULLET_HARDWARE_OPENGL)
        new_obs = np.absolute(depths-1.0)
        new_obs[new_obs > 0] =1
        self.current_observation = new_obs.flatten()
        info = [42] #answer to life,TODO use real values
        pb.stepSimulation()

        self.steps +=1
        #reward if moving towards the object or touching the object
        reward = 0
        max_steps = 1000
        if self.is_touching():
            touch_reward = 10
            if 'debug' in self.options and self.options['debug'] == True:
                print("TOUCHING!!!!")
        else:
            touch_reward = 0
        obj_po = pb.getBasePositionAndOrientation(self.obj_to_classify)
        distance = math.sqrt(sum([(xi-yi)**2 for xi,yi in zip(obj_po[0],hand_po[0])])) #TODO faster euclidean
        #distance =  np.linalg.norm(obj_po[0],hand_po[0])
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
