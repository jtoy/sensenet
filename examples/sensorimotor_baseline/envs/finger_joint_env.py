import pybullet as pb
import numpy as np
import time,os,math,inspect,re,errno
import random,glob,math
from shutil import copyfile
import sensenet
from sensenet.error import Error
from sensenet import spaces
from sensenet.envs.handroid.hand_env import HandEnv
from sklearn.model_selection import train_test_split
import math

class FingerJointEnv(HandEnv):
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
        self.move = 0.015 # 0.01
        self.prev_distance = 10000000
        self.wandLength = 0.3
        self.wandSide = 0.005
        self.max_steps = 1000
        self.eyeOffset = 10e-3#0.26
        self.eye_pos = [0,0,0]
        self.target_pos = [0,0,0]
        self.action_space = spaces.Discrete(8)
        #self.cameraImageHeight = int((2 * 0.851 * self.wandSide)*23600) # yields 200x200 for height**2
        self.cameraImageHeight = int((2 * 0.851 * self.wandSide)*11800*1.2) # 100
        self.cameraImageWidth = int((2 * 0.851*self.wandSide)*11800*1.5) # 150
        # we will have 5 cameras at 125x100, for 5 * 12,500 pixels = 62,500 pixels
        # this helps with the CNN dimensions later
        self.files = None
        self.train_files = None
        self.test_files = None
        self.file_pointer = 0
        self.done_training = False
        self.load_files()

    def load_files(self):
        obj_type = 'obj'
        if 'data_path' in self.options and self.options['data_path'] is not None:
            path = self.options['data_path']
        else:
            raise KeyError('Expected data_path in command line')
        self.files = glob.glob(path+"/**/*."+obj_type,recursive=True)

        if 'test_split_ratio' in self.options and self.options['test_split_ratio'] is not None:
            testsize = self.options['test_split_ratio']
            self.train_files, self.test_files = train_test_split(self.files,
                                                                 test_size=testsize,
                                                                 random_state=42,
                                                                 shuffle=True)
    def load_object(self, mode=None):
        obj_x = 0
        obj_y = -1
        obj_z = 0
        obj_type = 'obj'

        if mode == 'train' and self.train_files is not None:
            stlfile = self.train_files[random.randrange(0, self.train_files.__len__())]
        elif mode == 'test' and self.test_files is not None:
            stlfile = self.test_files[random.randrange(0, self.test_files.__len__())]
        elif mode == 'train-all' and self.train_files is not None:
            if self.file_pointer + 1 < len(self.train_files):
                self.file_pointer += 1
            else:
                self.done_training = True
            stlfile = self.train_files[self.file_pointer]

        elif mode == None:
            stlfile = self.files[random.randrange(0, self.files.__len__())]
        #print(stlfile)
        if os.name == 'nt':
            self.class_label = int(stlfile.split("\\")[-3].split("_")[0])
        elif os.name == 'posix' or os.name == 'mac':
            self.class_label = int(stlfile.split("/")[-2].split("_")[0])

        dir_path = os.path.dirname(os.path.realpath(__file__))
        copyfile(stlfile, dir_path+"/data/file."+obj_type)
        urdf_path = dir_path+"/data/loader."+obj_type+".urdf"
        self.obj_to_classify = pb.loadURDF(urdf_path,(obj_x,obj_y,obj_z),useFixedBase=1)
        pb.changeVisualShape(self.obj_to_classify,-1,rgbaColor=[1,0,0,1])

    def load_agent(self):
        obj = pb.getBasePositionAndOrientation(self.obj_to_classify)
        target = obj[0]
        xyz=[target[0] - 1.0, target[1], target[2]]

        orientation = [0,0,0,1]
        dir_path = os.path.dirname(os.path.realpath(__file__))
        agent_path = dir_path + "/data/finger.urdf"
        self.agent = pb.loadURDF(agent_path,
                                 basePosition=xyz,
                                 baseOrientation=orientation)

        self.agent_cid = pb.createConstraint(self.agent,-1,-1,-1,jointType=pb.JOINT_FIXED,
                               jointAxis =[0,0,0],parentFramePosition=[0,0,0],
                               childFramePosition=[0,0,0])

        self.fov = 90
    def observation_space(self):
        #TODO return Box/Discrete
        #height = int((2 * 0.851 * self.wandSide)*11800)
        #height = int((2 * 0.851 * self.wandSide)*23600)
        return 5*self.cameraImageHeight*self.cameraImageWidth # 100x125


    def get_linkPosition_and_up_axis(self, direction):
        link_state = pb.getLinkState(self.agent, 0)
        link_p = link_state[0]
        link_o = link_state[1]

        # we define the up axis in terms of the object to classify since it is
        # stationary. If you define up in terms of the orientation of the agent
        # then your up axis can rotate as the agent rotates due to the normal
        # contact forces with the object to classify. This causes the camera
        # image to exhibit some unwanted behavior, such as inverting the up
        # and down directions.
        # If your object to classify is not stationary, then you could switch to
        # agent orientation to define your up axis.

        state = pb.getBasePositionAndOrientation(self.obj_to_classify)
        state_o = state[1]
        mat = pb.getMatrixFromQuaternion(state_o)
        axisZ2 = [mat[2], mat[5], mat[8]] # real axisZ2
        axisY2 = [-mat[1], -mat[4], -mat[7]]
        axisX2 = [mat[0], mat[3], mat[6]]
        if direction == 'negY' or direction == 'posX' or direction == 'posY':
            return link_p, axisZ2
        elif direction == 'posZ' or direction == 'negZ':
            return link_p, axisX2
        else:
            return ValueError('Unexpected value for direction.')

    def negY_view(self):
        link_p, up = self.get_linkPosition_and_up_axis('negY')
        self.eye_pos = [link_p[0]+0.01, link_p[1]-0.01, link_p[2]]
        self.target_pos = [link_p[0]+0.01,link_p[1]-0.01-10e-6,link_p[2]]

        if 'debug' in self.options and self.options['debug'] == True:
            pb.addUserDebugLine(link_p, self.eye_pos, [0.0, 1.0, 0.0], 10.0, lifeTime=0.1)

        viewMatrix = pb.computeViewMatrix(self.eye_pos,self.target_pos,up)

        return viewMatrix

    def negZ_view(self):
        link_p, up = self.get_linkPosition_and_up_axis('negZ')
        self.eye_pos = [link_p[0]-0.01, link_p[1], link_p[2]-0.01]
        self.target_pos = [link_p[0]-0.01,link_p[1],link_p[2]-0.01-10e-3]

        if 'debug' in self.options and self.options['debug'] == True:
            pb.addUserDebugLine(link_p, self.eye_pos, [0.0, 1.0, 0.0], 10.0, lifeTime=0.1)

        viewMatrix = pb.computeViewMatrix(self.eye_pos,self.target_pos,up)

        return viewMatrix

    def posX_view(self):
        link_p, up = self.get_linkPosition_and_up_axis('posX')
        self.eye_pos = [link_p[0],link_p[1], link_p[2]]
        self.target_pos = [link_p[0]+10e-6,link_p[1],link_p[2]]

        if 'debug' in self.options and self.options['debug'] == True:
            pb.addUserDebugLine(link_p, self.eye_pos, [0.0, 1.0, 0.0], 10.0, lifeTime=0.1)

        viewMatrix = pb.computeViewMatrix(self.eye_pos,self.target_pos,up)

        return viewMatrix

    def posY_view(self):
        link_p, up = self.get_linkPosition_and_up_axis('posY')
        self.eye_pos = [link_p[0]-0.01, link_p[1]+0.01, link_p[2]]
        self.target_pos = [link_p[0]-0.01,link_p[1]+0.01+10e-3,link_p[2]]

        if 'debug' in self.options and self.options['debug'] == True:
            pb.addUserDebugLine(link_p, self.eye_pos, [0.0, 1.0, 0.0], 10.0, lifeTime=0.1)

        viewMatrix = pb.computeViewMatrix(self.eye_pos,self.target_pos,up)

        return viewMatrix

    def posZ_view(self):
        link_p, up = self.get_linkPosition_and_up_axis('posZ')
        self.eye_pos = [link_p[0]-0.01, link_p[1], link_p[2]+0.01]
        self.target_pos = [link_p[0]-0.01,link_p[1],link_p[2]+0.01+10e-3]

        if 'debug' in self.options and self.options['debug'] == True:
            pb.addUserDebugLine(link_p, self.eye_pos, [0.0, 1.0, 0.0], 10.0, lifeTime=0.1)

        viewMatrix = pb.computeViewMatrix(self.eye_pos,self.target_pos,up)

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
        agent_po = pb.getBasePositionAndOrientation(self.agent)
        x = agent_po[0][0]
        y = agent_po[0][1]
        z = agent_po[0][2]

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

        elif action == 6: # move finger
            joint1Pos = pb.getJointState(self.agent, 0)[0]
            newJoint1Pos = joint1Pos - 0.01
            pb.setJointMotorControl2(self.agent,0,pb.POSITION_CONTROL,newJoint1Pos,force=1)

        elif action == 7: # move finger
            joint1Pos = pb.getJointState(self.agent, 0)[0]
            newJoint1Pos = joint1Pos + 0.01
            pb.setJointMotorControl2(self.agent,0,pb.POSITION_CONTROL,newJoint1Pos,force=1)

        elif action == 9:
            #print(self.current_observation[:100])
            #print(np.amax(self.current_observation))
            print(np.unique(self.current_observation))
            if self.is_touching():
                print('I am touching')
            else:
                print('I am not touching')
        pivot = [x,y,z]

        orn = pb.getQuaternionFromEuler([0,0,0])
        pb.changeConstraint(self.agent_cid,pivot,
                            jointChildFrameOrientation=orn,
                            maxForce=50)

        posX_viewMatrix = self.posX_view()
        posY_viewMatrix = self.posY_view()
        posZ_viewMatrix = self.posZ_view()
        negY_viewMatrix = self.negY_view()
        negZ_viewMatrix = self.negZ_view()

        projectionMatrix = pb.computeProjectionMatrixFOV(self.fov,aspect,
                                                    nearPlane,farPlane)
        #cameraImageHeight = int((2 * 0.851 * self.wandSide)*11800)
        #cameraImageHeight = int((2 * 0.851 * self.wandSide)*23600)

        w,h,img_arr,depths_posX,mask = pb.getCameraImage(self.cameraImageWidth,
                                                    self.cameraImageHeight,
                                                    posX_viewMatrix,
                                                    projectionMatrix,
                                                    lightDirection,
                                                    lightColor,
                                                    renderer=pb.ER_TINY_RENDERER)

        w,h,img_arr,depths_posY,mask = pb.getCameraImage(self.cameraImageWidth,
                                                    self.cameraImageHeight,
                                                    posY_viewMatrix,
                                                    projectionMatrix,
                                                    lightDirection,
                                                    lightColor,
                                                    renderer=pb.ER_TINY_RENDERER)

        w,h,img_arr,depths_posZ,mask = pb.getCameraImage(self.cameraImageWidth,
                                                    self.cameraImageHeight,
                                                    posZ_viewMatrix,
                                                    projectionMatrix,
                                                    lightDirection,
                                                    lightColor,
                                                    renderer=pb.ER_TINY_RENDERER)

        w,h,img_arr,depths_negY,mask = pb.getCameraImage(self.cameraImageWidth,
                                                    self.cameraImageHeight,
                                                    negY_viewMatrix,
                                                    projectionMatrix,
                                                    lightDirection,
                                                    lightColor,
                                                    renderer=pb.ER_TINY_RENDERER)

        w,h,img_arr,depths_negZ,mask = pb.getCameraImage(self.cameraImageWidth,
                                                    self.cameraImageHeight,
                                                    negZ_viewMatrix,
                                                    projectionMatrix,
                                                    lightDirection,
                                                    lightColor,
                                                    renderer=pb.ER_TINY_RENDERER)

        info = [42] #answer to life,TODO use real values
        pb.stepSimulation()

        self.steps += 1

        reward = 0

        if self.is_touching():
            touch_reward = 100000
            new_obs_posX = np.absolute(depths_posX - 1.0, dtype=np.float16)
            new_obs_posY = np.absolute(depths_posY - 1.0, dtype=np.float16)
            new_obs_posZ = np.absolute(depths_posZ - 1.0, dtype=np.float16)
            new_obs_negY = np.absolute(depths_negY - 1.0, dtype=np.float16)
            new_obs_negZ = np.absolute(depths_negZ - 1.0, dtype=np.float16)
            # if you want binary representation of depth camera, uncomment
            # the line below
            #new_obs[new_obs > 0] = 1
            new_obs = np.concatenate((new_obs_posX, new_obs_posY, new_obs_posZ, new_obs_negY,  new_obs_negZ))
            self.current_observation = new_obs.flatten()

        else:
            touch_reward = 0
            self.current_observation = np.zeros((self.observation_space()))
        agent_po = pb.getBasePositionAndOrientation(self.agent)
        obj_po = pb.getBasePositionAndOrientation(self.obj_to_classify)

        distance = math.sqrt(sum([(xi-yi)**2 for xi,yi in zip(obj_po[0],agent_po[0])])) #TODO faster euclidean

        if distance < self.prev_distance:
            reward += 1 * (self.max_steps - self.steps)
        elif distance > self.prev_distance:
            reward -= 10
        reward -= distance
        reward += touch_reward
        self.prev_distance = distance
        if self.steps >= self.max_steps or self.is_touching():
            done = True
        return self.current_observation,reward,done,info

    def is_touching(self):
        points = pb.getContactPoints(self.agent,self.obj_to_classify)
        return len(points) > 0 #and np.amax(self.current_observation) > 0

    def _reset(self,mode=None, options={}): # mode is meant to accommodate splitting data for train/test
        # load a new object to classify
        # move agent to -1.25,0,0 relative to object

        if bool(options):
            self.options = options #for reloading a specific shape
        pb.resetSimulation()

        self.load_object(mode)
        self.load_agent()
        default = np.zeros((self.observation_space()), dtype=np.float16)
        self.steps = 0
        self.current_observation = default
        return default
