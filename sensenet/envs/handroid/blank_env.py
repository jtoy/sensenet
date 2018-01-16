import pybullet as pb
import numpy as np
import time,os,math,inspect,re,errno
import random,glob,math
from shutil import copyfile
import sensenet
from sensenet.error import Error

class BlankEnv(sensenet.SenseEnv):
    def getKeyboardEvents(self):
        return pb.getKeyboardEvents()

    def get_data_path(self):
        if 'data_path' in  self.options:
            path = self.options['data_path']
        else:
            #path = os.path.dirname(inspect.getfile(inspect.currentframe()))
            path = None
        return path

    def __init__(self,options={}):
        self.options = options
        self.steps = 0

        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        parentdir = os.path.dirname(os.path.dirname(currentdir))
        os.sys.path.insert(0,parentdir)
        #TODO check if options is a string, so we know which environment to load
        if 'render' in self.options and self.options['render'] == True:
            pb.connect(pb.GUI)
        else:
            pb.connect(pb.DIRECT)
        pb.setGravity(0,0,0)
        pb.setRealTimeSimulation(0)
        self.move = 0.05 # 0.01
        self.load_object()
        #self.load_agent()
    def load_object(self):
        #we assume that the directory structure is: SOMEPATH/classname/SHA_NAME/file
        #TODO make path configurable
        #TODO refactor this whole mess later
        obj_x = 0
        obj_y = -1
        obj_z = 0
        if 'obj_type' in self.options:
            obj_type = self.options['obj_type']
        elif 'obj_path' in self.options and 'obj' in self.options['obj_path']:
            obj_type = 'obj'
        else: #TODO change default to obj after more performance testing
            obj_type = 'stl'
        if 'obj_path' not in self.options:
            path = self.get_data_path()
            if path == None:
                dir_path = os.path.dirname(os.path.realpath(__file__))
                stlfile = dir_path + "/data/pyramid.stl"
            else:
                files = glob.glob(path+"/**/*."+obj_type,recursive=True)
                try:
                    stlfile = files[random.randrange(0,files.__len__())]
                except ValueError:
                    raise Error("No %s objects found in %s folder!"
                                    % (obj_type, path))
                #TODO copy this file to some tmp area where we can guarantee writing
                if os.name == 'nt':
                    self.class_label = int(stlfile.split("\\")[-3].split("_")[0])
                elif os.name == 'posix' or os.name == 'mac':
                    self.class_label = int(stlfile.split("/")[-3].split("_")[0])
                #class labels are folder names,must be integer or N_text
                #print("class_label: ",self.class_label)
        else:
            stlfile = self.options['obj_path']
        dir_path = os.path.dirname(os.path.realpath(__file__))
        copyfile(stlfile, dir_path+"/data/file."+obj_type)
        urdf_path = dir_path+"/data/loader."+obj_type+".urdf"
        self.obj_to_classify = pb.loadURDF(urdf_path,(obj_x,obj_y,obj_z),useFixedBase=1)
        pb.changeVisualShape(self.obj_to_classify,-1,rgbaColor=[1,0,0,1])


    def load_agent(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        agent_path = dir_path + "/data/MPL/MPL.xml"
        objects = pb.loadMJCF(agent_path,flags=0)
        self.agent=objects[0]  #1 total
        #if self.obj_to_classify:
        obj_po = pb.getBasePositionAndOrientation(self.obj_to_classify)
        self.hand_cid = pb.createConstraint(self.agent,-1,-1,-1,pb.JOINT_FIXED,[0,0,0],[0,0,0],[0,0,0])

    def observation_space(self):
        #TODO return Box/Discrete
        return 1

    def action_space(self):
        return [0]

    def action_space_n(self):
        return len(self.action_space())

    def _step(self,action):
        done = False
        pb.stepSimulation()
        self.steps +=1
        reward =1
        info=[42]
        default = np.zeros((1))
        self.current_observation = default
        return self.current_observation,reward,done,info

    def disconnect(self):
        pb.disconnect()
    def _reset(self):
        pb.resetSimulation()
        self.load_object()
        #self.load_agent()
        default = np.zeros((1))
        self.current_observation = default
        return default
