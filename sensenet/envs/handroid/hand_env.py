import pybullet as pb
import numpy as np
import time,os,math,inspect,re,errno
import random,glob,math
from shutil import copyfile
import sensenet
from sensenet import spaces
from sensenet.error import Error

class HandEnv(sensenet.SenseEnv):
    metadata = {
        'render.modes': ['rgb_array'],
        'video.frames_per_second' : 30
    }
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
        #print("options",options)
        self.steps = 0

        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        parentdir = os.path.dirname(os.path.dirname(currentdir))
        os.sys.path.insert(0,parentdir)
        if 'render' in self.options and self.options['render'] == True:
            pb.connect(pb.GUI)
        else:
            pb.connect(pb.DIRECT)
        if 'video' in self.options and self.options['video'] != True:
            video_name = self.options['video'] #how to default to video.mp4 is empty
            pb.startStateLogging(pb.STATE_LOGGING_VIDEO_MP4, video_name)
        if 'debug' in self.options and self.options['debug'] == True:
            pb.configureDebugVisualizer(pb.COV_ENABLE_GUI,1)
            pb.configureDebugVisualizer(pb.COV_ENABLE_TINY_RENDERER,1)
        else:
            pb.configureDebugVisualizer(pb.COV_ENABLE_GUI,0)
            pb.configureDebugVisualizer(pb.COV_ENABLE_TINY_RENDERER,0)
        pb.setGravity(0,0,-10)
        pb.setRealTimeSimulation(0)
        self.move = 0.05 # 0.01
        self.load_object()
        self.load_agent()
        self.pi = 3.1415926535
        self.pinkId = 0
        self.middleId = 1
        self.indexId = 2
        self.thumbId = 3
        self.ring_id = 4
        self.indexEndID = 21 # Need get position and orientation from index finger parts
        self.offset = 0.02 # Offset from basic position
        self.downCameraOn = False
        self.prev_distance = 10000000
        self.action_space = spaces.Discrete(8)
        self.touch_width = 200
        self.touch_height = 200
        self.observation_space = spaces.Box(0, 1, [self.touch_width, self.touch_height])
    def load_object(self):
        #we assume that the directory structure is: SOMEPATH/classname/SHA_NAME/file
        #TODO refactor this whole mess later after addings tons of unit testing!!!!
        obj_x = 0
        obj_y = -1
        obj_z = 0
        if 'random_orientation' in self.options:
            orientation = (random.random(),random.random(),random.random(),random.random())
        else:
            orientation = (0,0,1,0)

        if 'obj_type' in self.options:
            obj_type = self.options['obj_type']
        elif 'obj_path' in self.options and self.options['obj_path'] != None and 'obj' in self.options['obj_path']:
            obj_type = 'obj'
        elif 'obj_path' in self.options and self.options['obj_path'] != None and 'stl' in self.options['obj_path']:
            obj_type = 'stl'
        else: #TODO change default to obj after more performance testing
            obj_type = 'obj'
        if 'obj_path' not in self.options or ('obj_path' in self.options and self.options['obj_path'] == None):
            path = self.get_data_path()
            if path == None:
                dir_path = os.path.dirname(os.path.realpath(__file__))
                stlfile = dir_path + "/data/pyramid.stl"
                stlfile = dir_path + "/data/missile.obj"
            else:
                files = glob.glob(path+"/**/*."+obj_type,recursive=True)
                try:
                    stlfile = files[random.randrange(0,files.__len__())]
                except ValueError:
                    raise Error("No %s objects found in %s folder!"
                                    % (obj_type, path))
                #TODO copy this file to some tmp area where we can guarantee writing
                if os.name == 'nt':
                    self.class_label = int(stlfile.split("\\")[-2].split("_")[0])
                elif os.name == 'posix' or os.name == 'mac':
                    self.class_label = int(stlfile.split("/")[-2].split("_")[0])
                #class labels are folder names,must be integer or N_text
                #print("class_label: ",self.class_label)
        else:
            stlfile = self.options['obj_path']
        dir_path = os.path.dirname(os.path.realpath(__file__))
        copyfile(stlfile, dir_path+"/data/file."+obj_type)
        urdf_path = dir_path+"/data/loader."+obj_type+".urdf"
        self.obj_to_classify = pb.loadURDF(urdf_path,(obj_x,obj_y,obj_z),orientation,useFixedBase=1)
        pb.changeVisualShape(self.obj_to_classify,-1,rgbaColor=[1,0,0,1])

    def classification_n(self):
        if self.get_data_path() == None:
            return None
        else:
            path = os.path.join(self.get_data_path(), "*/")
            subd = glob.glob(path)
            return len(subd)


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
        #pb.resetBasePositionAndOrientation(self.agent,(obj_po[0][0],obj_po[0][1]+0.5,obj_po[0][2]),obj_po[1])

    def observation_space(self):
        #TODO return Box/Discrete
        return 40000  #200x200
        #return 10000  #100x100
    def label(self):
        pass

    def action_space(self):
        #yaw pitch role finger
        #yaw pitch role hand
        #hand forward,back
        #0 nothing
        # 1 2 x + -
        # 3 4 y + -
        # 5 6 z + -
        # 21-40 convert to -1 to 1 spaces for finger movement
        #return base_hand + [x+21 for x in range(20)]
        base = [x for x in range(26)]
        base = [x for x in range(8)]
        #base = [x for x in range(6,26)]
        return base

    def action_space_n(self):
        return len(self.action_space())

    def _render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])
        base_pos = [0,0,0]
        _cam_dist = 5  #.3
        _cam_yaw = 50
        _cam_pitch = -35
        _render_width=480
        _render_height=480

        view_matrix = pb.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=_cam_dist,
            yaw=_cam_yaw,
            pitch=_cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = pb.computeProjectionMatrixFOV(
            fov=90, aspect=float(_render_width)/_render_height,
            nearVal=0.01, farVal=100.0)
        #proj_matrix=[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]
        (_, _, px, _, _) = pb.getCameraImage(
            width=_render_width, height=_render_height, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=pb.ER_BULLET_HARDWARE_OPENGL)
            #projectionMatrix=proj_matrix, renderer=pb.ER_TINY_RENDERER) #ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (_render_height, _render_width, 4))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def ahead_view(self):
        link_state = pb.getLinkState(self.agent,self.indexEndID)
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
            #p.addUserDebugLine(link_p,[link_p[0]+0.1*axisY[0],link_p[1]+0.1*axisY[1],link_p[2]+0.1*axisY[2]],[1,0,0],2,0.05) # Debug line in camera direction
            #pb.addUserDebugLine(link_p,[link_p[0]+0.1*axisY[0],link_p[1]+0.1*axisY[1],link_p[2]+0.1*axisY[2]],[1,0,0],2,0.2)
            pass

        return viewMatrix

    def down_view():
        link_state = pb.getLinkState(hand,indexEndID)
        link_p = link_state[0]
        link_o = link_state[1]
        handmat = pb.getMatrixFromQuaternion(link_o)

        axisX = [handmat[0],handmat[3],handmat[6]]
        axisY = [-handmat[1],-handmat[4],-handmat[7]] # Negative Y axis
        axisZ = [handmat[2],handmat[5],handmat[8]]

        eye_pos    = [link_p[0]-self.offset*axisZ[0],link_p[1]-self.offset*axisZ[1],link_p[2]-self.offset*axisZ[2]]
        target_pos = [link_p[0]-axisZ[0],link_p[1]-axisZ[1],link_p[2]-axisZ[2]] # Target position based on negative Z axis
        up = axisY # Up is Y axis
        viewMatrix = pb.computeViewMatrix(eye_pos,target_pos,up)

        if 'render' in self.options and self.options['render'] == True:
            pb.addUserDebugLine(link_p,[link_p[0]-0.1*axisZ[0],link_p[1]-0.1*axisZ[1],link_p[2]-0.1*axisZ[2]],[1,0,0],2,0.05) # Debug line in camera direction

        return viewMatrix

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
        pb.changeDynamics(self.agent, linkIndex=-1, spinningFriction=100.0, angularDamping=35.0,
                contactStiffness=0.0, contactDamping=100)

        if action == 65298 or action == 0: #down
            pb.changeConstraint(self.hand_cid,(hand_po[0][0]+self.move,hand_po[0][1],hand_po[0][2]),ho, maxForce=50)
        elif action == 65297 or action == 1: #up
            pb.changeConstraint(self.hand_cid,(hand_po[0][0]-self.move,hand_po[0][1],hand_po[0][2]),ho, maxForce=50)
        elif action == 65295 or action == 2: #left
            pb.changeConstraint(self.hand_cid,(hand_po[0][0],hand_po[0][1]+self.move,hand_po[0][2]),ho, maxForce=50)
        elif action== 65296 or action == 3: #right
            pb.changeConstraint(self.hand_cid,(hand_po[0][0],hand_po[0][1]-self.move,hand_po[0][2]),ho, maxForce=50)
        elif action == 44 or action == 4: #<
            pb.changeConstraint(self.hand_cid,(hand_po[0][0],hand_po[0][1],hand_po[0][2]+self.move), ho, maxForce=50)
        elif action == 46 or action == 5: #>
            pb.changeConstraint(self.hand_cid,(hand_po[0][0],hand_po[0][1],hand_po[0][2]-self.move), ho, maxForce=50)
        elif action >= 6 and action <= 7:
            # keeps the hand from moving towards origin
            pb.changeConstraint(self.hand_cid,(hand_po[0][0],hand_po[0][1],hand_po[0][2]),ho, maxForce=50)
            if action == 7:
                action = 25 #bad kludge redo all this code
            """
            self.pinkId = 0
            self.middleId = 1
            self.indexId = 2
            self.thumbId = 3
            self.ring_id = 4

            pink = convertSensor(self.pinkId) #pinkId != indexId -> return random uniform
            middle = convertSensor(self.middleId) # middleId != indexId -> return random uniform

            thumb = convertSensor(self.thumbId) # thumbId != indexId -> return random uniform
            ring = convertSensor(self.ring_id) # ring_id != indexId -> return random uniform
            """
            index = convertAction(action) # action = 6 or 25 due to kludge -> return -1 or 1
            """
            pb.setJointMotorControl2(self.agent,7,pb.POSITION_CONTROL,self.pi/4.)
            pb.setJointMotorControl2(self.agent,9,pb.POSITION_CONTROL,thumb+self.pi/10)
            pb.setJointMotorControl2(self.agent,11,pb.POSITION_CONTROL,thumb)
            pb.setJointMotorControl2(self.agent,13,pb.POSITION_CONTROL,thumb)
            """
            # Getting positions of the index joints to use for moving to a relative position
            joint17Pos = pb.getJointState(self.agent, 17)[0]
            joint19Pos = pb.getJointState(self.agent, 19)[0]
            joint21Pos = pb.getJointState(self.agent, 21)[0]
            # need to keep the multiplier relatively small otherwise the joint will continue to move
            # when you take other actions
            finger_jump = 0.2
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
            """
            pb.setJointMotorControl2(self.agent,17,pb.POSITION_CONTROL,index)
            pb.setJointMotorControl2(self.agent,19,pb.POSITION_CONTROL,index)
            pb.setJointMotorControl2(self.agent,21,pb.POSITION_CONTROL,index)

            pb.setJointMotorControl2(self.agent,24,pb.POSITION_CONTROL,middle)
            pb.setJointMotorControl2(self.agent,26,pb.POSITION_CONTROL,middle)
            pb.setJointMotorControl2(self.agent,28,pb.POSITION_CONTROL,middle)

            pb.setJointMotorControl2(self.agent,40,pb.POSITION_CONTROL,pink)
            pb.setJointMotorControl2(self.agent,42,pb.POSITION_CONTROL,pink)
            pb.setJointMotorControl2(self.agent,44,pb.POSITION_CONTROL,pink)

            ringpos = 0.5*(pink+middle)
            pb.setJointMotorControl2(self.agent,32,pb.POSITION_CONTROL,ringpos)
            pb.setJointMotorControl2(self.agent,34,pb.POSITION_CONTROL,ringpos)
            pb.setJointMotorControl2(self.agent,36,pb.POSITION_CONTROL,ringpos)
            """
        if self.downCameraOn: viewMatrix = down_view()
        else: viewMatrix = self.ahead_view()
        projectionMatrix = pb.computeProjectionMatrixFOV(fov,aspect,nearPlane,farPlane)
        w,h,img_arr,depths,mask = pb.getCameraImage(self.touch_width,self.touch_height, viewMatrix,projectionMatrix, lightDirection,lightColor,renderer=pb.ER_TINY_RENDERER)
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
        if 'debug' in self.options and self.options['debug'] == True:
            print("touch reward ",touch_reward)
            print("action ",action)
            print("reward ",reward)
            print("distance ",distance)
        if self.steps >= max_steps or self.is_touching():
            done = True
        return self.current_observation,reward,done,info

    def is_touching(self):
        points = pb.getContactPoints(self.agent,self.obj_to_classify)
        return len(points) > 0 and np.amax(self.current_observation) > 0

    def disconnect(self):
        pb.disconnect()
    def _reset(self,options={}):
        # load a new object to classify
        # move hand to 0,0,0
        if bool(options):
            self.options = options #for reloading a specific shape
        pb.resetSimulation()
        self.load_object()
        self.load_agent()
        #return observation
        default = np.zeros((self.touch_width * self.touch_height))
        self.steps = 0
        self.current_observation = default
        return default
