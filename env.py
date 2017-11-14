import pybullet as pb
import time,os,math,inspect,re
import random,glob,math
from shutil import copyfile
import numpy as np
class TouchEnv:
  def __init__(self,options={}):
    self.options = options
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
    self.move = 0.01
    self.load_random_object()
    self.load_hand()
    self.pi = 3.1415926535
    self.pinkId = 0
    self.middleId = 1
    self.indexId = 2
    self.thumbId = 3
    self.ring_id = 4
    self.indexEndID = 21 # Need get position and orientation from index finger parts
    self.offset = 0.02 # Offset from basic position
    self.downCameraOn = False
    self.past_x = 0
    self.past_y = 0
    self.past_z = 0
    self.prev_distance = 10000000

  def load_random_object(self):
    #we assume that the directory structure is: SOMEPATH/classname/SHA_NAME/file
    #TODO make path configurable
    obj_x = 0
    obj_y = -1
    obj_z = 0 
    #TODO fix path issues
    files = glob.glob("../../touchable_data/objects/**/*.stl",recursive=True)
    stlfile = files[random.randrange(0,files.__len__())]
    copyfile(stlfile, "../data/file.stl")
    self.class_label = int(stlfile.split("/")[-3])
    print("class_label: ",self.class_label)
    self.obj_to_classify = pb.loadURDF("loader.urdf",(obj_x,obj_y,obj_z),useFixedBase=1)
    pb.changeVisualShape(self.obj_to_classify,-1,rgbaColor=[1,0,0,1])

  def classification_n(self):
    subd = glob.glob("../../touchable_data/objects/*/")
    return len(subd)


  def load_hand(self):
    objects = pb.loadMJCF("MPL/MPL.xml",flags=0)
    self.hand=objects[0]  #1 total

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
    #base = [x for x in range(4)]
    #base = [x for x in range(6,26)]
    return base

  def action_space_n(self):
    return len(self.action_space())

  def render(self):
    pass
  def ahead_view(self):
    link_state = pb.getLinkState(self.hand,self.indexEndID)
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
      pb.addUserDebugLine(link_p,[link_p[0]+0.1*axisY[0],link_p[1]+0.1*axisY[1],link_p[2]+0.1*axisY[2]],[1,0,0],2,0.2)

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


      #return random.random()
  def step(self,action):
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
      converted = (action-16)/10
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

    hand_po = pb.getBasePositionAndOrientation(self.hand)
    ho = pb.getQuaternionFromEuler([0,0,0]) #dont really know what this does
    hand_cid = pb.createConstraint(self.hand,-1,-1,-1,pb.JOINT_FIXED,[0,0,0],(0.1,0,0),hand_po[0],ho,hand_po[1])
    if action == 65298 or action == 0: #down
      pb.changeConstraint(hand_cid,(hand_po[0][0]+self.move,hand_po[0][1],hand_po[0][2]),hand_po[1], maxForce=50)
    elif action == 65297 or action == 1: #up
      pb.changeConstraint(hand_cid,(hand_po[0][0]-self.move,hand_po[0][1],hand_po[0][2]),hand_po[1], maxForce=50)
    elif action == 65295 or action == 2: #left
      pb.changeConstraint(hand_cid,(hand_po[0][0],hand_po[0][1]+self.move,hand_po[0][2]),hand_po[1], maxForce=50)
    elif action== 65296 or action == 3: #right
      pb.changeConstraint(hand_cid,(hand_po[0][0],hand_po[0][1]-self.move,hand_po[0][2]),hand_po[1], maxForce=50)
    elif action == 44 or action == 4: #<
      pb.changeConstraint(hand_cid,(hand_po[0][0],hand_po[0][1],hand_po[0][2]+self.move),hand_po[1], maxForce=50)
    elif action == 46 or action == 5: #>
      pb.changeConstraint(hand_cid,(hand_po[0][0],hand_po[0][1],hand_po[0][2]-self.move),hand_po[1], maxForce=50)
    elif action >= 6 and action <= 25:
    #elif action >= 21 and action <= 40:
      pink = convertSensor(self.pinkId)
      middle = convertSensor(self.middleId)
      index = convertAction(action)
      thumb = convertSensor(self.thumbId)
      ring = convertSensor(self.ring_id)

      pb.setJointMotorControl2(self.hand,7,pb.POSITION_CONTROL,self.pi/4.)	
      pb.setJointMotorControl2(self.hand,9,pb.POSITION_CONTROL,thumb+self.pi/10)
      pb.setJointMotorControl2(self.hand,11,pb.POSITION_CONTROL,thumb)
      pb.setJointMotorControl2(self.hand,13,pb.POSITION_CONTROL,thumb)

      # That's index finger parts
      pb.setJointMotorControl2(self.hand,17,pb.POSITION_CONTROL,index)
      pb.setJointMotorControl2(self.hand,19,pb.POSITION_CONTROL,index)
      pb.setJointMotorControl2(self.hand,21,pb.POSITION_CONTROL,index)

      pb.setJointMotorControl2(self.hand,24,pb.POSITION_CONTROL,middle)
      pb.setJointMotorControl2(self.hand,26,pb.POSITION_CONTROL,middle)
      pb.setJointMotorControl2(self.hand,28,pb.POSITION_CONTROL,middle)

      pb.setJointMotorControl2(self.hand,40,pb.POSITION_CONTROL,pink)
      pb.setJointMotorControl2(self.hand,42,pb.POSITION_CONTROL,pink)
      pb.setJointMotorControl2(self.hand,44,pb.POSITION_CONTROL,pink)

      ringpos = 0.5*(pink+middle)
      pb.setJointMotorControl2(self.hand,32,pb.POSITION_CONTROL,ringpos)
      pb.setJointMotorControl2(self.hand,34,pb.POSITION_CONTROL,ringpos)
      pb.setJointMotorControl2(self.hand,36,pb.POSITION_CONTROL,ringpos)
    if self.downCameraOn: viewMatrix = down_view()
    else: viewMatrix = self.ahead_view()
    projectionMatrix = pb.computeProjectionMatrixFOV(fov,aspect,nearPlane,farPlane)
    w,h,img_arr,depths,mask = pb.getCameraImage(200,200, viewMatrix,projectionMatrix, lightDirection,lightColor,renderer=pb.ER_TINY_RENDERER)
    #w,h,img_arr,depths,mask = pb.getCameraImage(200,200, viewMatrix,projectionMatrix, lightDirection,lightColor,renderer=pb.ER_BULLET_HARDWARE_OPENGL)
    #red_dimension = img_arr[:,:,0]  #TODO change this so any RGB value returns 1, anything else is 0
    red_dimension = img_arr[:,:,0].flatten()  #TODO change this so any RGB value returns 1, anything else is 0
    #observation = red_dimension
    self.img_arr = img_arr
    observation = (np.absolute(red_dimension -255) > 0).astype(int)
    self.observation = observation
    self.img_arr = img_arr
    self.depths= depths
    info = [123123] #TODO use real values
    pb.stepSimulation()
    #reward if moving towards the object or touching the object
    reward = 0
    reward += np.clip(np.amax(observation),0,1) #touching
    obj_po = pb.getBasePositionAndOrientation(self.obj_to_classify)
    distance = math.sqrt(sum([(xi-yi)**2 for xi,yi in zip(obj_po[0],hand_po[0])])) #TODO faster euclidean
    #distance =  np.linalg.norm(obj_po[0],hand_po[0])
    #print("distance:",distance)
    if distance < self.prev_distance:
      reward += 1
    elif distance > self.prev_distance:
      reward -= 10
    self.prev_distance = distance
    #print("reward",reward)
    return observation,reward,done,info

  def is_touching(self):
    #this function probably shouldnt be here
    #depth_f = self.depths.flatten()
    #m = np.ma.masked_where(depth_f>=1.0, depth_f) 
    #red_dimension = self.img_arr[:,:,0].flatten()  #TODO change this so any RGB value returns 1, anything else is 0
    #o = (np.ma.masked_where(np.ma.getmask(m), np.absolute(red_dimension -255) > 0)).astype(int)
    #return (np.amax(o) > 0)
    r = self.img_arr[:,:,0]
    #print("shape",r.size)
    g = self.img_arr[:,:,1]
    b = self.img_arr[:,:,2]
    return(np.max(g) == 0 and np.max(b) == 0 and np.max(r) > 0)
    #return (np.amax(self.observation) > 0)

  def reset(self):
    # load a new object to classify
    # move hand to 0,0,0
    pb.resetSimulation()
    self.load_random_object()
    self.load_hand()
    hand_po = pb.getBasePositionAndOrientation(self.hand)
    pb.resetBasePositionAndOrientation(self.hand,(0,0,0),hand_po[1])
    pb.stepSimulation()
    if self.downCameraOn: viewMatrix = down_view()
    else: viewMatrix = self.ahead_view()
    fov = 50  #10 or 50
    aspect = 1
    nearPlane = 0.01
    farPlane = 0.05
    lightDirection = [0,1,0]
    lightColor = [1,1,1]#optional
    projectionMatrix = pb.computeProjectionMatrixFOV(fov,aspect,nearPlane,farPlane)
    w,h,img_arr,depths,mask = pb.getCameraImage(200,200, viewMatrix,projectionMatrix, lightDirection,lightColor,renderer=pb.ER_TINY_RENDERER)
    red_dimension = img_arr[:,:,0].flatten()  #TODO change this so any RGB value returns 1, anything else is 0
    observation = red_dimension
    print("sizWTF")
    print("size",observation.size)
    return observation

  def disconnect(self):
    pb.disconnect()


