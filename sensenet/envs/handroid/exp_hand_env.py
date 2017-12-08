import pybullet as pb
import time,os,math,inspect,re
import random,glob,math
from shutil import copyfile
import numpy as np
import RenseEnv
class HandEnv: #SenseEnv
  def bootstrap_env(self,options={}):
    self.move = 0.01
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
    self.load_random_object()

  def load_random_object(self):
    #we assume that the directory structure is: SOMEPATH/classname/SHA_NAME/file
    #TODO make path configurable
    obj_x = 0
    obj_y = -1
    obj_z = 0 
    path = self.get_path()
    files = glob.glob(path+"/../touchable_data/objects/**/*.stl",recursive=True)
    stlfile = files[random.randrange(0,files.__len__())]
    copyfile(stlfile, path+"/data/file.stl")
    self.class_label = int(stlfile.split("/")[-3])
    print("class_label: ",self.class_label)
    self.obj_to_classify = pb.loadURDF("loader.urdf",(obj_x,obj_y,obj_z),useFixedBase=1)
    pb.changeVisualShape(self.obj_to_classify,-1,rgbaColor=[1,0,0,1])

  def classification_n(self):
    subd = glob.glob("../../touchable_data/objects/*/")
    return len(subd)


  def load_agent(self):
    self.agent = env.loadAgentModel("MPL/MPL.xml")

  def observation_space(self):
    #TODO  figure out how to read from a human bodyo
    total = 0
    for s in self.agent_sensors:
      total += s['output_size']

    return total

  def action_space(self):
    base = [x for x in range(26)]
    base = [x for x in range(8)]
    #base = [x for x in range(6,26)]
    #some description of actions
    return base

  def action_space_n(self):
    return len(self.action_space())

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
      pb.addUserDebugLine(link_p,[link_p[0]+0.1*axisY[0],link_p[1]+0.1*axisY[1],link_p[2]+0.1*axisY[2]],[1,0,0],2,0.2)

    return viewMatrix



  def stepSimulation(self,action):
    done = False
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

    hand_po = pb.getBasePositionAndOrientation(self.agent)
    #print("action",action)
    ho = pb.getQuaternionFromEuler([0,0,0])
    #hand_cid = pb.createConstraint(self.hand,-1,-1,-1,pb.JOINT_FIXED,[0,0,0],(0.1,0,0),hand_po[0],ho,hand_po[1])
    if action == 65298 or action == 0: #down
      #pb.changeConstraint(hand_cid,(hand_po[0][0]+self.move,hand_po[0][1],hand_po[0][2]),hand_po[1], maxForce=50)
      pb.resetBasePositionAndOrientation(self.agent,(hand_po[0][0]+self.move,hand_po[0][1],hand_po[0][2]),hand_po[1])
    elif action == 65297 or action == 1: #up
      pb.resetBasePositionAndOrientation(self.agent,(hand_po[0][0]-self.move,hand_po[0][1],hand_po[0][2]),hand_po[1])
      #pb.changeConstraint(hand_cid,(hand_po[0][0]-self.move,hand_po[0][1],hand_po[0][2]),hand_po[1], maxForce=50)
    elif action == 65295 or action == 2: #left
      pb.resetBasePositionAndOrientation(self.agent,(hand_po[0][0],hand_po[0][1]+self.move,hand_po[0][2]),hand_po[1])
      #pb.changeConstraint(hand_cid,(hand_po[0][0],hand_po[0][1]+self.move,hand_po[0][2]),hand_po[1], maxForce=50)
    elif action== 65296 or action == 3: #right
      pb.resetBasePositionAndOrientation(self.agent,(hand_po[0][0],hand_po[0][1]-self.move,hand_po[0][2]),hand_po[1])
      #pb.changeConstraint(hand_cid,(hand_po[0][0],hand_po[0][1]-self.move,hand_po[0][2]),hand_po[1], maxForce=50)
    elif action == 44 or action == 4: #<
      pb.resetBasePositionAndOrientation(self.agent,(hand_po[0][0],hand_po[0][1],hand_po[0][2]+self.move),hand_po[1])
      #pb.changeConstraint(hand_cid,(hand_po[0][0],hand_po[0][1],hand_po[0][2]+self.move),hand_po[1], maxForce=50)
    elif action == 46 or action == 5: #>
      pb.resetBasePositionAndOrientation(self.agent,(hand_po[0][0],hand_po[0][1],hand_po[0][2]-self.move),hand_po[1])
      #pb.changeConstraint(hand_cid,(hand_po[0][0],hand_po[0][1],hand_po[0][2]-self.move),hand_po[1], maxForce=50)
    elif action >= 6 and action <= 7:
    #elif action >= 6 and action <= 40:
      if action == 7:
        action = 25 #bad kludge redo all this code
      pink = convertSensor(self.pinkId)
      middle = convertSensor(self.middleId)
      index = convertAction(action)
      thumb = convertSensor(self.thumbId)
      ring = convertSensor(self.ring_id)

      pb.setJointMotorControl2(self.agent,7,pb.POSITION_CONTROL,self.pi/4.)	
      pb.setJointMotorControl2(self.agent,9,pb.POSITION_CONTROL,thumb+self.pi/10)
      pb.setJointMotorControl2(self.agent,11,pb.POSITION_CONTROL,thumb)
      pb.setJointMotorControl2(self.agent,13,pb.POSITION_CONTROL,thumb)

      # That's index finger parts
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
    self.current_observation = observation
    self.img_arr = img_arr
    self.depths= depths
    info = [42] #TODO use real values
    #reward if moving towards the object or touching the object
    reward = 0
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
      reward += 1
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
    return observation,reward,done,info

  def is_sensing(self):
    r = self.img_arr[:,:,0]
    g = self.img_arr[:,:,1]
    b = self.img_arr[:,:,2]
    #print("g max", (np.max(g) == 0))
    #print("b max", (np.max(b) == 0))
    #print("r max", (np.max(r) == 0))
    #no red
    return(np.max(g) == 0 and np.max(b) == 0 and np.max(r) > 0)
    #return (np.amax(self.observation) > 0)

