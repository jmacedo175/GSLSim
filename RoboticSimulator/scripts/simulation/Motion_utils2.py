import math
import random
#import silkworm_moth
#import dung_beetle
class MotionUtils:
	def __init__(self,dist_threshold, angular_threshold,simulation_step,com_duration,robot,params, clock):
		self.dist_threshold=dist_threshold
		self.angular_threshold=angular_threshold
		self.simulation_step=simulation_step
		self.com_duration=com_duration
		self.robot = robot
		self.params = params
		self.plumeLostThreshold = self.params['plumeLostThreshold']
		
		self.initialize_variables()
		self.clock = clock

	def initialize_variables(self):

		self.start_position=None
		self.linear_displacement=None
		self.com_start=None
		self.angular_displacement=None
		self.action = None
		self.action_completed = True
		self.spiral_count = 0
		self.spiral_direction = None
		self.rotation_amplitude=0
		self.last_location=None
		self.dest_cos=None
		self.dest_sin=None	
		self.state = 1
		self.targ_heading = None


	def stop(self):
		self.robot.target=[0.0,0.0]
		return True


	def move_straight(self, motion_amplitude,x,y,h):
		motion_terminated = False
		#self.odom_lock.acquire()

		if(self.linear_displacement==None or self.start_position==None):
			self.com_start=self.clock.get_time()
			self.start_position = [x,y,h]
			self.linear_displacement=0.0
			#self.last_location = [x,y,h]
		self.linear_displacement=math.sqrt(pow(x-self.start_position[0],2)+pow(y-self.start_position[1],2))

		
		#self.last_location = [self.robot.x,self.robot.y,self.robot.heading]
		#print('motion utils',	self.linear_displacement,self.dist_threshold,motion_amplitude,self.linear_displacement+self.dist_threshold<motion_amplitude)
		if(self.linear_displacement+self.dist_threshold<motion_amplitude):# and (self.clock.get_time()-self.com_start)<self.com_duration):
			x = min(motion_amplitude-self.linear_displacement,1)
			self.robot.target=[x,0.0]#self.cmd_publisher.publish('<v'+str(x)+',0>')
		else:
			#self.robot.target=[0.0,0.0]#  STOP ROBOT self.cmd_publisher.publish('<v0,0>')
			self.linear_displacement=None
			self.start_position=None
			motion_terminated=True
			#self.last_location=None
		#self.odom_lock.release()
		
		return motion_terminated

	def allign(self,angle):
		#print(abs(angle),self.angular_threshold)
		if(self.targ_heading == None):
			self.targ_heading = angle#self.fix_angle(angle+self.robot.heading)
			#print('Setting the target heading to ',self.targ_heading)
		#print(self.robot.x, self.robot.y, self.robot.heading)
		#remaining = self.targ_heading-self.robot.heading
		remaining = self.fix_angle(self.targ_heading-self.robot.heading)#self.fix_angle(self.targ_heading-self.robot.heading)
		#print('Allign target', angle+self.robot.heading,self.targ_heading, 'remaining', remaining, 'robot heading', self.robot.heading)

		#print('angle', angle, 'target_heading', self.targ_heading, 'rob_heading',self.robot.heading, 'remaining turn raw',self.targ_heading-self.robot.heading, 'fixed', remaining)

		if(abs(remaining)>self.angular_threshold):
			#print('Must rotate', abs(remaining))
			y=min(1.0, math.sin(remaining))
			if(remaining<0):
				y=max(-1.0, math.sin(remaining))

			self.robot.target=[0,y]#self.cmd_publisher.publish('<v0.0,'+str(y)+'>')
			return False
		else:
			#print('ended at ',remaining)
			#self.robot.target=[0.0,0.0]#self.cmd_publisher.publish('<v0,0>')
			self.targ_heading=None
			return True

	def fix_angle(self,w):
		w = w%(2*math.pi)
		if(w>math.pi):
			w-=2*math.pi
		return w

	def distance(self,x1,y1,x2,y2):
		return math.sqrt(math.pow(x1-x2,2)+math.pow(y1-y2,2))

