from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import os
import math

def normalize(vector):
    return vector / np.linalg.norm(vector)

def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None

def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance

def ray_tracing(x, y):
	# screen is on origin 
	pixel = np.array([x, y, 0]) 
	origin = camera 
	direction = normalize(pixel - origin) 
	color = np.zeros((3)) 
	reflection = 1 
	for k in range(max_depth): 
		# check for intersections 
		nearest_object, min_distance = nearest_intersected_object(objects, origin, direction) 
		if nearest_object is None: 
			break 
		intersection = origin + min_distance * direction 
		normal_to_surface = normalize(intersection - nearest_object['center']) 
		shifted_point = intersection + 1e-5 * normal_to_surface 
		intersection_to_light = normalize(light['position'] - shifted_point) 
		_, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light) 
		intersection_to_light_distance = np.linalg.norm(light['position'] - intersection) 
		is_shadowed = min_distance < intersection_to_light_distance 
		if is_shadowed: 
			break 
		illumination = np.zeros((3)) 
		# ambiant 
		illumination += nearest_object['ambient'] * light['ambient'] 
		# diffuse 
		illumination += nearest_object['diffuse'] * light['diffuse'] * np.dot(intersection_to_light, normal_to_surface) 
		# specular 
		intersection_to_camera = normalize(camera - intersection) 
		H = normalize(intersection_to_light + intersection_to_camera) 
		illumination += nearest_object['specular'] * light['specular'] * np.dot(normal_to_surface, H) ** (nearest_object['shininess'] / 4) 
		# reflection 
		color += reflection * illumination 
		reflection *= nearest_object['reflection'] 
		origin = shifted_point 
		direction = reflected(direction, normal_to_surface)
	return color

def orbitX(rad, period, now):
    ang = 2*(math.pi/period)*now
    return rad*math.cos(ang)

def orbitZ(rad, period, now):
    ang = 2*(math.pi/period)*now
    return rad*math.sin(ang)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

start_time = MPI.Wtime()

max_depth = 3

#### parameters

for num in range(120):

    width = 1620
    height = 1080
    camera = np.array([0, math.sin(math.pi*num/60), 3]) 
    light = { 'position': np.array([5, 5, 5]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) }
    objects = [
		{ 'center': np.array([0, 0, 0]), 'radius': 0.3, 'ambient': np.array([0.2, 0.1, 0]), 'diffuse': np.array([0.8, 0.1, 0.0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.7 },
		# { 'center': np.array([-0.2, 0, -1]), 'radius': 0.2, 'ambient': np.array([0.1, 0, 0]), 'diffuse': np.array([0.7, 1, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 80, 'reflection': 0.1 },
		{ 'center': np.array([orbitX(0.5,80,num), 0, orbitZ(0.5,80,num)]), 'radius': 0.04, 'ambient': np.array([0.1, 0.1, 0]), 'diffuse': np.array([0.5, 0.6, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
        { 'center': np.array([orbitX(0.65,90,num), 0, orbitZ(0.65,90,num)]), 'radius': 0.06, 'ambient': np.array([0.2, 0.1, 0]), 'diffuse': np.array([0.7, 0.3, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },        
        { 'center': np.array([orbitX(0.8,95,num), 0, orbitZ(0.8,95,num)]), 'radius': 0.06, 'ambient': np.array([0, 0.2, 0.1]), 'diffuse': np.array([0, 0.7, 0.2]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
        { 'center': np.array([orbitX(0.95,110,num), 0, orbitZ(0.95,110,num)]), 'radius': 0.05, 'ambient': np.array([0.15, 0.15, 0.1]), 'diffuse': np.array([0.6, 0.5, 0.4]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
        { 'center': np.array([orbitX(1.3,85,num), 0, orbitZ(1.3,85,num)]), 'radius': 0.13, 'ambient': np.array([0.15, 0.15, 0.05]), 'diffuse': np.array([0.6, 0.4, 0.1]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
        { 'center': np.array([orbitX(1.6,100,num), 0, orbitZ(1.6,100,num)]), 'radius': 0.1, 'ambient': np.array([0.2, 0.2, 0]), 'diffuse': np.array([0.7, 0.7, 0.5]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },        
        { 'center': np.array([orbitX(1.9,115,num), 0, orbitZ(1.9,155,num)]), 'radius': 0.08, 'ambient': np.array([0.1, 0.1, 0.2]), 'diffuse': np.array([0.5, 0.5, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
        { 'center': np.array([orbitX(2.2,105,num), 0, orbitZ(2.2,105,num)]), 'radius': 0.08, 'ambient': np.array([0, 0, 0.2]), 'diffuse': np.array([0, 0, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
		
		# { 'center': np.array([-0.3, 0, 0]), 'radius': 0.15, 'ambient': np.array([0, 0.1, 0]), 'diffuse': np.array([0, 0.6, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
		{ 'center': np.array([0, -101, 0]), 'radius': 100, 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.3 }
	]

    ratio = float(width) / height 
    screen = (-3, 3 / ratio, 3, -3 / ratio) # left, top, right, bottom


    image = np.zeros((height, width,3), dtype=float)
    Y = np.linspace(screen[1], screen[3], height)	
    X = np.linspace(screen[0], screen[2], width)


    localHeight = height // size + (height % size > rank) #하나의 코어에서 계산할 높이범위
    localY = Y[comm.scan(localHeight)-localHeight:comm.scan(localHeight)]  #하나의 코어에서 계산할 Y좌표들의 연속
    localImage = np.zeros((localHeight, width, 3))

    for i, y in enumerate(localY): #기존 Y => localY 로 바꾸고, 시작지점을 0에서 localY의 시작점으로 바군 for문
    	for j, x in enumerate(X):
            color = ray_tracing(x,y) 
            localImage[i, j] = np.clip(color, 0, 1)

    local_sizes = comm.gather(localImage.size)

	# Perform the Gatherv operation
    comm.Gatherv(sendbuf=localImage, recvbuf=(image, local_sizes), root=0)

	# Save the image on the root process
    if rank == 0:
        plt.imsave(('image' + str(num).zfill(2) + '.jpg'), image)

end_time = MPI.Wtime()
if rank == 0:
    command = 'ffmpeg -i ".\\image%02d.jpg" -vcodec libx264 -filter:v fps=30 ".\\output.mp4"'
    os.system(command)
    print("Overall elapsed time: " + str(end_time - start_time))


