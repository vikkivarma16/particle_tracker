import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

def calculate_scale(image_path, known_length_um=200):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    bar_length_px = max(w, h)
    scale_um_per_pixel = known_length_um / bar_length_px
    return scale_um_per_pixel

class KalmanFilter:
    def __init__(self, dt, u_noise, std_acc, std_meas):
        self.dt = dt
        self.u_noise = u_noise
        self.std_acc = std_acc
        self.std_meas = std_meas

        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 500
        self.F = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.B = np.array([[0.5 * self.dt**2],
                           [0.5 * self.dt**2],
                           [self.dt],
                           [self.dt]])
        self.Q = np.array([[self.dt**4/4, 0, self.dt**3/2, 0],
                           [0, self.dt**4/4, 0, self.dt**3/2],
                           [self.dt**3/2, 0, self.dt**2, 0],
                           [0, self.dt**3/2, 0, self.dt**2]]) * self.std_acc**2
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.R = np.array([[self.std_meas**2, 0],
                           [0, self.std_meas**2]])

    def predict(self, u=0):
        self.x = np.dot(self.F, self.x) + self.B * u
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

    def current_state(self):
        return self.x[:2].flatten()

# Example usage for scale calculation
scale_image_path = "image.png"  # Replace with your scale image path
scale_um_per_pixel = calculate_scale(scale_image_path)
print(f"Scale: {scale_um_per_pixel} um/pixel")

# Particle tracking script
video_path = "data_3.avi"
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

output_video_path = "particle_tracking_result.avi"
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

binary_output_video_path = "binary_particle_tracking_result.avi"
binary_out = cv2.VideoWriter(binary_output_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)

previous_positions = []
kalman_filters = []
missed_frames = []
particle_displacements = []
valid_particles = []
particle_colors = {}

particle_positions=[]

frame_counter = 0
calibration_frames = 50  # Number of frames to calibrate the Kalman filter
sudden_jump_threshold = 500 / scale_um_per_pixel  # Threshold for sudden jump detection (in pixels)
max_missed_frames = 10  # Number of frames to wait before considering a particle lost
max_distance = 25  # Maximum allowed distance for a particle to be considered the same

# Read the first frame outside the main loop
ret, first_frame = cap.read()
if ret:
    gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray_frame, 128, 255, cv2.THRESH_BINARY)
    labeled_frame, num_features = measure.label(binary_frame, connectivity=2, return_num=True)
    regions = measure.regionprops(labeled_frame)
    for region in regions:
        if 100 <= region.area <= 2500:
            y, x = region.centroid
            kf = KalmanFilter(dt=1/fps, u_noise=1, std_acc=1, std_meas=1)
            kf.update(np.array((x, y)).reshape((2, 1)))
            kalman_filters.append(kf)
            previous_positions.append((x, y))
            particle_positions.append([])
            missed_frames.append(0)
            particle_displacements.append([])
            valid_particles.append(len(kalman_filters) - 1)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            particle_colors[len(kalman_filters) - 1] = color

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    
    print("Frame counter is given as", frame_counter)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray_frame, 128, 255, cv2.THRESH_BINARY)
    labeled_frame, num_features = measure.label(binary_frame, connectivity=2, return_num=True)
    regions = measure.regionprops(labeled_frame)

    current_frame_particle_counter = 0
    current_positions = []

    for region in regions:
        if 100 <= region.area <= 3500:  # Adjust this threshold based on your particle size
            y, x = region.centroid
            current_positions.append((x, y))
            current_frame_particle_counter += 1

            # Perform Kalman filter calibration
            if frame_counter <= calibration_frames:
                min_dist, min_index = float('inf'), -1
                for j, prev_pos in enumerate(previous_positions):
                    dist = np.linalg.norm(np.array(prev_pos) - np.array((x, y)))
                    if dist < max_distance:
                        min_dist = dist
                        min_index = j

                if min_index == -1:
                    kf = KalmanFilter(dt=1/fps, u_noise=1, std_acc=1, std_meas=1)
                    kf.update(np.array((x, y)).reshape((2, 1)))
                    kalman_filters.append(kf)
                    previous_positions.append((x, y))
                    particle_positions.append([])
                    missed_frames.append(0)
                    particle_displacements.append([])
                    valid_particles.append(len(kalman_filters) - 1)
                    color = tuple(np.random.randint(0, 255, 3).tolist())
                    particle_colors[len(previous_positions) - 1] = color
                else:
                    kf = kalman_filters[min_index]
                    kf.update(np.array((x, y)).reshape((2, 1)))
                    previous_positions[min_index] = (x, y)
                    color = particle_colors[min_index]

                cv2.circle(frame, (int(x), int(y)), 10, color, 3)

    if frame_counter > calibration_frames:
        for i in range(len(previous_positions) - 1, -1, -1):
            kf = kalman_filters[i]
            kf.predict()
            min_dist, min_index = float('inf'), -1
            previous_pos_list = list(previous_positions[i])

            for j, pos in enumerate(current_positions):
                pos_list = list(pos)
                dist = np.linalg.norm(np.array(previous_pos_list) - np.array(pos_list))
                if dist < max_distance:
                    min_dist = dist
                    min_index = j

            if min_index != -1:
                kf.update(np.array(current_positions[min_index]).reshape((2, 1)))
                displacement = np.linalg.norm(np.array(list(current_positions[min_index])) - np.array(previous_positions[i])) * scale_um_per_pixel
                particle_displacements[i].append(displacement)
                particle_positions[i].append([list(current_positions[min_index]), frame_counter])
                previous_positions[i] = current_positions[min_index]
                missed_frames[i] = 0
                color = particle_colors[i]
                deviation = np.linalg.norm(kf.current_state() - np.array(list(current_positions[min_index]))) * scale_um_per_pixel

                cv2.circle(frame, (int(np.array(list(current_positions[min_index]))[0]), int(np.array(list(current_positions[min_index]))[1])), 10, color, 3)
                cv2.circle(binary_frame, (int(np.array(list(current_positions[min_index]))[0]), int(np.array(list(current_positions[min_index]))[1])), 10, color, 3)
                del current_positions[min_index]
            else:
                missed_frames[i] += 1
                if missed_frames[i] > max_missed_frames:
                    try:
                        valid_particles.remove(i)
                    except ValueError:
                        pass
    else:
        print("Number of detected particles during calibration: ", len(particle_displacements))

    

    #if frame_counter == 1:  # Draw scale bar on the first frame
    scale_bar_length = int(50 / scale_um_per_pixel)  # 50 micrometer scale bar in pixels
    scale_bar_position = (frame_width - 100, frame_height - 50)  # Position of the scale bar
    cv2.line(frame, scale_bar_position, (scale_bar_position[0] + scale_bar_length, scale_bar_position[1]), (255, 255, 255), 2)
    cv2.putText(frame, '50 um', (scale_bar_position[0], scale_bar_position[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
    binary_out.write(binary_frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
binary_out.release()
cv2.destroyAllWindows()

valid_displacements = []
valid_positions = []

for i, displacements in enumerate(particle_displacements):
    if i in valid_particles:
        valid_displacements.extend(displacements)

for i, posi in enumerate(particle_positions):
    if i in valid_particles:
        valid_positions.append(posi)

fu = open("MSD_vikki.txt", "w")

displacement = []
weight = []
for i in range(150):  # maximum up to the number of frames
    displacement.append(0.0)
    weight.append(0)
for i in range(len(valid_positions)):
    x_in = valid_positions[i][0][0][0]
    y_in = valid_positions[i][0][0][1]
    t_in = valid_positions[i][0][1]

    for j in range(len(valid_positions[i])):
        x = valid_positions[i][j][0][0]
        y = valid_positions[i][j][0][1]
        t = valid_positions[i][j][1]

        displace = ((x_in - x)**2.0) + ((y_in - y)**2.0)

        if j == 0:
            print(displace, x, y, x_in, y_in)
        t_index = int(t - t_in)
        weight[t_index] += 1
        displacement[t_index] += displace

data = []

for i in range(len(displacement)):
    if weight[i] > 0:
        fu.write(str(i))
        fu.write("   ")
        fu.write(str(displacement[i] / float(weight[i])))
        fu.write("\n")
        data.append([i / fps, displacement[i] / float(weight[i])])
fu.close()

if len(valid_displacements) > 0:
    time_intervals = np.arange(1, len(valid_displacements) + 1) / fps
    msd = np.cumsum(np.square(valid_displacements)) / np.arange(1, len(valid_displacements) + 1)

    # Save time vs. MSD data to a text file
    time_msd_file = 'time_vs_msd.txt'
    with open(time_msd_file, 'w') as f:
        f.write('Time (s)\tMSD (um^2)\n')
        for t, m in zip(time_intervals, msd):
            f.write(f'{t}\t{m}\n')

    # Calculate average MSD over valid particles
    average_msd = np.mean(msd)
    print(f"Average MSD: {average_msd} um^2")

    # Save average MSD to a text file
    average_msd_file = 'average_msd_excluding_jumps.txt'
    with open(average_msd_file, 'w') as f:
        f.write(f'Average MSD (um^2): {average_msd}\n')

    # Calculate the diffusivity (D)
    # MSD = 4 * D * t for 2D diffusion
    D = np.mean(msd / (4 * time_intervals))
    print(f"Estimated Diffusivity (D): {D} um^2/s")
else:
    print("No valid particles to calculate MSD and diffusivity.")

