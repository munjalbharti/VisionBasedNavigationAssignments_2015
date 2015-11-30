# Message containing images and synchronized imu-data
# Timing:
# ------ t -------->
#              |       image[0]
#              |       image[1]
#              |       image[n] 
#  |  |  |  |  |       imu data 
# 
# Imu data contains all imu data since last image(s).
# The last element aligns with the current image(s).

# The image(s) from the camera(s)
sensor_msgs/Image[] image

# Imu message(s)
sensor_msgs/Imu[] imu