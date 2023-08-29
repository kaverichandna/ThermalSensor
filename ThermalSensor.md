# Thermal Sensor

**_Overview_**

> The project aims to develop code for a sensor that captures faces and lists their corresponding temperatures.

## Prerequisites

- Raspberry Pi 4 with 2+ Ram
- Thermal Imaging Sensor - [MLX90640](https://robu.in/product/waveshare-mlx90640-ir-array-thermal-imaging-camera-32x24-pixels-110-fov/)
- USB Camera - Webcam
- Python 3.9
- OpenCV `pip install opencv-python`
- Thermal Camera Library `pip install adafruit-circuitpython-mlx90640`
- Jumper Wires for connecting raspberry to sensor

## Connections

Raspberry pi and Camera communicates using **i2c protocol**, raspberry pi comes with **i2c** turned off. Turn it on using `sudo raspi-config`

![Wiring Diagram](https://images.squarespace-cdn.com/content/v1/59b037304c0dbfb092fbe894/1591731759228-C66M7BWPEH5KPK3UYZ9A/mlx90640_rpi_wiring_diagram_w_table.png?format=1500w)

<!-- | Raspberry Pi Pin | Camera Pin | -->
<!-- | ---------------- | ---------- | -->
<!-- | 3.3v             | 3-6v       | -->
<!-- | SDA              | SDA        | -->
<!-- | SCL              | SCL        | -->
<!-- | GND              | GND        | -->

## Functionality

```mermaid
flowchart TD
    Start --> Get-Video-->|Resize Aspect Ratio|Detect-Face
    Detect-Face --> Get-Temperature-Array --> Get-Average-Temperature --> Display-Video
    Get-Video <--> Display-Video
```

## Face Tracking

### Getting Video from Webcam

Intially we started by capturing the video. Here we have utilised the OpenCV library to capture the video live from the webcam and use a face classifier to track the face.

```python
import cv2

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

video_capture = cv2.VideoCapture(0)
```

### Mapping Thermal video frame to Webcam Video

The aspect ratios of the thermal camera and the webcam can be differerent,
we need to make it both in the same aspect ratio so that its **easier to map the webcam pixels to thermal camera pixels**

| Thermal | Webcam    |
| ------- | --------- |
| 32x24   | 1920x1080 |

> We need to make the camera the same aspect ratio of thermal camera for the best results

```python
def update_webcam_ratio(frame, inter=cv2.INTER_AREA):
    # 20 Times bigger than the sensor
    width = 640
    height = 480
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = frame.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return frame

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(frame, dim, interpolation=inter)

    # return the resized image
    return resized
```

### Get the bounding box of the face

The algorithm will track your face and create a green bounding box around it regardless of where you move within the frame.

```python
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces
```

When calling this function it will return an array of

- x
- y
- width
- height

**<span style="color:teal"> x,y values denote the left top corner of the bounding box </span>**

## Getting thermal grid from the sensor

```python
import time,board,busio
import numpy as np
import adafruit_mlx90640

i2c = busio.I2C(board.SCL, board.SDA, frequency=400000) # setup I2C
mlx = adafruit_mlx90640.MLX90640(i2c) # begin MLX90640 with I2C comm
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ # set refresh rate

frame = np.zeros((24*32,)) # setup array for storing all 768 temperatures

def get_thermal_grid():
    try:
        mlx.getFrame(frame) # read MLX temperatures into frame var
        break
    except ValueError:
        pass
    return frame
```

### Getting the thermal values

We needed to map the values and then take the average of all the grids that are mapped

```python
def map_cam_to_thermal(x):
    return x // 20

def get_thermal_value(x, y, w, h):
    grid = get_thermal_grid().reshape(24, 32)
    x = map_cam_to_thermal(x)
    y = map_cam_to_thermal(y)
    w = map_cam_to_thermal(w)
    h = map_cam_to_thermal(h)

    # TODO: Average instead of center point

    center_x = x + (w / 2)
    center_y = y + (h / 2)

    return grid[center_x][center_y]
```

## Putting it all together

Once we have functions to get the temperature average and face detection we can combine both to get the temperature of the face that is dectected

```python
while True:

    result, video_frame = video_capture.read()
    if result is False:
        break

    frame = update_webcam_ratio(video_frame)
    faces = detect_bounding_box(frame)

    for (x, y, w, h) in faces:
        avg_temp = get_thermal_value(x, y, w, h)
        frame = cv2.putText(
            frame,
            f"avg temp {avg_temp}c",
            (x, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow( "Thermal body sensor", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
```

## Extra Features for Better Accuracy

- <span style="color:green">Webcam - Thermal Camera Grid Array Aspect Size Matching &rarr; Done</span>
- <span style="color:orange">Keeping the Webcam close to the Thermal Camera physically &rarr; Comes under Enclosure Design</span>
- <span style="color:red">Take Average of the bounding box mapped temperatures &rarr; Not Done</span>
- <span style="color:red">Calibrating Webcam - Thermal Camera detection position using physical tests and adding x and y offset &rarr; Not Done</span>

## Links

- [Face Detection with OpenCV](https://www.datacamp.com/tutorial/face-detection-python-opencv)
- [Thermal Camera](https://makersportal.com/blog/2020/6/8/high-resolution-thermal-camera-with-raspberry-pi-and-mlx90640)
- [Image Crop Reize](https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv)
