===============================================================================
 ROBOTIC WASTE MANAGEMENT SYSTEM
 Subsystem 2: Vision-Guided Floor Trash Collection Robotic Arm
 Installation and Deployment Guide
===============================================================================

 Author     : Ho Shuang Quan (Subsystem 2)
 Partner    : Harry Liow Siang Yi (Subsystem 1)
 Repository : https://github.com/jaft952/autobin
 Supervisor : Dr. Tew Yiqi

===============================================================================
 FOLDER STRUCTURE
===============================================================================

 autobin/
 |-- ReadMe.txt                  this file
 |-- requirements.txt            Python packages
 |-- cloudflared.deb             Cloudflare Tunnel installer (Raspberry Pi, arm64)
 |-- deploy/
 |   |-- autobin-server.service  systemd service (auto-start at boot)
 |   `-- README.md               short auto-start notes
 |-- src/
 |   |-- web/                    server.py (robot server), dashboard.py, static UI
 |   |-- subsumption/            arbitrator, layers 0-4, motion and arm executors
 |   |-- scanning/               zigzag patrol timing and geometry
 |   |-- visual_servoing/        reactive approach controller
 |   |-- perception/             YOLO detector, target lock
 |   |-- arm/                    arc grasp grid, calibration, arm settings
 |   |   `-- config/arc_grasp.yaml   calibrated grasp grid
 |   |-- hardware/               motor, servo, ultrasonic and camera drivers
 |   |-- motion/                 wheel pins and drive calibration
 |   |-- safety/                 obstacle avoidance
 |   `-- models/                 model weights (NOT in git, copy in by hand)
 |-- tests/                      hardware test and calibration scripts
 `-- ml/                         model training scripts (development PC)


===============================================================================
HARDWARE WIRING
===============================================================================

 All GPIO numbers below are BCM numbers, not physical pin numbers.

 Drive motors -> ZK-BM1 (src/motion/calibration.py)
 --------------------------------------------------
   IN1  BCM 12  (physical pin 32)    left motor
   IN2  BCM 13  (physical pin 33)    left motor
   IN3  BCM 18  (physical pin 12)    right motor
   IN4  BCM 19  (physical pin 35)    right motor

 Ultrasonic sensors (src/web/runtime.py)
 ---------------------------------------
   Front        TRIG BCM 23   ECHO BCM 24
   Back         TRIG BCM 17   ECHO BCM 20
   Front-left   TRIG BCM 27   ECHO BCM 22
   Front-right  TRIG BCM 5    ECHO BCM 6

 Arm servos -> PCA9685 over I2C (SDA pin 3, SCL pin 5)
 ----------------------------------------------------
   PCA9685 channels 0-5 = CH1 base, CH2 shoulder, CH3 elbow, CH4 wrist,
                          CH5 roll, CH6 gripper

 WARNING
   - HC-SR04 ECHO outputs 5 V. Use a voltage divider so the Pi pin sees 3.3 V.
   - Never power the servos from the Pi 5 V pin. Use the UBEC.
   - Keep hands clear of the arm and wheels whenever the battery is connected.
   - Read docs/hardware_safety_patterns.md before changing motor or servo code.


===============================================================================
 INSTALL ON THE RASPBERRY PI
===============================================================================

 Step 1. Prepare the operating system
 ------------------------------------
   Flash Raspberry Pi OS 64-bit with Raspberry Pi Imager.
   In Imager settings, set the user name, Wi-Fi and enable SSH.
   Boot the Pi, then connect over SSH or with a keyboard and screen.

     sudo apt update && sudo apt full-upgrade -y
     sudo apt install -y git python3-venv python3-dev build-essential \
                         libgl1 pigpio

 Step 2. Enable I2C (needed for the PCA9685 servo driver)
 --------------------------------------------------------
     sudo raspi-config
       -> Interface Options -> I2C -> Enable
     sudo reboot

   Check the servo board is detected (address 40 should appear):
     sudo apt install -y i2c-tools
     i2cdetect -y 1

 Step 3. (Optional) Start the pigpio daemon
 ------------------------------------------
   pigpio gives smoother motor PWM. Without it the code falls back to
   RPi.GPIO software PWM automatically.
     sudo systemctl enable --now pigpiod

 Step 4. Get the code
 --------------------
     cd ~/Desktop
     git clone https://github.com/jaft952/autobin.git
     cd autobin

 Step 5. Create the virtual environment and install packages
 -----------------------------------------------------------
     python3 -m venv venv
     source venv/bin/activate
     pip install --upgrade pip
     pip install -r requirements.txt

 Step 6. Replace PyTorch with the CPU-only build (REQUIRED on the Pi)
 --------------------------------------------------------------------
   The default PyTorch build crashes on the Pi with "Illegal instruction"
   (SIGILL). Replace it:
     pip uninstall -y torch torchvision
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

   Check it (after Step 7, once the model file is in place):
     python tests/diagnose_runtime.py --model src/models/yolov11n-seg.pt

 Step 7. Copy in the model and convert it to NCNN
 ------------------------------------------------
   Copy yolov11n-seg.pt into src/models/, then run:
     python tests/export_ncnn.py --model src/models/yolov11n-seg.pt

   This creates src/models/yolov11n-seg_ncnn_model/. The detector uses the
   NCNN folder automatically when it exists (2 to 4 times faster on the Pi).
   Run the export again whenever the .pt file is replaced.

===============================================================================
 RUN THE ROBOT
===============================================================================

 Start the server on the Pi
 --------------------------
     cd ~/Desktop/autobin
     source venv/bin/activate
     python src/web/server.py

   Options:
     --port 8000       server port (default 8000)
     --host 0.0.0.0    listen address (default all interfaces)
     --hz 20           control loop rate
     --no-camera       run without camera and YOLO (bench testing)

 Open the dashboard
 ------------------
   On any laptop or phone on the same network, open:
     http://<pi-ip>:8000

   Find <pi-ip> on the Pi with:   hostname -I

 Use the dashboard
 -----------------
   START            full autonomy: scan, approach, collect
   SCAN ONLY        zigzag patrol without collecting
   EMERGENCY STOP   stops all motion immediately
   Restart server   reloads the server after code changes
   Shutdown         stops the server and powers off the Pi
                    (robot must be stopped first)

 Operating area
 --------------
   Use an enclosed room with walls on all sides and the door closed.
   The robot has no map. A lane ends when a wall is 30 cm ahead or after 30 s.

===============================================================================
 CHECK THE HARDWARE
===============================================================================

 Run these from the repository root with the venv active
 (source venv/bin/activate). Lift the wheels off the floor for wheel tests.

   python tests/test_ultrasonic.py        ultrasonic readings from all sensors
   python tests/test_wheels.py            each wheel direction, speed and brake
   python tests/test_turn_direction.py    left and right pivots
   python tests/test_servo_travel.py      servo command vs real angle
   python tests/test_inference.py --model src/models/yolov11n-seg.pt
                                          camera and YOLO detection
   python tests/test_arc_live.py          live arc grasp viewer, press g to grab

 Calibration (only when the arm, wiring or floor changes)
 --------------------------------------------------------
   python tests/test_arc_grasp.py         record the arc grasp grid
                                          (press a to add a sample, saved to
                                          src/arm/config/arc_grasp.yaml)
   python tests/servo_jog.py              jog the arm and capture fixed poses
                                          such as the bin drop position
   python tests/test_arc_calibrate.py     tune wheel arc and pivot motions
   Zigzag timing (TURN_90_S, SHIFT_S, TURN_AT_CM) is in src/scanning/tuning.py.
   Turns are timed, so re-check them on each new floor surface.

 Logic tests without hardware (also run on Windows)
 ---------------------------------------------------
   python tests/test_scan_logic.py
   python tests/test_collect_logic.py
   (or: python -m pytest tests/test_scan_logic.py tests/test_collect_logic.py)








