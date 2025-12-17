# Speaker Notes for Pre-Defense Presentation

## Slide 1: Title Page (10 seconds)
- Good morning/afternoon everyone
- Today I'll present my thesis: Hawkeye System - 3D Reconstruction and Structure from Motion for volleyball tracking
- Takes about 9 minutes, happy to answer questions after

---

## Slide 2: Problem & Solution (1 minute)

### Problem:
- Professional systems like Hawk-Eye cost $60-70K
- The algorithms and methods aren't publicly available
- Limited research on accessible 3D ball tracking for volleyball

### Solution:
- I developed a stereo vision system for 3D reconstruction of volleyball trajectories
- Uses YOLOv8 for ball detection in synchronized stereo camera pairs
- Applies structure from motion principles - epipolar geometry and triangulation
- Open source implementation to enable further research

---

## Slide 3: System Architecture (1 minute)

### Walk through the pipeline:
- Start with a stereo camera pair capturing at 60 FPS
- YOLOv8n detects the ball in both camera views
  - Achieves 89.6% precision and 73% recall
- Stereo matching uses epipolar geometry to match the ball between left and right views
  - 89.2% success rate
- 3D triangulation reconstructs the ball position in 3D space
  - Gets a median error of just 3.8 centimeters
- Finally displays everything in an interactive 3D court visualization
  - Users can rotate, zoom, and analyze the trajectory

---

## Slide 4: Technical Implementation (1 minute)

### Ball Detection Module:
- Used YOLOv8n - the nano version optimized for speed
- Trained on 39,750 labeled images combining real volleyball footage from Roboflow and synthetic Blender data
- Detects the ball independently in both left and right camera frames

### Stereo Matching Module:
- Uses Semi-Global Block Matching - SGBM algorithm from OpenCV
- Computes pixel correspondences between the two camera views
- (Point to disparity map if shown) The disparity map shows how much pixels shift between left and right views
- (Point to depth map if shown) This gets converted to a depth map where color indicates distance from the camera

### 3D Reconstruction:
- Once we have matching points in both views, we use triangulation
- The math is straightforward: Z = f × B / d where f is focal length, B is baseline, d is disparity
- Camera-to-world transformation converts from camera coordinates to real-world volleyball court coordinates

### Visualization Features:
- Interactive 3D court view with ball trajectory
- Homography mapping provides a top-down bird's-eye view of the court
- Makes it easy to see exactly where the ball is positioned on the court

---

## Slide 5: Challenges (1.5 minutes)

### Challenge 1: Data Acquisition
- I didn't have access to stereo camera equipment
- Even if I had cameras, the footage needs to be synchronized between the two views
- **Solution**: Built a complete synthetic dataset in Blender with realistic volleyball rallies
- Had to manually synchronize frames using sliders, but it worked

### Challenge 2: Domain Gap
- This is a big problem in AI - models perform worse on data that looks different from training data
- My model's performance dropped from around 80% to just 31.5% on the Blender synthetic data
- **Solution**: Mixed-domain training - combined real and synthetic data
- This brought performance back up to 79.9% mAP50

### Challenge 3: Camera Calibration
- The system needs accurate calibration for both intrinsic and extrinsic camera parameters
- Currently requires checkerboard pattern and manual parameter tuning
- Takes about 30 minutes and needs technical expertise
- This is still a barrier for non-expert users to adopt the system
- It's an honest limitation I haven't fully solved yet

---

## Slide 5: Results (2 minutes)

### Left Side - 3D Accuracy (point to histogram):
- The histogram shows the distribution of position errors
- **Median vs Mean**: It's important to distinguish between these two metrics
  - Median is 3.8 cm - this means 50% of frames have errors below this value
  - Mean is 9.9 cm - this is the arithmetic average of all errors
  - The mean is higher because it's influenced by outliers - a few frames with large errors
  - Median is more representative of typical performance because it's not affected by outliers
- So the median of 3.8 cm tells us the typical accuracy - less than one-fifth of the ball's 21 cm diameter
- 95th percentile is 41 cm - meaning 95% of frames are within this accuracy
- This shows most frames achieve excellent centimeter-level accuracy

### Right Side - Performance Metrics (point to text):
- Processing speed: 9.7 frames per second on GTX 1650
- Suitable for post-match analysis workflows
- Success rate: 87% - the full pipeline successfully reconstructed 3D position in 87 out of 100 frames
- Detection metrics: 89.6% precision means high confidence when ball is detected
- 73% recall means the detector finds the ball in about 3 out of 4 frames

---

## Slide 6: Demo Video (4 minutes total: 3-4 min video + brief intro)

### Before playing video (30 seconds):
- Now I'll show you the system in action
- The video demonstrates loading footage, detecting the ball, and visualizing it in 3D
- You'll see the interactive features like rotation and trajectory tracking

### During video:
- (Stay quiet, let video play)
- (Optional: Point out key moments if needed)

### After video (15 seconds):
- As you can see, the system provides real-time visual feedback
- Coaches and players can use this to analyze ball trajectories and improve technique

---

## Slide 7: Limitations (1 minute)

### Validation:
- The system was validated only on Blender synthetic footage
- Real-world testing is the logical next step
- There's still some uncertainty about how well it performs on actual volleyball matches

### Hardware:
- Requires a discrete GPU for real-time processing
- Tested on GTX 1650 - represents mid-range consumer hardware
- Performance scales with GPU capability

### Analysis Type:
- Currently designed for post-match analysis only
- 9.7 FPS isn't fast enough for live tracking - would need at least 30 FPS
- With optimization or better hardware, live tracking could be possible in the future

---

## Slide 8: Summary (30 seconds)

### Key achievements:
- Successfully implemented 3D reconstruction for volleyball tracking using stereo vision
- Achieved centimeter-level accuracy with median error of 3.8 centimeters
- 87% pipeline success rate validated on 100 stereo frame pairs
- Demonstrated that structure from motion techniques can be applied effectively to fast-moving sports objects
- Hardware cost is significantly lower than commercial systems ($300-700 vs $60-70K)

### Impact:
- Provides an open-source framework for sports ball tracking research
- Demonstrates feasibility of stereo-based 3D reconstruction for volleyball analysis
- Opens possibilities for further research in real-time sports tracking

---

## Slide 9: Thank You (10 seconds)
- Thank you for your attention
- I'm happy to answer any questions you have

---

## Anticipated Questions & Answers

### Q: Why only 100 frames for validation?
**A:** "Good question. I validated on 100 stereo frame pairs because I needed ground truth data - the exact 3D position of the ball. Blender provides this accurately, which is impossible to get with real footage unless you have a professional system like Hawk-Eye. The 100 frames cover a complete rally with serves, volleys, and spikes, which gives good coverage of different scenarios."

### Q: How does this compare to Hawk-Eye accuracy?
**A:** "Hawk-Eye uses multi-camera triangulation with proprietary algorithms and achieves millimeter-level accuracy around 3-5 mm. My system achieves 3.8 cm median using two-camera stereo vision. The difference is expected given the hardware and methodology differences. For trajectory analysis and tactical feedback, centimeter-level accuracy provides sufficient precision."

### Q: Why not test on real volleyball footage?
**A:** "I didn't have access to stereo camera equipment, and even if I captured real footage, there's no way to know the exact 3D position of the ball for validation. Blender solves this by providing perfect ground truth while still capturing the visual complexity of volleyball - player occlusions, lighting changes, fast motion."

### Q: What's the main bottleneck for performance?
**A:** "Detection is the main bottleneck. Looking at my results, all 13 failures happened at the detection stage - zero failures at matching or triangulation. So improving YOLO detection speed and accuracy would directly improve the overall system."

### Q: Can this work for live matches?
**A:** "Not yet. At 9.7 FPS, it's too slow for live broadcast which needs 30+ FPS. However, for post-match analysis where you process footage after the game, 9.7 FPS is perfectly fine. With GPU optimization or a better graphics card, live speed could be achievable."

### Q: What about the calibration problem?
**A:** "That's an honest limitation. Right now calibration takes 30-60 minutes with a checkerboard pattern and manual tuning. It's a barrier for non-technical users. Future work could include an automated calibration wizard or using structure-from-motion to calibrate from the footage itself."

### Q: Did you consider using more than 2 cameras?
**A:** "More cameras would improve accuracy through multi-view triangulation and reduce occlusion issues. However, it also increases calibration complexity and computational requirements. Two cameras represents the minimal stereo configuration while still enabling 3D reconstruction. This is a standard approach in stereo vision research."
