
# Links
- <a href="https://linktr.ee/noahcabral">Linktree</a> 
- <a href="/TODO.md#project-definition">Project Definition</a> 
- <a href="/TODO.md#the-mission">Mission Statement</a> 
- <a href="/TODO.md#original-timeline">Original Timeline</a> 
- <a href="/TODO.md#current-timeline">Revised Timeline</a>
- <a href="/TODO.md#todo">TODO</a>

# Project Definition
- Generate a 3D mesh + 21 keypoints of two-hands through high hand-to-hand and hand-to-object contact. 
- Use a single camera.
- Why: VR/AR, Sign Language, Medical, Human-Computer interaction, etc... 
- **The motivation of this research is primarily an engineering one**, where the goal is to develop an intelligent system that solves a real-world problem better than all alternative approaches. 

# The Mission
- **Hold an interactive demo at <a href="https://cucai.ca/">CUCAI</a>** 
  - User has both their hands tracked in real-time by an off-the-shelf RGB camera. 
  - Results of tracking appear on a monitor in front of them, and the camera will be positioned face down towards the floor. 
  - As a stretch goal, the demo might include a physics-based interaction of the hands with virtual objects.
- **MVP**
  - Produce two 3D meshes of the hands using a single RGB camera. 
  - No requirement for real-time performance. 
  - Along with the 3D meshes, output the 3D joint locations.
  - Mesh articulations are invariant to occlusions from external objects as well as those produced by inter- and intra-hand interactions. 
  - The accuracy when the hands are occluded is not important, but from a qualitative perspective the tracking should not fail catastrophically. 
  - The accuracy of the hands when not under occlusion should compare to state-of-the-art methods.  
- **The Dream 😍**
  - CUCAI demo is in realtime as opposed to, for example, a sparse 5 FPS.     

# Progress Thus Far

The most up-to-date version of our project is hosted in this Github repository. Access the primary Jupyter notebook <a href="src/HandTracking.ipynb">here</a>.

Our midterm project presentation can be accessed <a href="https://docs.google.com/presentation/d/1TA6MvU6VWnCx7TWTQFaYR-NhMFc4iGwGTj08Lq5pFxI/edit?usp=sharing">here</a>.

We also keep a lab notebook of all our progress. 

<a href="https://queensuca.sharepoint.com/:w:/r/teams/GROUP-3DHandTrackingQMIND2021-2022/Shared%20Documents/General/Research%20Notebook.docx?d=w23ce3abb8be04355ae977d643496121b&csf=1&web=1&e=j8QDVn">Lab Notebook</a>

# Revised Scope
- No longer caring about the estimation under hand-to-object and hand-to-hand contact.
- And thus sticking with just the RHD (Rendered Hand Pose dataset).
- The concern is then, "How do we still do research?"
  - "One step at a time". If we can do the mimized scope sufficiently fast, we can look into doing more.
  - Asynchronously brainstorm answers to this question.
  - Asynchronously braintorm novel architectures to solve the problem as opposed to going with an existing implementation.
  - Keep on top of the our lab notebook.

# TODO
  
The final goal is to implement the following model: <a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Ge_3D_Hand_Shape_and_Pose_Estimation_From_a_Single_RGB_CVPR_2019_paper.pdf">3D Hand Pose and Shape Estimation from a single RGB Image</a>

There will be a 3-step plan in implementing this:
- Step 1: Implement <a href="https://gmntu.github.io/mobilehand/.">MobileHand: Real-time 3D Hand Shape and Pose Estimation from Color Image</a>
- Step 2: Add stacked hourglass to generate heatmaps as seen in both
    - <a href="https://arxiv.org/pdf/1902.09305.pdf">End-to-end Hand Mesh Recovery from a Monocular RGB Image</a>
    - <a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Ge_3D_Hand_Shape_and_Pose_Estimation_From_a_Single_RGB_CVPR_2019_paper.pdf">3D Hand Pose and Shape Estimation from a single RGB Image</a>
- Step 3: Converge on the final model as seen in <a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Ge_3D_Hand_Shape_and_Pose_Estimation_From_a_Single_RGB_CVPR_2019_paper.pdf">3D Hand Pose and Shape Estimation from a single RGB Image</a>, or any other chosen paper.

## Tasks

Here is a list of TODO as far as we can see ahead of us.

- Complete the implementation of the MobileHand MANO layer through a place of deep understanding. 
- Reconfigure the training loop to stream in the training data from a remote server, enabling the use of the entire dataset while keeping RAM usage low. 
