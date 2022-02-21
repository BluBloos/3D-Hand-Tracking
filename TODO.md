
# Links
- <a href="https://linktr.ee/noahcabral">Linktree</a> 
- <a href="/TODO.md#project-definition">Project Definition</a> 
- <a href="/TODO.md#the-mission">Mission Statement</a> 
- <a href="/TODO.md#original-timeline">Original Timeline</a> 
- <a href="/TODO.md#currentrevised-timeline---20220126">Revised Timeline</a>
- <a href="/TODO.md#todo">TODO</a>

# Project Definition
- Generate a 3D mesh + 21 keypoints of two-hands through high hand-to-hand and hand-to-object contact. 
- Use a single camera.
- Why: VR/AR, Sign Language, Medical, Human-Computer interaction, etc... 
- **The motivation of this research is primarily an engineering one**, where the goal is to develop an intelligent system that solves a real problem better than all alternative approaches. 

# The Mission
- **Hold an interactive demo at <a href="https://cucai.ca/">CUCAI</a>** 
  - user has both their hands tracked in real-time by an off-the-shelf RGB camera. 
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
We keep a lab notebook of all our progress. 

<a href="https://queensuca.sharepoint.com/:w:/r/teams/GROUP-3DHandTrackingQMIND2021-2022/Shared%20Documents/General/Research%20Notebook.docx?d=w23ce3abb8be04355ae977d643496121b&csf=1&web=1&e=j8QDVn">Lab Notebook</a>

We didn't actually start this from the beginning of the project, so it may not include a complete and accurate picture of our efforts.

The most up-to-date version of our project is hosted in this Github repository. Access the primary Jupyter notebook <a href="src/HandTracking.ipynb">here</a>.

Our midterm project presentation can be accessed <a href="https://docs.google.com/presentation/d/1TA6MvU6VWnCx7TWTQFaYR-NhMFc4iGwGTj08Lq5pFxI/edit?usp=sharing">here</a> 

# What we have learned

- We work well in long in-person meetings that are somewhat unplanned.
- Consistency = 💯.
- Keep it simple.
- Tasks must be well-defined.
  - How (general method).
  - What (libraries used).
  - When (by when should it be done).
  - Why (larger picture of task in project).

# Current Timeline

NOTE: Schedule is to be modified as we continue to learn more \
about the exact specifics with the rest of the work to be done.

## Revised Scope
- No longer caring about the estimation under hand-to-object and hand-to-hand contact.
- And thus sticking with just the RHD (Rendered Hand Pose dataset).
- The concern is then, "How do we still do research?"
  - "One step at a time". If we can do the mimized scope sufficiently fast, we can look into doing more.
  - Asynchronously brainstorm answers to this question.
  - Asynchronously braintorm novel architectures to solve the problem as opposed to going with an existing implementation.
  - Keep on top of the our lab notebook.

## TODO
  
- Final goal is, implement the following model: <a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Ge_3D_Hand_Shape_and_Pose_Estimation_From_a_Single_RGB_CVPR_2019_paper.pdf">3D Hand Pose and Shape Estimation from a single RGB Image</a>
- There will be a 3-step plan in implementing this:
  - Step 1: Implement <a href="https://gmntu.github.io/mobilehand/.">MobileHand: Real-time 3D Hand Shape and Pose Estimation from Color Image</a>
  - Step 2: Add stacked hourglass to generate heatmaps as seen in both
      - <a href="https://arxiv.org/pdf/1902.09305.pdf">End-to-end Hand Mesh Recovery from a Monocular RGB Image</a>
      - <a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Ge_3D_Hand_Shape_and_Pose_Estimation_From_a_Single_RGB_CVPR_2019_paper.pdf">3D Hand Pose and Shape Estimation from a single RGB Image</a>
  - Step 3: Converge on the final model as seen in <a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Ge_3D_Hand_Shape_and_Pose_Estimation_From_a_Single_RGB_CVPR_2019_paper.pdf">3D Hand Pose and Shape Estimation from a single RGB Image</a>, or any other chosen paper.

- Maybe:
  - Understand the current architecture before "deleting" it
  - And figure out why our efforts prior to UNET used so much memory.
- Add a rending routine for MANO-defined hands (for the purpose of supporting the future demo).
  - This Github repo may be useful: https://github.com/ikalevatykh/mano_pybullet
- Transition to using Google Cloud Storage as opposed to Google Drive for hosting the dataset.
  - This is so that everyone does not need to upload the dataset to their own drive.

## February
- Complete step 1 and 2 of the plan.
- Potentially begin working on demo.
- Setup Google Cloud for dataset hosting.

## March

- Complete Step 3 of plan.
- Finalize paper.
- Finishing touches on demo.

## Early April
- ATTEND CUCAI AND FLEX 🔥🔥🔥🔥.


# Original Timeline

## October 
- Read, read, and read. 
- Complete mini projects corresponding to the topics that we are reading. Will help to solidify our understanding and give us concrete work to show off. 
- 1 on 1s. 
- Get into the “groove”. 
- 2 regular meetings / week. 
- Learn Git and software collaboration, if needed. 

## November 
- Work on preliminary models to solve the problem. 
- Just get the simplest thing possible that works. 

## December 
- Not much expected during exam season. 
- Would be more reasonable to expect activity by the team during the break, and might even be optimal due to having more time.  
- Overall no hard/soft goals for this month.  

## January 
- Brainstorm new solutions to the problem, building off the preliminary model. 
- We can think of this as the optimization phase. 

## February and Early March 
- Run experiments. Try to do something called an ablation study. Put our paper above the rest. 
- Finalize paper. 
- Prepare demo.
- ATTEND CUCAI AND FLEX 🔥🔥🔥🔥.
