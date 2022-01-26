
# Links
- <a href="https://linktr.ee/noahcabral">Linktree</a> 
- <a href="/#project-definition">Project Definition</a> 
- <a href="/#the-mission">Mission Statement</a> 
- <a href="/#original-timeline">Original Timeline</a> 
- <a href="/#currentrevised-timeline---20220126">Revised Timeline</a>
- <a href="/#todo">TODO</a>

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

The most up-to-date version of our project is <a href="/HandTracking.ipynb">this Jupyter Notebook</a>.

# What we have learned

- We work well in long in-person meetings that are somewhat unplanned.
- Consistency = 💯.
- Keep it simple.
- Tasks must be well-defined.
  - How (general method).
  - What (libraries used).
  - When (by when should it be done).
  - Why (larger picture of task in project).

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


# Current/Revised Timeline - 2022.01.26



# TODO
- All team to understand how the current categorical cross entropy loss function works as written in the code.
- Figure out why UNET architecture works, then try to get the old architecture working.
- Add onto network predictions of 21 keypoints + a second loss term.
- Swap from using RHD to Interhand.
- Add onto end of network (after keypoint prediction), MANO parameter estimation for hand mesh prediction, and use laplacian loss term maybe.
- Add a rending routine for MANO-defined hands. 
- Transition to using Google Cloud Storage as opposed to Google Drive for hosting the dataset.
  - This is so that everyone does not need to upload the dataset to their own drive.
- Figure out why we are using so much memory and techniques to reduce this.

# What we are Doing Right Now

## Noah
I am working on:
- Model architecture.

## Maddie
I am working on:
- Model Architecture.

## Lucas
I am working on:
- Investigating the Interhand https://mks0601.github.io/InterHand2.6M/ dataset. Then transitioning our project from RHD to this dataset.

## Oscar
I am working on:
- Implementing a rendering routine to render the 3D mesh of a hand from a MANO model.

## Max
I am working on:
- Model Architecture.


