# QMIND2021-2022

**Team:**
- Noah Cabral
- Maddie Mackie
- Lucas Coster 
- Max Vincent 
- Oscar Lu


**<a style="text-align:left" href="https://qmind.ca/#Research">
DAIR @ QMIND
</a>**

Project to track the 3D shape and pose of two-hands through high hand-to-hand and hand-to-object contact, using a monocular RGB input.
See the project <a href="/TODO.md">roadmap</a> for anything regarding this project.

## Preliminary Results

Pictured from Left -> Right

Input image, Ground truth segmentation mask, Model prediction


<div style="display:flex; flex-direction:row;">


<img width="264" alt="Screen Shot 2021-12-08 at 4 51 07 PM" src="https://user-images.githubusercontent.com/38915815/145290279-1e4a2250-e7be-48fc-b3dc-30ecf2a63d03.png">
<img width="264" alt="Screen Shot 2021-12-08 at 4 51 15 PM" src="https://user-images.githubusercontent.com/38915815/145290290-48eac1cf-21da-481c-bd0d-6c582623b976.png">
<img width="264" alt="Screen Shot 2021-12-08 at 4 51 33 PM" src="https://user-images.githubusercontent.com/38915815/145290292-f546ce0f-7178-49d0-9504-8d227f0ebacc.png">

  </div>

## Running PyBullet Demo

- Ensure you are in the root project directoy.
- Make sure mano_v1_2 is in the root project as well (unzipped).
- Run (macOS)
```bash
./build.sh
``` 
OR (Windows)
```
./build.bat
```



