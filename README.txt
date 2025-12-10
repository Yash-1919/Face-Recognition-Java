
Face Recognition (Console) - Project structure

After extracting this ZIP, your folder should contain:

 - lib/      (already present, put opencv-4120.jar here)
 - data/     (already present, will hold captured face images)
 - resources/
 - src/
 - README.txt

Next steps (short):

1) Copy opencv-4120.jar into lib/
2) Copy opencv_java4120.dll into the project root (same level as lib/src/data)
3) Copy haarcascade_frontalface_default.xml into the project root
4) In IntelliJ:
   - Add lib/opencv-4120.jar as a Module dependency
   - Set VM options: -Djava.library.path=PATH_TO_PROJECT_ROOT
5) Run App.main, choose:
   - 1 for detection
   - 2 for capture
   - 3 for train + recognize
