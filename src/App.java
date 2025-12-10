import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class App {

    static final String CASCADE_FILE = "haarcascade_frontalface_default.xml";
    static final String DATA_DIR = "data";
    static final String PERSON_NAME = "Yash";
    static final String PERSON_DIR = DATA_DIR + File.separator + PERSON_NAME;
    static final Size FACE_SIZE = new Size(200, 200);   // normalized face size

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        System.out.println("Face Recognition - Console");
        System.out.println("1 = Detect faces (Mode 1)");
        System.out.println("2 = Capture images for training (Mode 2 - capture)");
        System.out.println("3 = Train and recognize (Mode 2 - simple recognizer)");
        System.out.print("Enter choice: ");

        try {
            int choice = System.in.read() - '0';
            if (choice == 1) {
                detectMode();
            } else if (choice == 2) {
                captureImages(30); // capture 30 images of Yash
            } else if (choice == 3) {
                trainAndRecognizeSimple();
            } else {
                System.out.println("Invalid choice");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void detectMode() {
        CascadeClassifier faceDetector = new CascadeClassifier(CASCADE_FILE);
        VideoCapture cam = new VideoCapture(0);
        if (!cam.isOpened()) {
            System.out.println("Camera not found");
            return;
        }

        Mat frame = new Mat();
        System.out.println("Waiting for a face... (press Ctrl+C in console to stop)");
        while (true) {
            if (!cam.read(frame)) continue;
            MatOfRect faces = new MatOfRect();
            faceDetector.detectMultiScale(frame, faces);

            if (faces.toArray().length > 0) {
                System.out.println("Face detected");
                break;
            }
        }
        cam.release();
    }

    public static void captureImages(int total) {
        new File(PERSON_DIR).mkdirs();

        CascadeClassifier faceDetector = new CascadeClassifier(CASCADE_FILE);
        VideoCapture cam = new VideoCapture(0);
        if (!cam.isOpened()) {
            System.out.println("Camera not found");
            return;
        }

        Mat frame = new Mat();
        int count = 0;
        System.out.println("Starting capture. Please look at the camera. Capturing " + total + " images.");
        while (count < total) {
            if (!cam.read(frame)) continue;

            MatOfRect faces = new MatOfRect();
            faceDetector.detectMultiScale(frame, faces);

            for (Rect r : faces.toArray()) {
                Mat face = new Mat(frame, r);

                // convert to gray & resize
                Imgproc.cvtColor(face, face, Imgproc.COLOR_BGR2GRAY);
                Mat resized = new Mat();
                Imgproc.resize(face, resized, FACE_SIZE);

                String filename = PERSON_DIR + File.separator + PERSON_NAME + "_" + count + ".png";
                Imgcodecs.imwrite(filename, resized);
                count++;
                System.out.println("Captured: " + filename);

                if (count >= total) break;
            }
        }
        cam.release();
        System.out.println("Capture complete. Images stored in: " + PERSON_DIR);
    }

    public static void trainAndRecognizeSimple() {
        // 1. Load training images of Yash
        List<Mat> images = new ArrayList<>();
        File dir = new File(PERSON_DIR);
        File[] files = dir.listFiles();
        if (files == null || files.length == 0) {
            System.out.println("No training images in " + PERSON_DIR + ". Run option 2 first.");
            return;
        }

        Arrays.sort(files);
        for (File f : files) {
            if (!f.getName().toLowerCase().endsWith(".png")) continue;
            Mat img = Imgcodecs.imread(f.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);
            if (img.empty()) continue;
            Imgproc.resize(img, img, FACE_SIZE);
            images.add(img);
        }
        if (images.size() == 0) {
            System.out.println("No valid images found for training.");
            return;
        }

        // 2. Compute mean face (average of all Yash images)
        Mat meanFace = computeMeanFace(images);
        Imgcodecs.imwrite("mean_face_yash.png", meanFace); // optional, just to see

        // 3. Start webcam & compare each detected face with mean face
        CascadeClassifier faceDetector = new CascadeClassifier(CASCADE_FILE);
        VideoCapture cam = new VideoCapture(0);
        if (!cam.isOpened()) {
            System.out.println("Camera not found");
            return;
        }

        Mat frame = new Mat();
        System.out.println("Recognition started. Look at the camera.");
        System.out.println("If average difference < threshold -> Recognized: Yash");

        double THRESHOLD = 45.0; // tune if needed

        while (true) {
            if (!cam.read(frame)) continue;

            MatOfRect faces = new MatOfRect();
            faceDetector.detectMultiScale(frame, faces);

            for (Rect r : faces.toArray()) {
                Mat roi = new Mat(frame, r);
                Imgproc.cvtColor(roi, roi, Imgproc.COLOR_BGR2GRAY);
                Mat resized = new Mat();
                Imgproc.resize(roi, resized, FACE_SIZE);

                double diff = averageAbsDifference(resized, meanFace);

                if (diff < THRESHOLD) {
                    System.out.println("Recognized: Yash (difference=" + diff + ")");
                } else {
                    System.out.println("Unknown person (difference=" + diff + ")");
                }
            }
        }
    }

    private static Mat computeMeanFace(List<Mat> images) {
        Mat mean = Mat.zeros((int) FACE_SIZE.height, (int) FACE_SIZE.width, CvType.CV_32F);

        for (Mat img : images) {
            Mat floatImg = new Mat();
            img.convertTo(floatImg, CvType.CV_32F);
            Core.add(mean, floatImg, mean);
        }

        Core.divide(mean, new Scalar(images.size()), mean);

        Mat meanU8 = new Mat();
        mean.convertTo(meanU8, CvType.CV_8U);
        return meanU8;
    }

    private static double averageAbsDifference(Mat a, Mat b) {
        Mat a32 = new Mat();
        Mat b32 = new Mat();
        a.convertTo(a32, CvType.CV_32F);
        b.convertTo(b32, CvType.CV_32F);

        Mat diff = new Mat();
        Core.absdiff(a32, b32, diff);
        Scalar mean = Core.mean(diff);
        return mean.val[0];
    }
}
