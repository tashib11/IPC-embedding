#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <windows.h>
#include <stdio.h>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace cv::dnn;

static const char* SHM_NAME = "Local\\YoloIPC_SharedMemory";

#define INPUT_WIDTH 640
#define INPUT_HEIGHT 640
#define CHANNELS 3
#define IMAGE_SIZE (INPUT_WIDTH * INPUT_HEIGHT * CHANNELS)
#define SHM_SIZE (IMAGE_SIZE + 1)

#define CONF_THRESHOLD 0.45f
#define SCORE_THRESHOLD 0.45f
#define NMS_THRESHOLD 0.50f
#define CLASS_COUNT 80

static const char* CLASS_NAMES[CLASS_COUNT] = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
};

int main() {
    int use_webcam = 1;
    char choice_str[10];
    char image_path[260] = {0};

    printf("YOLO IPC PRODUCER (OpenCV DNN)\n");
    printf("1. Webcam\n");
    printf("2. Image File\n");
    printf("Choice: ");
    fgets(choice_str, sizeof(choice_str), stdin);
    if (choice_str[0] == '2') use_webcam = 0;

    VideoCapture cap;
    Mat image_input;

    if (use_webcam) {
        cap.open(0);
        if (!cap.isOpened()) {
            printf("Error: Cannot open webcam\n");
            return -1;
        }
    } else {
        printf("Image Path: ");
        fgets(image_path, sizeof(image_path), stdin);
        size_t len = strlen(image_path);
        if (len && image_path[len-1] == '\n') image_path[len-1] = '\0';
        image_input = imread(image_path);
        if (image_input.empty()) {
            printf("Error: Cannot read image\n");
            return -1;
        }
    }

    // --- Load YOLOv5 ONNX ---
    Net net = readNetFromONNX("yolov5s.onnx");
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // --- Shared memory ---
    HANDLE hMapFile = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, SHM_SIZE, SHM_NAME);
    if (!hMapFile) {
        printf("Error: CreateFileMappingA failed\n");
        return -1;
    }

    unsigned char* shm_ptr = (unsigned char*)MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, SHM_SIZE);
    if (!shm_ptr) {
        printf("Error: MapViewOfFile failed\n");
        CloseHandle(hMapFile);
        return -1;
    }

    shm_ptr[0] = 0;
    printf("Producer running. Press 'q' to quit.\n");

    Mat frame, blob, out, resized_frame, rgb_frame;

    while (true) {
        if (use_webcam) {
            cap >> frame;
            if (frame.empty()) break;
        } else {
            frame = image_input.clone();
            Sleep(33);
        }

        // Create blob
        blobFromImage(frame, blob, 1.0/255.0, Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false, CV_32F);
        net.setInput(blob);
        out = net.forward(); // YOLOv5: [1,25200,85] usually

        float x_factor = (float)frame.cols / INPUT_WIDTH;
        float y_factor = (float)frame.rows / INPUT_HEIGHT;

        float* data = (float*)out.data;
        int rows = out.size[1];

        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<Rect> boxes;

        for (int i = 0; i < rows; i++) {
            float confidence = data[4]; // objectness
            if (confidence >= CONF_THRESHOLD) {
                float* classes_scores = data + 5;
                Mat scores(1, CLASS_COUNT, CV_32FC1, classes_scores);
                Point class_id_point;
                double max_class_score;
                minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);

                if (max_class_score >= SCORE_THRESHOLD) {
                    float cx = data[0];
                    float cy = data[1];
                    float w  = data[2];
                    float h  = data[3];

                    int left = int((cx - 0.5f*w) * x_factor);
                    int top  = int((cy - 0.5f*h) * y_factor);
                    int width  = int(w * x_factor);
                    int height = int(h * y_factor);

                    class_ids.push_back(class_id_point.x);
                    confidences.push_back((float)(confidence * max_class_score));
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
            data += 85;
        }

        std::vector<int> nms_result;
        NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);

        // draw detections
        for (int idx : nms_result) {
            Rect box = boxes[idx];
            int class_id = class_ids[idx];
            rectangle(frame, box, Scalar(0,255,0), 2);
            char txt[80];
            sprintf(txt, "%s %.2f", CLASS_NAMES[class_id], confidences[idx]);
            putText(frame, txt, Point(box.x, std::max(0, box.y-5)),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,0,0), 2);
        }

        imshow("Producer", frame);

        // send resized RGB frame to shared memory
        resize(frame, resized_frame, Size(INPUT_WIDTH, INPUT_HEIGHT));
        cvtColor(resized_frame, rgb_frame, COLOR_BGR2RGB);
        if (!rgb_frame.isContinuous()) rgb_frame = rgb_frame.clone();

        while (shm_ptr[0] != 0) Sleep(1);
        memcpy(shm_ptr + 1, rgb_frame.data, IMAGE_SIZE);
        shm_ptr[0] = 1;

        if (waitKey(1) == 'q') break;
    }

    UnmapViewOfFile(shm_ptr);
    CloseHandle(hMapFile);
    return 0;
}
