// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

// CrowdHuman detection - Single class (person) detection for crowded scenes
// Model output: [N, 65] where 65 = 64 (4x16 DFL bbox) + 1 (class score)
// Output is already processed with DFL and anchor points in the model

#include "yolo11.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <float.h>
#include <stdio.h>
#include <vector>

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

// Generate proposals for CrowdHuman model
// Input: pred - [N, 65] tensor where N = total anchors (8400 for 640x640)
//        pred contains already processed bbox coordinates and class scores
static void generate_proposals(const ncnn::Mat& pred, const ncnn::Mat& in_pad, 
                               float prob_threshold, std::vector<Object>& objects)
{
    const int w = in_pad.w;
    const int h = in_pad.h;
    
    const int num_anchors = pred.h;  // N = total anchors
    
    // For CrowdHuman, the output is already processed with DFL and anchor points
    // pred is shape [N, 65] where:
    //   - First 64 channels: 4 bbox coordinates * 16 DFL bins (already processed)
    //   - Last 1 channel: class confidence
    
    for (int i = 0; i < num_anchors; i++)
    {
        const float* pred_row = pred.row(i);
        
        // Extract class confidence (sigmoid already applied in model)
        float score = pred_row[64];  // Last channel
        
        if (score >= prob_threshold)
        {
            // Extract bbox coordinates (already decoded from DFL and anchor points)
            float x0 = pred_row[0];
            float y0 = pred_row[1];
            float x1 = pred_row[2];
            float y1 = pred_row[3];
            
            // Ensure box is valid (x1 > x0, y1 > y0)
            if (x1 <= x0 || y1 <= y0)
                continue;
            
            Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.label = 0;  // Person class
            obj.prob = score;
            
            objects.push_back(obj);
        }
    }
}

int YOLO11_crowdhuman::detect(const cv::Mat& rgb, std::vector<Object>& objects)
{
    const int target_size = det_target_size;  // 320, 480, or 640
    const float prob_threshold = 0.25f;       // Confidence threshold for crowded scenes
    const float nms_threshold = 0.45f;        // NMS threshold
    
    int img_w = rgb.cols;
    int img_h = rgb.rows;
    
    // CrowdHuman uses same stride pattern as YOLO detection
    const int max_stride = 32;
    
    // Letterbox pad to multiple of max_stride
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }
    
    // Convert to ncnn::Mat and resize
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, img_w, img_h, w, h);
    
    // Letterbox pad to target_size rectangle
    int wpad = (w + max_stride - 1) / max_stride * max_stride - w;
    int hpad = (h + max_stride - 1) / max_stride * max_stride - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, 
                           ncnn::BORDER_CONSTANT, 114.f);
    
    // Normalize
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);
    
    // Run inference
    ncnn::Extractor ex = yolo11.create_extractor();
    ex.input("in0", in_pad);
    
    ncnn::Mat out;
    ex.extract("out0", out);
    
    // Generate proposals
    std::vector<Object> proposals;
    generate_proposals(out, in_pad, prob_threshold, proposals);
    
    // Sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);
    
    // Apply NMS
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);
    
    int count = picked.size();
    
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];
        
        // Adjust coordinates back to original image size
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;
        
        // Clip to image boundaries
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);
        
        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }
    
    // Sort objects by area (largest first)
    struct
    {
        bool operator()(const Object& a, const Object& b) const
        {
            return a.rect.area() > b.rect.area();
        }
    } objects_area_greater;
    std::sort(objects.begin(), objects.end(), objects_area_greater);
    
    return 0;
}

int YOLO11_crowdhuman::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    static const char* class_names[] = {"person"};
    
    static cv::Scalar colors[] = {
        cv::Scalar( 67,  54, 244),
        cv::Scalar( 30,  99, 233),
        cv::Scalar( 39, 176, 156),
        cv::Scalar( 58, 183, 103),
        cv::Scalar( 81, 181,  63),
        cv::Scalar(150, 243,  33),
        cv::Scalar(169, 244,   3),
        cv::Scalar(188, 212,   0),
        cv::Scalar(150, 136,   0),
        cv::Scalar(175,  80,  76),
        cv::Scalar(195,  74, 139),
        cv::Scalar(220,  57, 205),
        cv::Scalar(235,  59, 255),
        cv::Scalar(193,   7, 255),
        cv::Scalar(152,   0, 255),
        cv::Scalar( 87,  34, 255),
        cv::Scalar( 85,  72, 121),
        cv::Scalar(158, 158, 158),
        cv::Scalar(125, 139,  96)
    };
    
    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
        const cv::Scalar& color = colors[i % 19];
        
        // Draw bounding box
        cv::rectangle(rgb, obj.rect, color, 2);
        
        // Draw label
        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);
        
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        
        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;
        
        // Background for text
        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), 
                      cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);
        
        // Text
        cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    
    return 0;
}